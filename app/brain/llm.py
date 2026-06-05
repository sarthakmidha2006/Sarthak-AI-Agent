"""Async LLM client wrapper used by the entire brain (spec §8).

This is the single backend boundary for the brain. It exposes four async
primitives used across the system, each now backed by a **free / local** stack:

* :meth:`LLMClient.chat` — chat completions with optional tool calling, served
  by **Groq** through its OpenAI-compatible API (the ``openai`` SDK is reused
  with a custom ``base_url``). The response message (including any
  ``tool_calls``) is converted to a plain ``dict`` so the rest of the codebase
  never depends on SDK object types.
* :meth:`LLMClient.embed` — text embeddings produced **locally** via
  ``sentence-transformers`` (``BAAI/bge-small-en-v1.5``, 384-dim, L2-normalised
  for cosine). No network, no quota.
* :meth:`LLMClient.transcribe` — speech-to-text via **Groq Whisper**
  (``whisper-large-v3``), reusing the same OpenAI-compatible client.
* :meth:`LLMClient.synthesize` — text-to-speech via **local Piper**, encoded to
  MP3 (``lameenc``) so the ``/voice`` response contract is unchanged.

Network calls (chat, transcribe) are wrapped with a :mod:`tenacity` retry policy
using exponential backoff; the attempt budget derives from
``settings.openai_max_retries``. Local calls (embed, synthesize) run in a worker
thread via :func:`asyncio.to_thread` so they never block the event loop. Heavy
optional dependencies (torch/sentence-transformers, piper, lameenc) are imported
lazily inside their methods to keep module import (and test collection) light.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings

logger = logging.getLogger(__name__)

# Exceptions that are worth retrying: transient transport/server/rate-limit errors.
# Deterministic failures (bad request, auth) are *not* retried so they surface fast.
_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


@lru_cache(maxsize=4)
def _load_sentence_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading local embedding model: %s", model_name)
        return SentenceTransformer(model_name)
    except Exception as e:
        logger.warning("Embeddings disabled: %s", e)
        return None

@lru_cache(maxsize=2)
def _load_piper_voice(model_path: str) -> Any:
    """Load (and process-cache) a Piper ONNX voice from ``model_path``.

    The matching ``<model_path>.json`` config is auto-discovered by Piper when
    it sits beside the ONNX file. Cached so the ~63 MB model loads once.
    """
    from pathlib import Path

    from piper import PiperVoice

    logger.info("Loading local Piper TTS voice: %s", model_path)
    return PiperVoice.load(Path(model_path))


@dataclass
class ChatResult:
    """Outcome of a single chat completion call.

    Attributes:
        message: The raw assistant message as a plain ``dict`` with ``role``,
            ``content`` (possibly ``None``), and an optional ``tool_calls`` list.
            Each tool call is ``{"id", "type", "function": {"name", "arguments"}}``.
        finish_reason: The completion's finish reason (e.g. ``"stop"`` or
            ``"tool_calls"``).
        prompt_tokens: Prompt token count if reported by the API, else ``None``.
        completion_tokens: Completion token count if reported, else ``None``.
    """

    message: dict
    finish_reason: str
    prompt_tokens: int | None
    completion_tokens: int | None


class LLMClient:
    """Thin async wrapper exposing chat/embed/transcribe/synthesize.

    Chat generation goes to **OpenRouter** and speech-to-text to **Groq Whisper**
    — both via the OpenAI-compatible SDK (custom ``base_url`` + key) — while
    embeddings and TTS run **locally**. Each provider has its own lazily-built
    client. The client is configured entirely from :class:`app.config.Settings`;
    no secrets or model names are hardcoded.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise the wrapper; provider clients are built lazily on use.

        Args:
            settings: Application settings carrying the chat/STT API keys, base
                URLs, model names, request timeout, and retry budget.
        """
        self._settings = settings
        # Provider clients are built lazily so purely local operations
        # (embeddings, Piper TTS) need no API key at all — ingestion runs offline.
        self._chat_client: AsyncOpenAI | None = None  # OpenRouter (chat)
        self._stt_client: AsyncOpenAI | None = None  # Groq (Whisper STT)
        # tenacity expects at least one attempt; clamp the configured budget.
        self._attempts = max(1, int(settings.openai_max_retries) + 1)

    def _get_chat_client(self) -> AsyncOpenAI:
        """Lazily construct the OpenRouter-backed chat client.

        Uses the OpenAI SDK pointed at OpenRouter's OpenAI-compatible endpoint.
        OpenRouter returns OpenAI-shaped responses and errors, so the tenacity
        retry policy and ``_RETRYABLE_EXCEPTIONS`` apply unchanged. The SDK's own
        ``max_retries`` is disabled so retries are driven uniformly here. The
        optional ``HTTP-Referer`` / ``X-Title`` ranking headers are sent only
        when configured.
        """
        if self._chat_client is None:
            headers: dict[str, str] = {}
            if self._settings.openrouter_http_referer:
                headers["HTTP-Referer"] = self._settings.openrouter_http_referer
            if self._settings.openrouter_app_title:
                headers["X-Title"] = self._settings.openrouter_app_title
            self._chat_client = AsyncOpenAI(
                api_key=self._settings.openrouter_api_key or self._settings.openai_api_key or None,
                base_url=self._settings.chat_base_url,
                timeout=self._settings.openai_request_timeout,
                max_retries=0,
                default_headers=headers or None,
            )
        return self._chat_client

    def _get_stt_client(self) -> AsyncOpenAI:
        """Lazily construct the Groq-backed STT client (Whisper).

        Kept separate from the chat client because OpenRouter has no
        audio/transcriptions endpoint; STT stays on Groq so the voice pipeline is
        unchanged.
        """
        if self._stt_client is None:
            self._stt_client = AsyncOpenAI(
                api_key=self._settings.groq_api_key or self._settings.openai_api_key or None,
                base_url=self._settings.groq_base_url,
                timeout=self._settings.openai_request_timeout,
                max_retries=0,
            )
        return self._stt_client

    def _retrying(self) -> AsyncRetrying:
        """Build a fresh :class:`AsyncRetrying` controller for one logical call.

        A new controller is created per call so retry state never leaks between
        concurrent invocations.
        """
        return AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(self._attempts),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=20.0),
            retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        )

    # ------------------------------------------------------------------ chat
    async def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        temperature: float | None = None,
        response_format: dict | None = None,
    ) -> ChatResult:
        """Run a chat completion and return a normalized :class:`ChatResult`.

        Args:
            messages: OpenAI chat messages (already-built plain dicts).
            tools: Optional OpenAI ``tools`` array enabling function calling.
            tool_choice: Tool-choice strategy; only forwarded when ``tools`` is set.
            temperature: Sampling temperature; falls back to
                ``settings.chat_temperature`` when ``None``.
            response_format: Optional response-format directive
                (e.g. ``{"type": "json_object"}``).

        Returns:
            A :class:`ChatResult` whose ``message`` is a plain dict including any
            ``tool_calls`` converted from SDK objects.

        Raises:
            openai.OpenAIError: Propagated after retries are exhausted.
        """
        params: dict[str, Any] = {
            "model": self._settings.openai_chat_model,
            "messages": messages,
            "temperature": (
                temperature if temperature is not None else self._settings.chat_temperature
            ),
        }
        # Bound completion length: caps cost and keeps OpenRouter's pre-flight
        # credit check within budget (it otherwise reserves the model's full
        # output window). Omitted when set to 0.
        if int(self._settings.chat_max_tokens) > 0:
            params["max_tokens"] = int(self._settings.chat_max_tokens)
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        if response_format is not None:
            params["response_format"] = response_format

        client = self._get_chat_client()
        start = time.perf_counter()
        async for attempt in self._retrying():
            with attempt:
                completion = await client.chat.completions.create(**params)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        choice = completion.choices[0]
        message = self._message_to_dict(choice.message)
        finish_reason = choice.finish_reason or "stop"

        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

        logger.debug(
            "chat completion model=%s finish=%s latency_ms=%.1f prompt_tokens=%s "
            "completion_tokens=%s tool_calls=%d",
            self._settings.openai_chat_model,
            finish_reason,
            elapsed_ms,
            prompt_tokens,
            completion_tokens,
            len(message.get("tool_calls") or []),
        )
        return ChatResult(
            message=message,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @staticmethod
    def _message_to_dict(message: Any) -> dict:
        """Convert an SDK assistant message object into a plain serializable dict.

        Only the fields the brain's tool loop needs are emitted: ``role``,
        ``content``, and ``tool_calls`` (when present). Each tool call carries its
        ``id``, ``type``, and nested ``function`` with ``name`` and ``arguments``
        (a JSON string exactly as returned by the API).

        Args:
            message: The SDK ``ChatCompletionMessage`` object.

        Returns:
            A plain ``dict`` suitable for re-sending as a chat message and for
            persistence/logging.
        """
        result: dict[str, Any] = {
            "role": getattr(message, "role", "assistant") or "assistant",
            "content": getattr(message, "content", None),
        }

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            converted: list[dict[str, Any]] = []
            for call in tool_calls:
                function = getattr(call, "function", None)
                converted.append(
                    {
                        "id": getattr(call, "id", None),
                        "type": getattr(call, "type", "function") or "function",
                        "function": {
                            "name": getattr(function, "name", "") if function else "",
                            "arguments": getattr(function, "arguments", "") if function else "",
                        },
                    }
                )
            result["tool_calls"] = converted

        return result

    # ------------------------------------------------------------- embeddings
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` in order using a local sentence-transformers model.

        Runs the (blocking, CPU-bound) encoder in a worker thread so the event
        loop stays responsive. Vectors are L2-normalised so cosine distance in
        ChromaDB behaves correctly. Order is preserved 1:1 with ``texts``.

        Args:
            texts: Input strings to embed.

        Returns:
            A list of embedding vectors aligned 1:1 with ``texts``.

        Raises:
            RuntimeError: If the encoder returns the wrong number of vectors.
        """
        if not texts:
            return []

        batch_size = max(1, int(self._settings.embedding_batch_size))
        model_name = self._settings.openai_embedding_model
        start = time.perf_counter()

        vectors = await asyncio.to_thread(self._encode_sync, texts, model_name, batch_size)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.debug(
            "embedded %d texts model=%s batch_size=%d latency_ms=%.1f",
            len(texts),
            model_name,
            batch_size,
            elapsed_ms,
        )
        if len(vectors) != len(texts):
            # Should never happen, but guard the alignment contract loudly.
            raise RuntimeError(
                f"embedding count mismatch: got {len(vectors)} for {len(texts)} inputs"
            )
        return vectors

    @staticmethod
    def _encode_sync(texts: list[str], model_name: str, batch_size: int) -> list[list[float]]:
        """Blocking encode helper run inside :func:`asyncio.to_thread`."""
        model = _load_sentence_model(model_name)
        array = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return array.tolist()

    # ----------------------------------------------------------- transcription
    async def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """Transcribe audio to text via the configured STT model.

        Args:
            audio_bytes: Raw audio file bytes (e.g. wav/mp3/m4a).
            filename: A filename hint; its extension helps the API detect format.

        Returns:
            The transcribed text (stripped). Empty string if nothing was returned.

        Raises:
            openai.OpenAIError: Propagated after retries are exhausted.
        """
        client = self._get_stt_client()
        start = time.perf_counter()
        async for attempt in self._retrying():
            with attempt:
                transcription = await client.audio.transcriptions.create(
                    model=self._settings.openai_stt_model,
                    file=(filename, audio_bytes),
                )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        text = (getattr(transcription, "text", "") or "").strip()
        logger.debug(
            "transcribed %d bytes model=%s chars=%d latency_ms=%.1f",
            len(audio_bytes),
            self._settings.openai_stt_model,
            len(text),
            elapsed_ms,
        )
        return text

    # ------------------------------------------------------------- synthesis
    async def synthesize(self, text: str) -> bytes:
        """Synthesize ``text`` to speech locally (Piper) and return MP3 bytes.

        Piper produces 16-bit PCM offline; it is encoded to mono MP3 with
        ``lameenc`` so the ``/voice`` response contract (``audio_format="mp3"``)
        is preserved. The blocking work runs in a worker thread.

        Args:
            text: The text to speak.

        Returns:
            MP3 audio bytes. Empty input returns empty bytes without any work.

        Raises:
            RuntimeError: If the local TTS stack fails to produce audio.
        """
        if not text or not text.strip():
            return b""

        start = time.perf_counter()
        audio = await asyncio.to_thread(
            self._synthesize_sync,
            text,
            self._settings.piper_model_path,
            float(self._settings.piper_length_scale),
            int(self._settings.tts_mp3_bitrate),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        logger.debug(
            "synthesized %d chars engine=piper model=%s bytes=%d latency_ms=%.1f",
            len(text),
            self._settings.openai_tts_model,
            len(audio),
            elapsed_ms,
        )
        return audio

    @staticmethod
    def _synthesize_sync(
        text: str, model_path: str, length_scale: float, bitrate_kbps: int
    ) -> bytes:
        """Blocking Piper synth + MP3 encode, run inside :func:`asyncio.to_thread`."""
        import lameenc
        from piper import SynthesisConfig

        voice = _load_piper_voice(model_path)
        syn_config = SynthesisConfig(length_scale=length_scale)

        pcm = bytearray()
        for chunk in voice.synthesize(text, syn_config=syn_config):
            pcm += chunk.audio_int16_bytes

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(bitrate_kbps)
        encoder.set_in_sample_rate(int(voice.config.sample_rate))
        encoder.set_channels(1)  # Piper voices are mono
        encoder.set_quality(5)  # 2=best ... 7=fastest; 5 is a good middle ground
        mp3 = encoder.encode(bytes(pcm)) + encoder.flush()
        if not mp3:
            raise RuntimeError("Piper/lameenc produced no audio bytes")
        return bytes(mp3)

    async def aclose(self) -> None:
        """Close the underlying async HTTP clients. Safe to call multiple times."""
        for client in (self._chat_client, self._stt_client):
            if client is None:
                continue
            try:
                await client.close()
            except Exception:  # pragma: no cover - defensive cleanup
                logger.debug("error while closing async client", exc_info=True)
