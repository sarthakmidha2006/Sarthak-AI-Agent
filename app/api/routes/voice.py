"""Voice channel route -- ``POST /voice``.

The voice channel is a thin Speech-to-Text / Text-to-Speech wrapper around the
*identical* shared brain used by the chat channel (spec section 0). A single
endpoint accepts two request shapes, distinguished by the ``Content-Type``
header:

* ``multipart/form-data`` with an ``audio`` file field (and optional ``session_id``
  / ``speak`` form fields). The audio is transcribed via
  :meth:`app.brain.llm.LLMClient.transcribe` (Whisper) before hitting the brain.
* ``application/json`` matching :class:`~app.models.api_schemas.VoiceTextRequest`
  -- a pre-transcribed / typed message, useful for testing and text-first voice
  clients.

After the brain answers on the ``"voice"`` channel, if ``speak`` is truthy the
answer is synthesised to MP3 via :meth:`app.brain.llm.LLMClient.synthesize` and
returned as base64 in :class:`~app.models.api_schemas.VoiceResponse`.

This route requires ``python-multipart`` to be installed (it is parsing
multipart form bodies).
"""

from __future__ import annotations

import base64
import logging
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import ValidationError

from app.api.deps import get_brain, get_llm
from app.models.api_schemas import VoiceResponse, VoiceTextRequest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.brain.llm import LLMClient
    from app.brain.persona import BrainResponse, PersonaBrain

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voice"])

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


def _coerce_speak(value: str | None, *, default: bool = True) -> bool:
    """Interpret a form field as a boolean, defaulting when absent/blank."""

    if value is None:
        return default
    normalized = value.strip().lower()
    if not normalized:
        return default
    return normalized in _TRUTHY


async def _parse_request(
    request: Request,
    llm: "LLMClient",
) -> tuple[str, str | None, str | None, bool]:
    """Resolve the inbound request into ``(text, transcript, session_id, speak)``.

    Supports both the multipart-audio and JSON-text shapes. Audio uploads are
    transcribed here so the brain only ever sees text.
    """

    content_type = (request.headers.get("content-type") or "").lower()

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("audio")
        session_id = form.get("session_id") or None
        speak = _coerce_speak(form.get("speak"))

        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="multipart/form-data requests must include an 'audio' file field.",
            )

        audio_bytes = await upload.read()
        if not audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Uploaded 'audio' file is empty.",
            )

        filename = getattr(upload, "filename", None) or "audio.wav"
        try:
            transcript = await llm.transcribe(audio_bytes, filename=filename)
        except Exception as exc:  # noqa: BLE001 - convert to a clean gateway error
            logger.exception("Speech-to-text transcription failed")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to transcribe audio.",
            ) from exc

        text = (transcript or "").strip()
        if not text:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Transcription produced no text from the supplied audio.",
            )
        # session_id may be a non-str form value in exotic clients; coerce defensively.
        return text, transcript, (str(session_id) if session_id is not None else None), speak

    if content_type.startswith("application/json"):
        try:
            raw = await request.json()
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Request body is not valid JSON.",
            ) from exc
        try:
            parsed = VoiceTextRequest.model_validate(raw)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=exc.errors(),
            ) from exc
        return parsed.message, None, parsed.session_id, parsed.speak

    raise HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail=(
            "Unsupported Content-Type. Use 'multipart/form-data' with an 'audio' "
            "field or 'application/json' with a text message."
        ),
    )


def _to_voice_response(
    result: "BrainResponse",
    *,
    session_id: str,
    transcript: str | None,
    audio_base64: str | None,
) -> VoiceResponse:
    """Map a :class:`~app.brain.persona.BrainResponse` to a :class:`VoiceResponse`."""

    return VoiceResponse(
        answer=result.answer,
        session_id=session_id,
        transcript=transcript,
        audio_base64=audio_base64,
        audio_format="mp3",
        citations=result.citations,
        tool_calls=result.tool_calls,
        injection_flagged=result.injection_flagged,
        latency_ms=result.latency_ms,
    )


@router.post("/voice", response_model=VoiceResponse, summary="Talk to the persona")
async def voice(
    request: Request,
    brain: "PersonaBrain" = Depends(get_brain),
    llm: "LLMClient" = Depends(get_llm),
) -> VoiceResponse:
    """Run a voice turn: STT (if audio) -> shared brain -> optional TTS.

    The brain runs on the ``"voice"`` channel so prompts are tuned for spoken,
    concise answers. When ``speak`` is requested the answer is synthesised to MP3
    and returned as base64; TTS failures degrade gracefully to a text-only
    response rather than failing the whole turn.
    """

    text, transcript, requested_session_id, speak = await _parse_request(request, llm)

    session_id = requested_session_id or uuid.uuid4().hex

    try:
        result = await brain.answer(
            text,
            channel="voice",
            history=None,
            conversation_id=session_id,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean 502 to the client
        logger.exception("Voice request failed for session %s", session_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate a response.",
        ) from exc

    resolved_session_id = result.conversation_id or session_id

    audio_base64: str | None = None
    if speak and result.answer.strip():
        try:
            audio_bytes = await llm.synthesize(result.answer)
            audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
        except Exception:  # noqa: BLE001 - TTS is best-effort; keep the text answer
            logger.exception(
                "Text-to-speech synthesis failed for session %s; returning text only",
                resolved_session_id,
            )
            audio_base64 = None

    return _to_voice_response(
        result,
        session_id=resolved_session_id,
        transcript=transcript,
        audio_base64=audio_base64,
    )
