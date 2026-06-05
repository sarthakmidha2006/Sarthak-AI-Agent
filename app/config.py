"""Application configuration.

Defines the :class:`Settings` model (backed by ``pydantic-settings``) and a
cached :func:`get_settings` accessor. All runtime configuration — OpenAI
credentials/models, RAG parameters, scheduling rules, persona identity, the
database URL, and security toggles — is sourced exclusively from here so that
no secret or tunable is ever hard-coded elsewhere in the codebase.

This module sits at the bottom of the dependency graph (spec §20): it imports
nothing from the rest of the ``app`` package.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from the environment / ``.env``.

    Field defaults match the authoritative build spec (§3). Environment
    variable names are the upper/lower-case field names (matching is
    case-insensitive). Unknown environment variables are ignored so that an
    operator can keep unrelated values in their ``.env`` file without breaking
    startup.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Chat provider (OpenRouter, OpenAI-compatible API) ---
    # Chat generation goes through OpenRouter via the OpenAI SDK (a custom
    # ``base_url`` + key). Switching the chat model later requires changing only
    # ``OPENAI_CHAT_MODEL`` in ``.env`` (e.g. "anthropic/claude-3.5-sonnet");
    # switching the whole provider means changing ``CHAT_BASE_URL`` + the key.
    chat_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str = ""  # chat credential (env: OPENROUTER_API_KEY)
    # OpenRouter model id (note the provider prefix). Switch models via .env only.
    openai_chat_model: str = "openai/gpt-4.1-mini"
    # Optional OpenRouter ranking headers (sent only when non-empty).
    openrouter_http_referer: str = ""  # env: OPENROUTER_HTTP_REFERER
    openrouter_app_title: str = ""  # env: OPENROUTER_APP_TITLE
    # Cap on completion tokens per chat call. Sent as ``max_tokens``; bounds cost
    # and, on OpenRouter, keeps the pre-flight credit/affordability check within a
    # low-credit account's budget (it otherwise reserves the model's full output
    # window). Persona answers are short, so 1024 is ample. Set 0 to omit.
    chat_max_tokens: int = 1024

    # --- Speech-to-text provider (Groq Whisper) ---
    # STT stays on Groq because OpenRouter has no audio/transcriptions endpoint.
    # This keeps the voice pipeline unchanged. ``openai_api_key`` is retained as a
    # legacy/test fixture key and an optional STT fallback credential.
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    openai_api_key: str = ""  # retained: legacy / test fixture key
    openai_stt_model: str = "whisper-large-v3"  # Groq Whisper STT

    # --- Embeddings (local sentence-transformers) ---
    openai_embedding_model: str = "BAAI/bge-small-en-v1.5"  # local sentence-transformers id

    # --- Shared HTTP behaviour (both providers) ---
    openai_request_timeout: float = 60.0
    openai_max_retries: int = 4

    # --- Local TTS (Piper) ---
    # Piper runs fully offline. ``synthesize`` loads this ONNX voice, produces
    # 16-bit PCM, and encodes MP3 (via lameenc) so the /voice contract is
    # unchanged (audio_format="mp3"). The matching ``<path>.json`` config must
    # sit next to the ONNX file.
    piper_model_path: str = "./data/piper/en_US-lessac-medium.onnx"
    piper_length_scale: float = 1.0  # >1 slower, <1 faster
    tts_mp3_bitrate: int = 64  # kbps for the mono MP3 output
    # Reported by the /health route's "tts" field; describes the local engine.
    openai_tts_model: str = "piper-en_US-lessac-medium"

    # --- Reranker ---
    # "none"  -> skip reranking; use fused RRF order directly (zero LLM calls).
    # "llm"   -> score candidates with a Groq JSON completion (token-heavy).
    # "cohere"-> use Cohere's dedicated rerank endpoint (needs cohere_api_key).
    # Default is "none": the LLM reranker is the single largest Groq token sink
    # (~4.3k tokens/query) and RRF fusion alone is a solid ranking for this
    # corpus. Set RERANKER_PROVIDER=llm to restore quality reranking.
    reranker_provider: str = "none"  # "none" | "llm" | "cohere"
    reranker_model: str = "llama-3.1-8b-instant"  # used when provider == "llm" (Groq)
    cohere_api_key: str | None = None
    cohere_rerank_model: str = "rerank-english-v3.0"

    # --- Markdown ingestion ---
    # Directory scanned (recursively) for hand-authored ``*.md`` knowledge files
    # (about, experience, portfolio, projects, ...). Generated stores and the
    # resume PDF directory are skipped; see ``app.ingestion.markdown_source``.
    markdown_data_dir: str = "./data"

    # --- GitHub ingestion ---
    github_token: str | None = None
    github_username: str = ""
    github_max_repos: int = 8
    github_max_commits_per_repo: int = 25
    github_max_source_files_per_repo: int = 6
    github_max_file_bytes: int = 60_000
    github_include_forks: bool = False
    github_source_extensions: list[str] = [
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".go",
        ".java",
        ".rs",
        ".rb",
        ".cpp",
        ".c",
        ".md",
    ]

    # --- RAG ---
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "persona_corpus"
    bm25_index_path: str = "./data/bm25/bm25_index.pkl"
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    embedding_batch_size: int = 96
    # Conservative defaults tuned for the Groq free tier. Smaller retrieval =
    # fewer prompt tokens. The retriever fuses max(rerank_candidates,
    # final_context_chunks) candidates, so rerank_candidates=0 is safe (it just
    # means "no extra candidates beyond the final context", used with
    # reranker_provider="none").
    top_k_vector: int = 4
    top_k_bm25: int = 4
    rrf_k: int = 60
    rerank_candidates: int = 0  # extra fused candidates for the reranker (0 = none)
    final_context_chunks: int = 2  # how many chunks land in the prompt
    # Source-aware fusion weighting (retrieval-only relevance fix). The corpus is
    # GitHub-heavy (~92% of chunks), so raw code chunks crowd out the curated
    # persona narrative. After RRF, each chunk's score is multiplied by a
    # per-source weight: curated narrative (resume/markdown) is boosted; raw
    # github source code is damped. Set both to 1.0 to disable.
    retrieval_narrative_boost: float = 2.5  # multiplier for resume + markdown chunks
    retrieval_github_source_weight: float = 0.4  # multiplier for github_source (raw code)

    # --- Persona identity ---
    persona_name: str = "the candidate"
    persona_title: str = "Software Engineer"
    persona_email: str = "candidate@example.com"
    persona_tagline: str = "A digital persona that answers from a verified corpus."

    # --- Scheduling ---
    timezone: str = "America/Los_Angeles"
    working_days: list[int] = [0, 1, 2, 3, 4]  # 0=Mon ... 6=Sun
    working_hours_start: int = 9
    working_hours_end: int = 17
    slot_minutes: int = 30
    booking_default_duration: int = 30
    booking_horizon_days: int = 14  # how far ahead availability is computed

    # --- Database ---
    database_url: str = "sqlite:///./data/persona.db"

    # --- Security ---
    injection_guard_enabled: bool = True
    injection_llm_classifier: bool = False  # heuristic by default; LLM optional
    grounding_check_enabled: bool = True
    # Grounding verifier backend. "llm" sends a JSON-mode judge completion;
    # "rule_based" validates citations + lexical support with zero LLM calls
    # (free-tier optimization). See app.security.grounding.check_grounding.
    grounding_check_provider: str = "llm"  # "llm" | "rule_based"

    # --- Vapi telephony bridge ---
    # Shared secret Vapi forwards to the /vapi bridge (as ``Authorization: Bearer``
    # on the custom-LLM URL, or ``x-vapi-secret`` on the server webhook). Empty
    # disables auth -- fine for local testing, but set it in any deployment Vapi
    # can reach. See app.api.routes.vapi.
    vapi_secret: str = ""

    # --- Brain ---
    chat_temperature: float = 0.2
    max_tool_iterations: int = 4
    max_history_messages: int = 8

    # --- Server ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["*"]
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance.

    The instance is constructed once (lazily) and reused. Because construction
    reads the environment / ``.env`` file, caching keeps configuration stable
    for the lifetime of the process and avoids repeated file I/O.

    Returns:
        Settings: The cached, validated settings object.
    """

    settings = Settings()
    logger.debug(
        "Settings loaded (chat_model=%s, embedding_model=%s, timezone=%s)",
        settings.openai_chat_model,
        settings.openai_embedding_model,
        settings.timezone,
    )
    return settings
