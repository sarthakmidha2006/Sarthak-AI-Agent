"""Health and service-info routes -- ``GET /health`` and ``GET /``.

``GET /health`` reports liveness plus a quick read of the corpus size (vector
store chunk count and BM25 index size) and the configured model names, so an
operator can tell at a glance whether ingestion has run. It never raises: if the
vector store or BM25 index cannot be read, the corresponding count degrades to a
sentinel value and the failure is logged rather than turning a health check into
a 500.

``GET /`` returns a small JSON banner describing the persona and pointing at the
interactive docs (spec section 14.2).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_settings_dep
from app.config import Settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["meta"])


def _safe_count(obj: Any, method: str) -> int:
    """Call ``obj.method()`` for a count, returning ``-1`` on any failure.

    Used so ``/health`` stays a cheap, non-throwing endpoint even when the
    underlying store is mid-initialisation or unavailable.
    """

    if obj is None:
        return -1
    fn = getattr(obj, method, None)
    if fn is None:
        return -1
    try:
        return int(fn())
    except Exception:  # noqa: BLE001 - health must not raise
        logger.warning("Failed to read %s.%s() for health check", type(obj).__name__, method)
        return -1


def _bm25_size(bm25: Any) -> int:
    """Read the BM25 index size from its ``size`` property, ``-1`` on failure."""

    if bm25 is None:
        return -1
    try:
        return int(bm25.size)
    except Exception:  # noqa: BLE001 - health must not raise
        logger.warning("Failed to read BM25 index size for health check")
        return -1


@router.get("/health", summary="Liveness and corpus status")
def health(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
) -> dict[str, Any]:
    """Return service liveness, corpus sizes, and configured model names."""

    state = request.app.state
    vector_store = getattr(state, "vector_store", None)
    bm25 = getattr(state, "bm25", None)

    return {
        "status": "ok",
        "corpus_chunks": _safe_count(vector_store, "count"),
        "bm25_size": _bm25_size(bm25),
        "models": {
            "chat": settings.openai_chat_model,
            "embedding": settings.openai_embedding_model,
            "stt": settings.openai_stt_model,
            "tts": settings.openai_tts_model,
            "reranker_provider": settings.reranker_provider,
            "grounding_provider": settings.grounding_check_provider,
        },
    }


@router.get("/diagnostics", summary="Backend configuration and corpus diagnostics")
def diagnostics(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
) -> dict[str, Any]:
    """Report the active backends, model names, and corpus sizes.

    Used to confirm at a glance which providers are wired (Groq for chat/STT,
    local sentence-transformers for embeddings, local Piper for TTS) and whether
    ingestion has populated the indexes. Never raises: counts degrade to ``-1``.
    """

    state = request.app.state
    vector_store = getattr(state, "vector_store", None)
    bm25 = getattr(state, "bm25", None)

    return {
        "chat_model": settings.openai_chat_model,
        "embedding_model": settings.openai_embedding_model,
        "reranker_provider": settings.reranker_provider,
        "grounding_provider": settings.grounding_check_provider,
        "corpus_chunks": _safe_count(vector_store, "count"),
        "bm25_chunks": _bm25_size(bm25),
        "groq_enabled": bool(settings.groq_api_key),
        "embedding_backend": "local",
        "tts_backend": f"piper:{settings.openai_tts_model}",
        "stt_backend": f"groq:{settings.openai_stt_model}",
        "status": "healthy",
    }


@router.get("/", summary="Service information")
def root(settings: Settings = Depends(get_settings_dep)) -> dict[str, Any]:
    """Return a small JSON banner describing the persona service."""

    return {
        "service": "ai-persona",
        "persona": {
            "name": settings.persona_name,
            "title": settings.persona_title,
            "tagline": settings.persona_tagline,
        },
        "endpoints": [
            "/chat",
            "/voice",
            "/availability",
            "/book",
            "/health",
            "/diagnostics",
        ],
        "docs": "/docs",
    }
