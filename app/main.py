"""FastAPI application factory and ASGI entrypoint for the AI Persona system.

This module wires the whole service together (spec section 14.3):

* :func:`create_app` builds the :class:`fastapi.FastAPI` instance, configures
  CORS from settings, installs a request-timing middleware that sets the
  ``X-Process-Time-ms`` response header, registers a global exception handler
  that returns ``{"error": ...}`` with HTTP 500, and includes every route
  router.
* The ``lifespan`` context manager constructs the expensive, long-lived
  singletons exactly once at startup and stores them on ``app.state`` so the
  dependency providers in :mod:`app.api.deps` and the routes can reach them:
  the :class:`~app.brain.llm.LLMClient`, :class:`~app.rag.vector_store.VectorStore`,
  :class:`~app.rag.embeddings.Embedder`, :class:`~app.rag.bm25_index.BM25Index`
  (loaded from disk if present, otherwise rebuilt from the vector store), the
  reranker, the :class:`~app.rag.retriever.HybridRetriever`, the
  :class:`~app.security.prompt_guard.PromptGuard`, and the shared
  :class:`~app.brain.persona.PersonaBrain`.

Running this module directly (``python -m app.main``) launches uvicorn.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.brain.llm import LLMClient
from app.brain.persona import PersonaBrain
from app.config import Settings, get_settings
from app.db.database import get_session, init_db, session_scope
from app.db.seed import seed_availability_overrides
from app.logging_config import setup_logging
from app.rag.bm25_index import BM25Index
from app.rag.embeddings import Embedder
from app.rag.reranker import get_reranker
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import VectorStore
from app.security.prompt_guard import PromptGuard

# Route routers
from app.api.routes import availability as availability_routes
from app.api.routes import booking as booking_routes
from app.api.routes import call as call_routes
from app.api.routes import chat as chat_routes
from app.api.routes import health as health_routes
from app.api.routes import vapi as vapi_routes
from app.api.routes import voice as voice_routes

logger = logging.getLogger(__name__)


def _build_bm25_index(vector_store: VectorStore, settings: Settings) -> BM25Index:
    """Load the persisted BM25 index, or rebuild it from the vector store.

    Per spec section 14.3 the BM25 index is loaded from
    ``settings.bm25_index_path`` when that file exists; otherwise it is rebuilt
    in-memory from the chunks already stored in the vector store so the lexical
    retriever stays consistent with the embeddings. Any failure to load falls
    back to a rebuild rather than crashing startup.
    """

    path = settings.bm25_index_path
    if path and os.path.exists(path):
        try:
            index = BM25Index.load(path)
            logger.info("Loaded BM25 index from %s (size=%d)", path, index.size)
            return index
        except Exception:  # noqa: BLE001 - degrade to a rebuild
            logger.warning(
                "Failed to load BM25 index from %s; rebuilding from vector store",
                path,
                exc_info=True,
            )

    try:
        chunks = vector_store.get_all_chunks()
    except Exception:  # noqa: BLE001 - empty index is acceptable for a fresh deploy
        logger.warning(
            "Could not read chunks from vector store to rebuild BM25 index; "
            "starting with an empty lexical index",
            exc_info=True,
        )
        chunks = []

    index = BM25Index.from_chunks(chunks)
    logger.info("Rebuilt BM25 index from vector store (size=%d)", index.size)
    return index


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build singletons on startup and tear them down on shutdown."""

    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("Starting AI Persona service")

    # 1. Database: ensure schema exists, then seed idempotent demo data.
    init_db()
    try:
        with session_scope() as session:
            seed_availability_overrides(session)
    except Exception:  # noqa: BLE001 - seeding is best-effort
        logger.warning("Availability override seeding failed", exc_info=True)

    # 2. Core clients / stores.
    llm = LLMClient(settings)
    vector_store = VectorStore(settings)
    embedder = Embedder(llm, settings)
    bm25 = _build_bm25_index(vector_store, settings)

    # 2b. Warm up the local embedding model at startup so the first real query
    # doesn't pay the one-time model-load cost (~1–3s). Best-effort: a failure
    # here must not prevent the service from starting.
    warm_start = time.perf_counter()
    try:
        await llm.embed(["warmup"])
        logger.info(
            "Embedding model warmed up (%s) in %.0fms",
            settings.openai_embedding_model,
            (time.perf_counter() - warm_start) * 1000.0,
        )
    except Exception:  # noqa: BLE001 - warm-up is best-effort
        logger.warning("Embedding model warm-up failed; will load lazily", exc_info=True)

    # 3. Retrieval + security + brain.
    reranker = get_reranker(settings, llm)
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25=bm25,
        embedder=embedder,
        reranker=reranker,
        settings=settings,
    )
    guard = PromptGuard(settings, llm)
    brain = PersonaBrain(
        retriever=retriever,
        llm=llm,
        guard=guard,
        settings=settings,
        session_factory=get_session,
    )

    # 4. Expose singletons for deps + routes.
    app.state.settings = settings
    app.state.llm = llm
    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.bm25 = bm25
    app.state.reranker = reranker
    app.state.retriever = retriever
    app.state.guard = guard
    app.state.brain = brain

    try:
        logger.info(
            "Startup complete: corpus_chunks=%s bm25_size=%s",
            _safe_count(vector_store),
            bm25.size,
        )
        yield
    finally:
        logger.info("Shutting down AI Persona service")
        await _aclose(llm)


def _safe_count(vector_store: VectorStore) -> int:
    """Return the vector store chunk count, ``-1`` if it cannot be read."""

    try:
        return vector_store.count()
    except Exception:  # noqa: BLE001 - startup logging must not crash
        return -1


async def _aclose(llm: LLMClient) -> None:
    """Best-effort async/sync close of the LLM client's network resources."""

    aclose = getattr(llm, "aclose", None)
    if callable(aclose):
        try:
            await aclose()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            logger.warning("Error closing LLM client", exc_info=True)
        return
    close = getattr(llm, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            logger.warning("Error closing LLM client", exc_info=True)


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""

    settings = get_settings()

    app = FastAPI(
        title="AI Persona",
        version="1.0.0",
        description=(
            "A digital persona that answers strictly from a verified corpus "
            "(resume + GitHub) and schedules meetings via tool calling. One "
            "shared brain serves both chat and voice channels."
        ),
        lifespan=lifespan,
    )

    # CORS from settings.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time-ms"],
    )

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Time each request and report it via the ``X-Process-Time-ms`` header."""

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.2f}"
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Convert any uncaught exception into a JSON ``{"error": ...}`` 500."""

        logger.exception("Unhandled error processing %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

    # Routers.
    app.include_router(chat_routes.router)
    app.include_router(voice_routes.router)
    app.include_router(availability_routes.router)
    app.include_router(booking_routes.router)
    app.include_router(health_routes.router)
    app.include_router(vapi_routes.router)
    app.include_router(call_routes.router)

    return app


app = create_app()


def main() -> None:
    """CLI entrypoint: run the app under uvicorn using configured host/port."""

    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
