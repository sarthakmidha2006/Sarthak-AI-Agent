"""FastAPI dependency providers for the AI Persona API.

The expensive, long-lived singletons (the :class:`~app.brain.persona.PersonaBrain`,
the :class:`~app.brain.llm.LLMClient`, the
:class:`~app.rag.retriever.HybridRetriever`, and the resolved
:class:`~app.config.Settings`) are constructed once during application startup
(see :mod:`app.main`) and stored on ``app.state``. The dependencies in this module
are thin accessors that hand those singletons to route handlers, plus a
request-scoped SQLAlchemy :class:`~sqlalchemy.orm.Session` provider.

Keeping these accessors here (rather than reaching into ``request.app.state``
inline in every route) makes the routes easy to override in tests via
``app.dependency_overrides``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator

from fastapi import Request

from app.config import Settings, get_settings
from app.db.database import SessionLocal

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from sqlalchemy.orm import Session

    from app.brain.llm import LLMClient
    from app.brain.persona import PersonaBrain
    from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)


def get_settings_dep() -> Settings:
    """Return the cached application :class:`~app.config.Settings` instance.

    Thin wrapper around :func:`app.config.get_settings` so route handlers can
    depend on settings via FastAPI's dependency injection and tests can override
    it cleanly.
    """

    return get_settings()


def get_brain(request: Request) -> "PersonaBrain":
    """Return the shared :class:`~app.brain.persona.PersonaBrain` singleton.

    The brain is built once in the application lifespan and stored on
    ``app.state.brain``.

    Raises:
        RuntimeError: if the brain has not been initialised (e.g. the lifespan
            did not run). This surfaces as a 500 via the global exception handler
            rather than an opaque ``AttributeError``.
    """

    brain = getattr(request.app.state, "brain", None)
    if brain is None:
        logger.error("PersonaBrain singleton is not available on app.state")
        raise RuntimeError("Application not fully initialised: brain is unavailable")
    return brain


def get_llm(request: Request) -> "LLMClient":
    """Return the shared :class:`~app.brain.llm.LLMClient` singleton.

    Stored on ``app.state.llm`` during the application lifespan.

    Raises:
        RuntimeError: if the LLM client has not been initialised.
    """

    llm = getattr(request.app.state, "llm", None)
    if llm is None:
        logger.error("LLMClient singleton is not available on app.state")
        raise RuntimeError("Application not fully initialised: llm is unavailable")
    return llm


def get_retriever(request: Request) -> "HybridRetriever":
    """Return the shared :class:`~app.rag.retriever.HybridRetriever` singleton.

    Stored on ``app.state.retriever`` during the application lifespan.

    Raises:
        RuntimeError: if the retriever has not been initialised.
    """

    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:
        logger.error("HybridRetriever singleton is not available on app.state")
        raise RuntimeError("Application not fully initialised: retriever is unavailable")
    return retriever


def get_db() -> Iterator["Session"]:
    """Yield a request-scoped SQLAlchemy session, closing it afterwards.

    Use as a FastAPI dependency::

        @router.get("/things")
        def list_things(db: Session = Depends(get_db)) -> ...:
            ...

    The session is always closed in the ``finally`` block, even if the route
    raises.
    """

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
