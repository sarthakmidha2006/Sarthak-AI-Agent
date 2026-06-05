"""Database engine and session management (spec §5.1).

Builds the SQLAlchemy 2.0 synchronous engine from ``settings.database_url``,
exposes the declarative :class:`Base`, a configured :data:`SessionLocal`
factory, and helpers to initialise the schema (:func:`init_db`), obtain a raw
session (:func:`get_session`), and run a transactional unit of work
(:func:`session_scope`).

SQLite gets ``check_same_thread=False`` so a single connection can be shared
across the FastAPI thread pool; ``pool_pre_ping=True`` guards against stale
connections for server-backed databases.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import urlparse

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""


def _build_engine(database_url: str) -> Engine:
    """Construct the SQLAlchemy engine for ``database_url``.

    SQLite URLs receive ``check_same_thread=False`` connect args. All engines
    enable ``pool_pre_ping`` to transparently recover from dropped connections.

    Args:
        database_url: A SQLAlchemy database URL.

    Returns:
        A configured :class:`~sqlalchemy.engine.Engine`.
    """

    is_sqlite = database_url.startswith("sqlite")
    connect_args: dict[str, object] = {"check_same_thread": False} if is_sqlite else {}

    engine = create_engine(
        database_url,
        connect_args=connect_args,
        pool_pre_ping=True,
        future=True,
    )
    logger.debug("Database engine created for %s (sqlite=%s)", database_url, is_sqlite)
    return engine


#: Process-wide synchronous engine.
engine: Engine = _build_engine(_settings.database_url)

#: Session factory. Sessions do not autoflush and keep attributes accessible
#: after commit so route handlers can read instance fields post-commit.
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
    class_=Session,
    future=True,
)


def _ensure_sqlite_dir(database_url: str) -> None:
    """Create the parent directory for a file-backed SQLite database.

    No-op for in-memory SQLite or non-SQLite URLs.

    Args:
        database_url: The configured database URL.
    """

    if not database_url.startswith("sqlite"):
        return

    # Strip the scheme; SQLAlchemy SQLite URLs look like ``sqlite:///path``.
    path_part = database_url.split("sqlite:///", 1)[-1]
    if not path_part or path_part == ":memory:" or path_part.startswith(":memory:"):
        return

    # Some URLs may carry a query string; urlparse handles the general case.
    parsed = urlparse(database_url)
    raw_path = parsed.path or path_part
    if database_url.startswith("sqlite:////"):
        # Four slashes → an ABSOLUTE filesystem path; keep the leading slash
        # (e.g. sqlite:////var/lib/app/persona.db → /var/lib/app/persona.db).
        db_path = raw_path
    elif database_url.startswith("sqlite:///"):
        # Three slashes → path RELATIVE to CWD; drop the single leading slash
        # urlparse prepends (e.g. sqlite:///./data/persona.db → ./data/persona.db).
        db_path = raw_path.lstrip("/")
    else:
        db_path = raw_path
    directory = os.path.dirname(db_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
        logger.debug("Ensured database directory exists: %s", directory)


def init_db() -> None:
    """Create the data directory (if needed) and all tables.

    Imports :mod:`app.db.models` for its side effect of registering every ORM
    class on :data:`Base.metadata`, then issues ``create_all``. Safe to call
    repeatedly; existing tables are left untouched.
    """

    _ensure_sqlite_dir(_settings.database_url)

    # Import models lazily so their tables are registered on Base.metadata
    # before create_all runs, without creating an import cycle at module load.
    from app.db import models as _models  # noqa: F401  (registration side effect)

    Base.metadata.create_all(bind=engine)
    logger.info("Database schema initialised (%s)", _settings.database_url)


def get_session() -> Session:
    """Return a brand-new :class:`~sqlalchemy.orm.Session`.

    The caller is responsible for closing the session (e.g. via a dependency's
    ``finally`` block). For automatic commit/rollback/close semantics prefer
    :func:`session_scope`.

    Returns:
        A new session bound to the global engine.
    """

    return SessionLocal()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope around a series of operations.

    Commits on success, rolls back on any exception, and always closes the
    session. Usage::

        with session_scope() as session:
            session.add(obj)

    Yields:
        An active session.

    Raises:
        Exception: Re-raises whatever was raised inside the block after rolling
            back the transaction.
    """

    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Rolling back database session after error")
        raise
    finally:
        session.close()
