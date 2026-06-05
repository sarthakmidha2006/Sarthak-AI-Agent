"""Centralized logging configuration.

Provides :func:`setup_logging`, which configures the root logger with a
consistent format. It is intended to be called once, early in the application
lifespan (spec §14.3) before other modules begin emitting log records.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: Canonical log line format used across the application.
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def setup_logging(level: str) -> None:
    """Configure the root logger.

    Idempotent and safe to call more than once: existing handlers are replaced
    so repeated invocations (e.g. across test runs or reload cycles) do not
    duplicate log output. An unrecognised ``level`` string falls back to
    ``INFO`` rather than raising.

    Args:
        level: A logging level name such as ``"DEBUG"``, ``"INFO"``,
            ``"WARNING"``, ``"ERROR"``. Case-insensitive.
    """

    resolved = logging.getLevelName(str(level).upper())
    if not isinstance(resolved, int):
        logger.warning("Unknown log level %r; defaulting to INFO", level)
        resolved = logging.INFO

    root = logging.getLogger()
    root.setLevel(resolved)

    # Replace any pre-existing handlers so the format/level is authoritative
    # and we never accumulate duplicate handlers on repeated setup.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setLevel(resolved)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(handler)

    _silence_noisy_third_party_loggers()

    logger.debug("Logging configured at level %s", logging.getLevelName(resolved))


def _silence_noisy_third_party_loggers() -> None:
    """Quiet known-noisy third-party loggers that emit harmless errors.

    ChromaDB's product-telemetry client is incompatible with recent ``posthog``
    releases and logs ``Failed to send telemetry event ...: capture() takes 1
    positional argument but 3 were given`` on startup and on each collection
    operation. Telemetry is already disabled via ``anonymized_telemetry=False``
    in :class:`app.rag.vector_store.VectorStore`, but the client still attempts
    (and fails) the send. The failure is purely cosmetic and has no functional
    impact, so we raise the telemetry logger above ERROR to suppress the spam.
    """
    logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
