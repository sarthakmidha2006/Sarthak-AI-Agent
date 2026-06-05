"""Ingestion source abstractions (spec §15.1).

Defines the :class:`Source` abstract base class that every ingestion source
(resume, GitHub, ...) implements, plus :func:`slug`, a small helper that turns
arbitrary strings into safe, stable identifier fragments used when composing
:class:`~app.rag.schemas.Document` ids.

A *source* is an async producer of :class:`~app.rag.schemas.Document` objects.
Sources must degrade gracefully: a missing resume file or an unreachable GitHub
API should yield an empty list and a logged warning rather than raising, so the
pipeline can still ingest whatever other sources succeed.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from app.rag.schemas import Document

logger = logging.getLogger(__name__)

# Characters that are safe inside an id fragment. Everything else collapses to a
# single hyphen so ids remain readable and free of path/URL-hostile characters.
_SLUG_STRIP_RE = re.compile(r"[^a-z0-9._/-]+")
_SLUG_DASH_RUN_RE = re.compile(r"-{2,}")


def slug(s: str) -> str:
    """Return a filesystem/id-safe slug for ``s``.

    The transform lowercases the input, replaces any run of unsafe characters
    with a single hyphen, and trims stray leading/trailing separators. Forward
    slashes, dots, and underscores are preserved because document/source ids in
    this system are conventionally shaped like ``"owner/repo:path/to/file.py"``
    and those separators carry meaning.

    Args:
        s: Arbitrary input string.

    Returns:
        A normalized slug. Returns ``"unknown"`` when the input is empty or
        reduces to nothing after normalization.
    """
    if not s:
        return "unknown"

    lowered = s.strip().lower()
    # Preserve the colon used in composite source ids (``repo:path``) by mapping
    # it explicitly; the generic strip below would otherwise drop it.
    lowered = lowered.replace(":", "/")
    cleaned = _SLUG_STRIP_RE.sub("-", lowered)
    cleaned = _SLUG_DASH_RUN_RE.sub("-", cleaned)
    cleaned = cleaned.strip("-/.")

    return cleaned or "unknown"


class Source(ABC):
    """Abstract base class for an asynchronous ingestion source.

    Concrete subclasses set a human-readable :attr:`name` and implement
    :meth:`load`, which returns the documents this source contributes to the
    corpus. Implementations are expected to handle their own external-call
    failures and return ``[]`` on unrecoverable errors rather than raising.

    Attributes:
        name: Short, stable identifier for the source (e.g. ``"resume"`` or
            ``"github"``). Used in logging and summaries.
    """

    #: Human-readable source name; subclasses override this.
    name: str = "source"

    @abstractmethod
    async def load(self) -> list[Document]:
        """Load and return this source's documents.

        Returns:
            A list of :class:`~app.rag.schemas.Document` objects. May be empty
            when the source is unconfigured or unavailable.
        """
        raise NotImplementedError
