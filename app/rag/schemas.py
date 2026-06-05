"""RAG data models (spec §4).

Plain dataclasses (deliberately not pydantic) used throughout the retrieval
layer: :class:`Document`, :class:`Chunk`, :class:`ScoredChunk`, and
:class:`RetrievalResult`. They are lightweight, hashable-by-id where useful,
and convert cleanly to/from the primitive-only metadata that ChromaDB accepts.

This module imports nothing from upper layers (spec §20).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Metadata value types that ChromaDB accepts in a metadata mapping.
_PRIMITIVE_TYPES = (str, int, float, bool)


@dataclass
class Document:
    """A single ingested source document prior to chunking.

    Attributes:
        id: Stable identifier, conventionally ``f"{source_type}:{source_id}"``
            (slugified by the ingestion layer).
        text: Full extracted text of the document.
        source_type: One of ``"resume"``, ``"markdown"``, ``"github_readme"``,
            ``"github_repo"``, ``"github_commit"``, ``"github_source"``.
        source_id: Logical source identifier, e.g. ``"resume"``,
            ``"about.md"``, ``"owner/repo"``, or
            ``"owner/repo:path/to/file.py"``.
        title: Human-readable title used in citations.
        url: Source link if available, else ``None``.
        metadata: Extra metadata. Must contain primitive values only (no
            nested objects) so it remains Chroma-safe once flattened.
    """

    id: str
    text: str
    source_type: str
    source_id: str
    title: str
    url: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A token-bounded slice of a :class:`Document`.

    Attributes:
        id: ``f"{document_id}::{chunk_index}"``.
        document_id: Id of the parent :class:`Document`.
        text: The chunk's text.
        source_type: Copied from the parent document.
        source_id: Copied from the parent document.
        title: Copied from the parent document.
        url: Copied from the parent document (may be ``None``).
        chunk_index: Zero-based position of this chunk within its document.
        metadata: Extra primitive metadata copied from the parent document.
    """

    id: str
    document_id: str
    text: str
    source_type: str
    source_id: str
    title: str
    url: str | None
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_chroma_metadata(self) -> dict[str, Any]:
        """Flatten this chunk's metadata to Chroma-safe primitives.

        Stores the core fields (``document_id``, ``source_type``,
        ``source_id``, ``title``, ``url`` — empty string if ``None`` —, and
        ``chunk_index``) plus any primitive (``str``/``int``/``float``/
        ``bool``) value from :attr:`metadata`. Non-primitive metadata values
        are coerced to their string representation so the result is always
        accepted by ChromaDB.

        Returns:
            A flat mapping containing only str/int/float/bool values.
        """

        meta: dict[str, Any] = {
            "document_id": self.document_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "title": self.title,
            "url": self.url if self.url is not None else "",
            "chunk_index": self.chunk_index,
        }

        for key, value in self.metadata.items():
            # Never let extra metadata clobber the reserved core fields.
            if key in meta:
                continue
            if isinstance(value, bool):
                # bool is a subclass of int; check it first to preserve type.
                meta[key] = value
            elif isinstance(value, _PRIMITIVE_TYPES):
                meta[key] = value
            elif value is None:
                meta[key] = ""
            else:
                meta[key] = str(value)

        return meta

    @classmethod
    def from_chroma(cls, chunk_id: str, document: str, metadata: dict[str, Any]) -> "Chunk":
        """Rebuild a :class:`Chunk` from data returned by ChromaDB.

        Args:
            chunk_id: The chunk's id (Chroma's record id).
            document: The stored document text for this chunk.
            metadata: The stored, flattened metadata mapping.

        Returns:
            A reconstructed :class:`Chunk`. The empty-string ``url`` sentinel is
            mapped back to ``None``, and the reserved core fields are stripped
            out of the residual :attr:`metadata` dict.
        """

        metadata = dict(metadata or {})

        document_id = str(metadata.get("document_id", ""))
        source_type = str(metadata.get("source_type", ""))
        source_id = str(metadata.get("source_id", ""))
        title = str(metadata.get("title", ""))

        raw_url = metadata.get("url", "")
        url: str | None = raw_url if isinstance(raw_url, str) and raw_url != "" else None

        try:
            chunk_index = int(metadata.get("chunk_index", 0))
        except (TypeError, ValueError):
            chunk_index = 0

        extra = {
            key: value
            for key, value in metadata.items()
            if key
            not in {
                "document_id",
                "source_type",
                "source_id",
                "title",
                "url",
                "chunk_index",
            }
        }

        return cls(
            id=chunk_id,
            document_id=document_id,
            text=document,
            source_type=source_type,
            source_id=source_id,
            title=title,
            url=url,
            chunk_index=chunk_index,
            metadata=extra,
        )


@dataclass
class ScoredChunk:
    """A :class:`Chunk` paired with a relevance score and its origin.

    Attributes:
        chunk: The underlying chunk.
        score: Relevance score; semantics depend on :attr:`retriever`.
        retriever: One of ``"vector"``, ``"bm25"``, ``"rrf"``, ``"rerank"``.
    """

    chunk: Chunk
    score: float
    retriever: str


@dataclass
class RetrievalResult:
    """The final ranked output of a hybrid retrieval pass.

    Attributes:
        query: The original query string.
        chunks: Final, ranked chunks (post-rerank).
        timings_ms: Per-stage timings in milliseconds, e.g. keys
            ``"vector"``, ``"bm25"``, ``"fuse"``, ``"rerank"``, ``"total"``.
        candidate_count: Number of fused candidates considered before rerank.
    """

    query: str
    chunks: list[ScoredChunk] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    candidate_count: int = 0
