"""Lexical (BM25) retrieval index over chunks.

This module provides a thin, persistable wrapper around
``rank_bm25.BM25Okapi``. It owns its own tokenizer (lowercase ``\\w+`` tokens
of length >= 2) and keeps the original :class:`~app.rag.schemas.Chunk` objects
so that querying can return fully-populated :class:`ScoredChunk` results.

The index is empty-safe: an unbuilt or empty index returns ``[]`` from
:meth:`BM25Index.query` instead of raising.
"""

from __future__ import annotations

import logging
import os
import pickle
import re

from rank_bm25 import BM25Okapi

from app.rag.schemas import Chunk, ScoredChunk

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+")

# Pickle protocol version: keep an explicit, stable value so saved indexes are
# portable across the supported Python range.
_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


def tokenize(text: str) -> list[str]:
    """Tokenize ``text`` for BM25.

    Lowercases the text, extracts ``\\w+`` runs, and drops tokens shorter than
    two characters (single letters/digits carry little lexical signal).

    Args:
        text: Raw text to tokenize.

    Returns:
        A list of lowercase token strings (possibly empty).
    """
    if not text:
        return []
    return [token for token in _TOKEN_RE.findall(text.lower()) if len(token) >= 2]


class BM25Index:
    """A BM25 lexical index over a fixed set of chunks.

    Attributes are private; use the public methods/properties. The index stores
    the original chunks (so results carry full metadata) and the tokenized
    corpus (so it can be persisted and reloaded without re-tokenizing).
    """

    def __init__(self) -> None:
        """Create an empty index. Call :meth:`build` or :meth:`from_chunks`."""
        self._chunks: list[Chunk] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        """(Re)build the index from ``chunks``.

        Args:
            chunks: Chunks to index. An empty list yields an empty (but valid)
                index whose :meth:`query` returns ``[]``.
        """
        self._chunks = list(chunks)
        self._corpus_tokens = [tokenize(chunk.text) for chunk in self._chunks]

        # ``BM25Okapi`` raises on an empty corpus, so only build when we have at
        # least one chunk with tokens. We still keep the (empty) state valid.
        non_empty = [tokens for tokens in self._corpus_tokens if tokens]
        if not non_empty:
            self._bm25 = None
            logger.info("BM25 index built over %d chunk(s) (no tokens; index inert)",
                        len(self._chunks))
            return

        self._bm25 = BM25Okapi(self._corpus_tokens)
        logger.info("BM25 index built over %d chunk(s)", len(self._chunks))

    def query(self, text: str, top_k: int) -> list[ScoredChunk]:
        """Return the ``top_k`` best lexical matches for ``text``.

        Args:
            text: Query text.
            top_k: Maximum number of results.

        Returns:
            Scored chunks ordered by descending BM25 score with
            ``retriever="bm25"``. Empty list if the index is empty/unbuilt, the
            query has no usable tokens, or ``top_k`` <= 0.
        """
        if self._bm25 is None or not self._chunks or top_k <= 0:
            return []

        query_tokens = tokenize(text)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Pair each score with its chunk, sort descending, and keep the top_k.
        ranked = sorted(
            zip(self._chunks, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )

        results: list[ScoredChunk] = []
        for chunk, score in ranked[:top_k]:
            results.append(
                ScoredChunk(chunk=chunk, score=float(score), retriever="bm25")
            )
        logger.debug("BM25 query returned %d result(s)", len(results))
        return results

    def save(self, path: str) -> None:
        """Persist the index to ``path`` via pickle.

        The on-disk payload contains the chunks (serialized as plain dicts so we
        do not pin the on-disk format to the dataclass layout) and the tokenized
        corpus, enabling a fast reload without re-tokenizing.

        Args:
            path: Destination file path. Parent directories are created.
        """
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

        payload = {
            "chunks": [self._chunk_to_dict(chunk) for chunk in self._chunks],
            "corpus_tokens": self._corpus_tokens,
        }
        with open(path, "wb") as handle:
            pickle.dump(payload, handle, protocol=_PICKLE_PROTOCOL)
        logger.info("Saved BM25 index (%d chunk(s)) to %s", len(self._chunks), path)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        """Load an index previously written by :meth:`save`.

        Args:
            path: Path to the pickled index.

        Returns:
            A populated :class:`BM25Index`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"BM25 index file not found: {path}")

        with open(path, "rb") as handle:
            payload = pickle.load(handle)

        index = cls()
        chunk_dicts = payload.get("chunks", [])
        index._chunks = [cls._chunk_from_dict(data) for data in chunk_dicts]
        index._corpus_tokens = payload.get("corpus_tokens") or [
            tokenize(chunk.text) for chunk in index._chunks
        ]

        non_empty = [tokens for tokens in index._corpus_tokens if tokens]
        index._bm25 = BM25Okapi(index._corpus_tokens) if non_empty else None

        logger.info("Loaded BM25 index (%d chunk(s)) from %s", len(index._chunks), path)
        return index

    @classmethod
    def from_chunks(cls, chunks: list[Chunk]) -> "BM25Index":
        """Build a new index directly from ``chunks``.

        Args:
            chunks: Chunks to index.

        Returns:
            A built :class:`BM25Index`.
        """
        index = cls()
        index.build(chunks)
        return index

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)

    @staticmethod
    def _chunk_to_dict(chunk: Chunk) -> dict:
        """Serialize a :class:`Chunk` to a plain dict for pickling."""
        return {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "text": chunk.text,
            "source_type": chunk.source_type,
            "source_id": chunk.source_id,
            "title": chunk.title,
            "url": chunk.url,
            "chunk_index": chunk.chunk_index,
            "metadata": dict(chunk.metadata),
        }

    @staticmethod
    def _chunk_from_dict(data: dict) -> Chunk:
        """Rebuild a :class:`Chunk` from its serialized dict form."""
        return Chunk(
            id=data["id"],
            document_id=data["document_id"],
            text=data["text"],
            source_type=data["source_type"],
            source_id=data["source_id"],
            title=data["title"],
            url=data.get("url"),
            chunk_index=data["chunk_index"],
            metadata=dict(data.get("metadata") or {}),
        )
