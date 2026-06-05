"""ChromaDB-backed vector store for the persona corpus.

This module wraps a ``chromadb.PersistentClient`` collection configured for
cosine distance. We supply pre-computed embeddings ourselves (the OpenAI
embedding model lives in :class:`app.rag.embeddings.Embedder`), so the
collection is created with ``embedding_function=None`` -- Chroma must never try
to embed anything on its own.

Cosine *distance* returned by Chroma is converted into a similarity *score*
via ``score = 1 - distance`` so that higher is better, matching the convention
used across the RAG layer.
"""

from __future__ import annotations

import logging
import os

# Disable ChromaDB anonymized telemetry process-wide *before* importing
# chromadb, so the (posthog-incompatible) telemetry client never tries to send
# events. This is the canonical env switch; combined with
# ``anonymized_telemetry=False`` on the client and the telemetry-logger silencing
# in app.logging_config, it cleanly removes the "capture() takes 1 positional
# argument but 3 were given" noise. ``setdefault`` respects an explicit override.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import Settings
from app.rag.schemas import Chunk, ScoredChunk

logger = logging.getLogger(__name__)

# Chroma collection metadata that selects cosine space for the HNSW index.
_COSINE_METADATA = {"hnsw:space": "cosine"}


class VectorStore:
    """A persistent cosine-similarity vector store over chunk embeddings.

    Embeddings are always supplied by the caller; the collection is created
    with ``embedding_function=None``. Documents (chunk text) and Chroma-safe
    metadata are stored alongside the vectors so chunks can be fully rebuilt on
    query without a second datastore.
    """

    def __init__(self, settings: Settings) -> None:
        """Open (or create) the persistent collection.

        Args:
            settings: Provides ``chroma_persist_dir`` and ``chroma_collection``.
        """
        self._settings = settings
        self._persist_dir = settings.chroma_persist_dir
        self._collection_name = settings.chroma_collection

        logger.info(
            "Opening Chroma persistent client at %s (collection=%s)",
            self._persist_dir,
            self._collection_name,
        )
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> "chromadb.api.models.Collection.Collection":
        """Return the cosine collection, creating it if necessary.

        The return type is annotated as a string to avoid importing Chroma's
        internal ``Collection`` symbol, whose module path has shifted across
        chromadb releases. ``from __future__ import annotations`` keeps this
        annotation un-evaluated, so import-safety does not depend on that path.
        """
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata=_COSINE_METADATA,
            embedding_function=None,
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Upsert chunks and their embeddings into the collection.

        Args:
            chunks: Chunks to store. Their ids, text, and flattened metadata are
                persisted.
            embeddings: Embedding vectors aligned 1:1 with ``chunks``.

        Raises:
            ValueError: If ``chunks`` and ``embeddings`` lengths differ.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                "add() requires aligned inputs: "
                f"{len(chunks)} chunks vs {len(embeddings)} embeddings"
            )
        if not chunks:
            logger.debug("VectorStore.add called with no chunks; nothing to do")
            return

        ids = [chunk.id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.to_chroma_metadata() for chunk in chunks]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info("Upserted %d chunk(s) into collection %s", len(ids), self._collection_name)

    def query(self, embedding: list[float], top_k: int) -> list[ScoredChunk]:
        """Return the ``top_k`` most similar chunks to ``embedding``.

        Args:
            embedding: The query embedding vector.
            top_k: Maximum number of results to return.

        Returns:
            Scored chunks ordered by descending similarity, with
            ``retriever="vector"`` and ``score = 1 - cosine_distance``. Empty
            list if the collection is empty or ``top_k`` <= 0.
        """
        if top_k <= 0:
            return []

        count = self.count()
        if count == 0:
            return []

        # Chroma errors if n_results exceeds the collection size in some
        # backends; clamp defensively.
        n_results = min(top_k, count)

        try:
            response = self._collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:  # pragma: no cover - defensive against chroma errors
            logger.exception("Chroma query failed; returning no vector results")
            return []

        ids_batches = response.get("ids") or [[]]
        documents_batches = response.get("documents") or [[]]
        metadatas_batches = response.get("metadatas") or [[]]
        distances_batches = response.get("distances") or [[]]

        ids = ids_batches[0] if ids_batches else []
        documents = documents_batches[0] if documents_batches else []
        metadatas = metadatas_batches[0] if metadatas_batches else []
        distances = distances_batches[0] if distances_batches else []

        results: list[ScoredChunk] = []
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            chunk = Chunk.from_chroma(
                chunk_id=chunk_id,
                document=document or "",
                metadata=dict(metadata or {}),
            )
            score = 1.0 - float(distance)
            results.append(ScoredChunk(chunk=chunk, score=score, retriever="vector"))

        logger.debug("Vector query returned %d result(s)", len(results))
        return results

    def get_all_chunks(self) -> list[Chunk]:
        """Return every stored chunk (used to (re)build the BM25 index).

        Returns:
            All chunks currently in the collection, rebuilt from stored
            documents and metadata. Empty list if the collection is empty.
        """
        if self.count() == 0:
            return []

        try:
            response = self._collection.get(include=["documents", "metadatas"])
        except Exception:  # pragma: no cover - defensive against chroma errors
            logger.exception("Chroma get_all_chunks failed; returning empty list")
            return []

        ids = response.get("ids") or []
        documents = response.get("documents") or []
        metadatas = response.get("metadatas") or []

        chunks: list[Chunk] = []
        for chunk_id, document, metadata in zip(ids, documents, metadatas):
            chunks.append(
                Chunk.from_chroma(
                    chunk_id=chunk_id,
                    document=document or "",
                    metadata=dict(metadata or {}),
                )
            )
        logger.debug("Loaded %d chunk(s) from collection", len(chunks))
        return chunks

    def count(self) -> int:
        """Return the number of chunks stored in the collection."""
        try:
            return int(self._collection.count())
        except Exception:  # pragma: no cover - defensive against chroma errors
            logger.exception("Chroma count failed; reporting 0")
            return 0

    def reset(self) -> None:
        """Delete and recreate the collection, discarding all stored vectors."""
        logger.warning("Resetting Chroma collection %s", self._collection_name)
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception:
            # Collection may not exist yet; that's fine for a reset.
            logger.debug(
                "delete_collection(%s) failed (likely missing); continuing",
                self._collection_name,
            )
        self._collection = self._get_or_create_collection()
