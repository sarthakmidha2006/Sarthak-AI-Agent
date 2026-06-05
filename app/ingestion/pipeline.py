"""End-to-end ingestion pipeline (spec §15.4).

:class:`IngestionPipeline` orchestrates the offline corpus build:

1. gather documents from all configured :class:`~app.ingestion.base.Source`
   objects (resume + GitHub by default), concurrently;
2. chunk them with :func:`app.rag.chunking.chunk_documents`;
3. embed the chunks with :class:`app.rag.embeddings.Embedder`;
4. upsert chunks + embeddings into the :class:`app.rag.vector_store.VectorStore`;
5. build a :class:`app.rag.bm25_index.BM25Index` from the same chunks and persist
   it to ``settings.bm25_index_path``.

It returns a JSON-serializable summary describing how many documents/chunks were
ingested and their breakdown by source type. The ``reset`` flag wipes the vector
collection before adding, enabling clean re-ingestion.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter

from app.brain.llm import LLMClient
from app.config import Settings
from app.ingestion.base import Source
from app.ingestion.github_source import GitHubSource
from app.ingestion.markdown_source import MarkdownSource
from app.ingestion.resume import ResumeSource
from app.rag.bm25_index import BM25Index
from app.rag.chunking import chunk_documents
from app.rag.embeddings import Embedder
from app.rag.schemas import Document
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _default_sources(settings: Settings) -> list[Source]:
    """Build the default source set (resume + markdown + GitHub) from settings."""
    return [ResumeSource(settings), MarkdownSource(settings), GitHubSource(settings)]


class IngestionPipeline:
    """Coordinate sources -> chunk -> embed -> index for the persona corpus."""

    def __init__(self, settings: Settings, *, sources: list[Source] | None = None) -> None:
        """Initialize the pipeline.

        Args:
            settings: Application settings driving every stage.
            sources: Optional explicit list of sources. When ``None``, the
                default resume + GitHub sources are constructed from settings.
        """
        self._settings = settings
        self._sources: list[Source] = (
            sources if sources is not None else _default_sources(settings)
        )

    async def run(self, *, reset: bool = False) -> dict:
        """Execute the full ingestion pipeline.

        Args:
            reset: When ``True``, the vector collection is wiped before new
                chunks are added (clean re-ingestion).

        Returns:
            A JSON-serializable summary dict::

                {
                    "documents": <int>,
                    "chunks": <int>,
                    "by_source_type": {<source_type>: <chunk_count>, ...},
                    "collection": <chroma collection name>,
                    "bm25_path": <path>,
                    "reset": <bool>,
                }
        """
        llm = LLMClient(self._settings)
        embedder = Embedder(llm, self._settings)
        vector_store = VectorStore(self._settings)

        try:
            if reset:
                logger.info("Reset requested; clearing existing vector collection")
                vector_store.reset()

            documents = await self._gather_documents()
            logger.info("Gathered %d document(s) from %d source(s)",
                        len(documents), len(self._sources))

            chunks = chunk_documents(documents, settings=self._settings)
            logger.info("Produced %d chunk(s)", len(chunks))

            if chunks:
                embeddings = await embedder.embed_chunks(chunks)
                vector_store.add(chunks, embeddings)
            else:
                logger.warning("No chunks produced; vector store left unchanged")

            bm25 = BM25Index.from_chunks(chunks)
            bm25.save(self._settings.bm25_index_path)
            logger.info(
                "Saved BM25 index (%d chunk(s)) to %s",
                bm25.size,
                self._settings.bm25_index_path,
            )

            summary = self._build_summary(documents, chunks, reset=reset)
            logger.info("Ingestion complete: %s", summary)
            return summary
        finally:
            await llm.aclose()

    async def _gather_documents(self) -> list[Document]:
        """Load every source concurrently and flatten the results.

        A failure in one source is logged and treated as an empty contribution
        so the rest of the corpus can still be ingested.
        """
        results = await asyncio.gather(
            *(self._safe_load(source) for source in self._sources),
            return_exceptions=False,
        )
        documents: list[Document] = []
        for docs in results:
            documents.extend(docs)
        return documents

    async def _safe_load(self, source: Source) -> list[Document]:
        """Load one source, swallowing errors into an empty list."""
        try:
            docs = await source.load()
            logger.info("Source %r contributed %d document(s)", source.name, len(docs))
            return docs
        except Exception:
            logger.warning(
                "Source %r failed to load; continuing without it",
                getattr(source, "name", source),
                exc_info=True,
            )
            return []

    def _build_summary(
        self, documents: list[Document], chunks: list, *, reset: bool
    ) -> dict:
        """Compose the JSON-serializable run summary."""
        by_source_type: Counter[str] = Counter(chunk.source_type for chunk in chunks)
        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "by_source_type": dict(by_source_type),
            "collection": self._settings.chroma_collection,
            "bm25_path": self._settings.bm25_index_path,
            "reset": reset,
        }
