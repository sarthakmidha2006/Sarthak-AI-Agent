"""Embedding helpers built on top of the shared :class:`LLMClient`.

:class:`Embedder` is a thin, RAG-aware adapter around
:meth:`app.brain.llm.LLMClient.embed`. It knows how to turn chunks into
vectors (aligned 1:1 with their input order) and how to embed a single query
string. Batching and retry concerns live inside the :class:`LLMClient`; this
class only deals with the RAG-specific shapes.
"""

from __future__ import annotations

import logging

from app.brain.llm import LLMClient
from app.config import Settings
from app.rag.schemas import Chunk

logger = logging.getLogger(__name__)


class Embedder:
    """Produce embedding vectors for chunks and queries.

    The embedder defers all network/batching/retry logic to the injected
    :class:`LLMClient`. Vector order is always preserved so callers can zip the
    returned vectors back against their inputs.
    """

    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        """Initialize the embedder.

        Args:
            llm: The shared async OpenAI client wrapper.
            settings: Application settings (embedding model/batch size live
                inside the client, but kept here for parity and future use).
        """
        self._llm = llm
        self._settings = settings

    async def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Embed the text of each chunk.

        Args:
            chunks: Chunks to embed. Their ``text`` fields are sent to the
                embedding model in order.

        Returns:
            A list of embedding vectors aligned 1:1 with ``chunks``. Empty list
            if ``chunks`` is empty.

        Raises:
            ValueError: If the number of returned vectors does not match the
                number of input chunks (defensive alignment check).
        """
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        vectors = await self._llm.embed(texts)

        if len(vectors) != len(chunks):
            raise ValueError(
                "Embedding count mismatch: expected "
                f"{len(chunks)} vectors, received {len(vectors)}"
            )

        logger.debug("Embedded %d chunk(s)", len(chunks))
        return vectors

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Args:
            query: The query text to embed.

        Returns:
            A single embedding vector.

        Raises:
            ValueError: If the embedding backend returns no vector for the query.
        """
        vectors = await self._llm.embed([query])
        if not vectors:
            raise ValueError("Embedding backend returned no vector for query")
        return vectors[0]
