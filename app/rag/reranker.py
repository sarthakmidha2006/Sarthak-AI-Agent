"""Cross-encoder-style reranking of fused retrieval candidates.

After RRF fusion we have a set of plausible candidates. A reranker reorders
them by *query-specific* relevance and truncates to the final context budget.

Two implementations are provided:

* :class:`LLMReranker` (default) -- asks the chat model to score each candidate
  via a single JSON-mode completion. It is *fail-open*: any error (network,
  parse, malformed output) degrades gracefully to the input ordering.
* :class:`CohereReranker` -- uses Cohere's dedicated rerank endpoint when a
  ``cohere_api_key`` is configured.
* :class:`NoOpReranker` -- skips reranking entirely and returns the fused
  candidates directly (selected via ``reranker_provider == "none"``).

:func:`get_reranker` chooses between them based on ``Settings``.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol, runtime_checkable

from app.brain.llm import LLMClient
from app.config import Settings
from app.rag.schemas import ScoredChunk

logger = logging.getLogger(__name__)

# Hard cap on how much chunk text we feed the reranker per candidate. Reranking
# only needs enough text to judge relevance; sending full chunks wastes tokens.
_MAX_SNIPPET_CHARS = 1200


@runtime_checkable
class Reranker(Protocol):
    """Protocol for rerankers used by :class:`app.rag.retriever.HybridRetriever`."""

    async def rerank(
        self, query: str, chunks: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        """Reorder ``chunks`` by relevance to ``query`` and return the top ``top_n``."""
        ...


def _truncate(text: str, limit: int = _MAX_SNIPPET_CHARS) -> str:
    """Return ``text`` clipped to ``limit`` characters."""
    if len(text) <= limit:
        return text
    return text[:limit]


class LLMReranker:
    """Rerank candidates using a single JSON-mode chat completion.

    The model is asked to return a JSON object containing a ``scores`` array of
    ``{"index": i, "score": 0..10}`` entries. Parsing is defensive: missing or
    malformed scores default to ``0``. On *any* failure the reranker falls back
    to the input order (truncated to ``top_n``) and logs a warning -- it never
    raises into the retrieval path.
    """

    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        """Initialize the LLM reranker.

        Args:
            llm: Shared async OpenAI client wrapper.
            settings: Provides ``reranker_model``.
        """
        self._llm = llm
        self._settings = settings

    async def rerank(
        self, query: str, chunks: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        """Rerank ``chunks`` by relevance to ``query``.

        Args:
            query: The user query.
            chunks: Candidate scored chunks (typically RRF output).
            top_n: Number of chunks to keep after reranking.

        Returns:
            Up to ``top_n`` chunks tagged ``retriever="rerank"``, ordered by
            descending model relevance score. On failure, the input order is
            preserved.
        """
        if not chunks:
            return []
        if top_n <= 0:
            return []

        messages = self._build_messages(query, chunks)

        try:
            result = await self._llm.chat(
                messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            scores_by_index = self._parse_scores(result.message, len(chunks))
        except Exception:
            logger.warning(
                "LLM reranker failed; falling back to input order", exc_info=True
            )
            return self._fallback(chunks, top_n)

        # Build (index, score) pairs for every candidate, defaulting missing
        # scores to 0, then sort descending. Python's sort is stable, so ties
        # retain the original (RRF) ordering.
        indexed = [(i, scores_by_index.get(i, 0.0)) for i in range(len(chunks))]
        indexed.sort(key=lambda pair: pair[1], reverse=True)

        reranked: list[ScoredChunk] = []
        for index, score in indexed[:top_n]:
            original = chunks[index]
            reranked.append(
                ScoredChunk(chunk=original.chunk, score=float(score), retriever="rerank")
            )
        logger.debug("LLM reranker returned %d chunk(s)", len(reranked))
        return reranked

    def _build_messages(self, query: str, chunks: list[ScoredChunk]) -> list[dict]:
        """Construct the chat messages for the rerank request."""
        numbered = []
        for index, scored in enumerate(chunks):
            chunk = scored.chunk
            snippet = _truncate(chunk.text)
            numbered.append(
                f"[{index}] (title: {chunk.title} | source: {chunk.source_type})\n{snippet}"
            )
        documents_block = "\n\n".join(numbered)

        system = (
            "You are a search relevance judge. You are given a user query and a "
            "numbered list of candidate documents. Score how relevant each document "
            "is to answering the query on a scale from 0 (irrelevant) to 10 "
            "(directly answers the query). Respond ONLY with a JSON object of the "
            'form {"scores": [{"index": <int>, "score": <number 0-10>}, ...]} '
            "containing exactly one entry per candidate index. Do not add commentary."
        )
        user = (
            f"Query:\n{query}\n\n"
            f"Candidates:\n{documents_block}\n\n"
            "Return the JSON object now."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _parse_scores(message: dict, num_candidates: int) -> dict[int, float]:
        """Parse the model's JSON response into an ``index -> score`` mapping.

        Robust to extra keys, string scores, out-of-range indices, and the
        scores being delivered either as a list under ``scores`` or as a bare
        list / index-keyed object.

        Raises:
            ValueError: If the message content is missing or not valid JSON, so
                the caller can fall open.
        """
        content = (message or {}).get("content")
        if not content or not isinstance(content, str):
            raise ValueError("Reranker response had no textual content")

        data = json.loads(content)

        # Accept several shapes the model might emit.
        entries: list = []
        if isinstance(data, dict):
            if isinstance(data.get("scores"), list):
                entries = data["scores"]
            elif isinstance(data.get("results"), list):
                entries = data["results"]
            else:
                # Possibly an index-keyed object like {"0": 8, "1": 3}.
                index_keyed: dict[int, float] = {}
                for key, value in data.items():
                    try:
                        idx = int(key)
                        index_keyed[idx] = float(value)
                    except (TypeError, ValueError):
                        continue
                if index_keyed:
                    return {
                        i: s for i, s in index_keyed.items() if 0 <= i < num_candidates
                    }
                raise ValueError("Reranker JSON object had no recognizable scores")
        elif isinstance(data, list):
            entries = data
        else:
            raise ValueError("Reranker JSON was neither object nor array")

        scores: dict[int, float] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            raw_index = entry.get("index")
            raw_score = entry.get("score")
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if not (0 <= index < num_candidates):
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = 0.0
            scores[index] = score
        return scores

    @staticmethod
    def _fallback(chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        """Return the input order, truncated and retagged as ``rerank``."""
        return [
            ScoredChunk(chunk=sc.chunk, score=sc.score, retriever="rerank")
            for sc in chunks[:top_n]
        ]


class CohereReranker:
    """Rerank candidates using Cohere's dedicated rerank endpoint.

    Used only when ``reranker_provider == "cohere"`` and a ``cohere_api_key`` is
    configured. Like :class:`LLMReranker`, it is fail-open: any error falls back
    to the input order.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the Cohere reranker.

        Args:
            settings: Provides ``cohere_api_key`` and ``cohere_rerank_model``.

        Raises:
            RuntimeError: If the ``cohere`` package is not installed.
            ValueError: If no Cohere API key is configured.
        """
        self._settings = settings
        if not settings.cohere_api_key:
            raise ValueError("CohereReranker requires settings.cohere_api_key")
        try:
            import cohere  # noqa: PLC0415 - imported lazily; optional dependency
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "cohere package is not installed; install 'cohere' or use the LLM reranker"
            ) from exc
        self._model = settings.cohere_rerank_model
        # AsyncClient lets us await rerank without blocking the event loop.
        self._client = cohere.AsyncClient(api_key=settings.cohere_api_key)

    async def rerank(
        self, query: str, chunks: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        """Rerank ``chunks`` via Cohere, falling open on error.

        Args:
            query: The user query.
            chunks: Candidate scored chunks.
            top_n: Number of chunks to keep.

        Returns:
            Up to ``top_n`` chunks tagged ``retriever="rerank"`` ordered by
            Cohere relevance score. Input order on failure.
        """
        if not chunks or top_n <= 0:
            return []

        documents = [_truncate(sc.chunk.text) for sc in chunks]
        try:
            response = await self._client.rerank(
                model=self._model,
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents)),
            )
        except Exception:
            logger.warning(
                "Cohere reranker failed; falling back to input order", exc_info=True
            )
            return self._fallback(chunks, top_n)

        results = getattr(response, "results", None)
        if not results:
            logger.warning("Cohere returned no rerank results; using input order")
            return self._fallback(chunks, top_n)

        reranked: list[ScoredChunk] = []
        for item in results:
            index = getattr(item, "index", None)
            score = getattr(item, "relevance_score", 0.0)
            if index is None or not (0 <= index < len(chunks)):
                continue
            original = chunks[index]
            reranked.append(
                ScoredChunk(
                    chunk=original.chunk, score=float(score), retriever="rerank"
                )
            )
        if not reranked:
            return self._fallback(chunks, top_n)
        logger.debug("Cohere reranker returned %d chunk(s)", len(reranked))
        return reranked[:top_n]

    @staticmethod
    def _fallback(chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        """Return the input order, truncated and retagged as ``rerank``."""
        return [
            ScoredChunk(chunk=sc.chunk, score=sc.score, retriever="rerank")
            for sc in chunks[:top_n]
        ]


class NoOpReranker:
    """Pass-through reranker that performs no reranking call.

    Selected when ``reranker_provider == "none"``. It returns the fused
    candidates directly (truncated to ``top_n``), making zero LLM/network
    calls. Output matches the other rerankers' shape -- chunks are tagged
    ``retriever="rerank"`` and capped at ``top_n`` -- so every downstream
    interface (:class:`~app.rag.retriever.HybridRetriever`, citations,
    grounding) is unchanged.

    NOTE: temporary free-tier optimization for Groq (skips the token-heavy
    rerank completion). Set ``RERANKER_PROVIDER=llm`` to restore reranking.
    """

    async def rerank(
        self, query: str, chunks: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        """Return the fused candidates directly, truncated to ``top_n``."""
        if not chunks or top_n <= 0:
            return []
        return [
            ScoredChunk(chunk=sc.chunk, score=sc.score, retriever="rerank")
            for sc in chunks[:top_n]
        ]


def get_reranker(settings: Settings, llm: LLMClient) -> Reranker:
    """Return the configured reranker implementation.

    Selects :class:`NoOpReranker` when ``reranker_provider == "none"`` (skips
    reranking entirely); :class:`CohereReranker` when ``reranker_provider ==
    "cohere"`` and a Cohere API key is present; otherwise (or if Cohere init
    fails) returns the default :class:`LLMReranker`.

    Args:
        settings: Application settings.
        llm: Shared LLM client (used by the default reranker).

    Returns:
        A reranker satisfying the :class:`Reranker` protocol.
    """
    provider = (settings.reranker_provider or "llm").lower()
    if provider == "none":
        logger.info("Reranking disabled (RERANKER_PROVIDER=none); using fused order")
        return NoOpReranker()
    if provider == "cohere" and settings.cohere_api_key:
        try:
            logger.info("Using Cohere reranker (model=%s)", settings.cohere_rerank_model)
            return CohereReranker(settings)
        except (RuntimeError, ValueError):
            logger.warning(
                "Cohere reranker unavailable; falling back to LLM reranker",
                exc_info=True,
            )
    logger.info("Using LLM reranker (model=%s)", settings.reranker_model)
    return LLMReranker(llm, settings)
