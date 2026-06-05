"""Hybrid retrieval orchestration.

:class:`HybridRetriever` ties the RAG components together for a single query:

1. Embed the query and run a dense vector search (ChromaDB).
2. Run a sparse lexical search (BM25).
3. Fuse the two ranked lists with Reciprocal Rank Fusion.
4. Rerank the fused candidates and truncate to the final context budget.

Each stage is timed; the per-stage and total latencies (milliseconds) are
returned on the :class:`RetrievalResult` so the brain and eval harness can
report latency breakdowns.
"""

from __future__ import annotations

import logging
import time

from app.config import Settings
from app.rag.bm25_index import BM25Index
from app.rag.embeddings import Embedder
from app.rag.hybrid import reciprocal_rank_fusion
from app.rag.reranker import Reranker
from app.rag.schemas import RetrievalResult, ScoredChunk
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _elapsed_ms(since: float) -> float:
    """Return milliseconds elapsed since the perf-counter timestamp ``since``."""
    return round((time.perf_counter() - since) * 1000.0, 3)


class HybridRetriever:
    """Dense + sparse retrieval with RRF fusion and reranking."""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        bm25: BM25Index,
        embedder: Embedder,
        reranker: Reranker,
        settings: Settings,
    ) -> None:
        """Initialize the retriever with its collaborators.

        Args:
            vector_store: Dense vector store (ChromaDB wrapper).
            bm25: Lexical BM25 index.
            embedder: Query embedder over the shared LLM client.
            reranker: Reranker used to reorder fused candidates.
            settings: Provides ``top_k_vector``, ``top_k_bm25``, ``rrf_k``,
                ``rerank_candidates`` and ``final_context_chunks``.
        """
        self._vector_store = vector_store
        self._bm25 = bm25
        self._embedder = embedder
        self._reranker = reranker
        self._settings = settings

    async def retrieve(self, query: str) -> RetrievalResult:
        """Run the full hybrid retrieval pipeline for ``query``.

        Args:
            query: The user query string.

        Returns:
            A :class:`RetrievalResult` whose ``chunks`` are the final,
            post-rerank, ranked context chunks, with timing for each stage and
            the count of fused candidates considered.
        """
        timings_ms: dict[str, float] = {}
        total_start = time.perf_counter()

        # --- 1) Dense vector search -------------------------------------
        vector_results: list[ScoredChunk] = []
        vector_start = time.perf_counter()
        try:
            query_embedding = await self._embedder.embed_query(query)
            vector_results = self._vector_store.query(
                query_embedding, self._settings.top_k_vector
            )
        except Exception:
            logger.warning(
                "Vector retrieval failed; continuing with lexical only", exc_info=True
            )
        timings_ms["vector"] = _elapsed_ms(vector_start)

        # --- 2) Sparse lexical (BM25) search ----------------------------
        bm25_start = time.perf_counter()
        try:
            bm25_results = self._bm25.query(query, self._settings.top_k_bm25)
        except Exception:
            logger.warning("BM25 retrieval failed; continuing without it", exc_info=True)
            bm25_results = []
        timings_ms["bm25"] = _elapsed_ms(bm25_start)

        # --- 3) RRF fusion ----------------------------------------------
        # Keep enough fused candidates to (a) feed the reranker when one is
        # active and (b) always fill the final context budget. Using
        # ``max(rerank_candidates, final_context_chunks)`` means a configured
        # ``rerank_candidates=0`` (reranking disabled) still yields the chunks
        # the prompt needs instead of fusing to an empty list.
        fuse_top_n = max(
            int(self._settings.rerank_candidates),
            int(self._settings.final_context_chunks),
        )
        # Source-aware weighting: boost curated persona narrative and damp raw
        # github source code so the (GitHub-heavy) corpus doesn't bury the
        # resume/markdown answers. Pure retrieval-side reweighting — the rank
        # maths are unchanged. See Settings.retrieval_* for the multipliers.
        source_weights = {
            "resume": self._settings.retrieval_narrative_boost,
            "markdown": self._settings.retrieval_narrative_boost,
            "github_source": self._settings.retrieval_github_source_weight,
        }
        fuse_start = time.perf_counter()
        fused = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            k=self._settings.rrf_k,
            top_n=fuse_top_n,
            source_weights=source_weights,
        )
        timings_ms["fuse"] = _elapsed_ms(fuse_start)
        candidate_count = len(fused)

        # --- 4) Rerank --------------------------------------------------
        rerank_start = time.perf_counter()
        try:
            final_chunks = await self._reranker.rerank(
                query, fused, self._settings.final_context_chunks
            )
        except Exception:
            # Rerankers are expected to fail open themselves, but guard the
            # orchestration too so retrieval never throws to the brain.
            logger.warning(
                "Reranker raised unexpectedly; using fused order", exc_info=True
            )
            final_chunks = [
                ScoredChunk(chunk=sc.chunk, score=sc.score, retriever="rerank")
                for sc in fused[: self._settings.final_context_chunks]
            ]
        timings_ms["rerank"] = _elapsed_ms(rerank_start)

        timings_ms["total"] = _elapsed_ms(total_start)

        logger.info(
            "Retrieved query=%r vector=%d bm25=%d fused=%d final=%d total=%.1fms",
            query[:80],
            len(vector_results),
            len(bm25_results),
            candidate_count,
            len(final_chunks),
            timings_ms["total"],
        )

        return RetrievalResult(
            query=query,
            chunks=final_chunks,
            timings_ms=timings_ms,
            candidate_count=candidate_count,
        )
