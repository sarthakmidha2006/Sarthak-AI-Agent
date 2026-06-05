"""Hybrid retrieval fusion via Reciprocal Rank Fusion (RRF).

RRF combines several independently-ranked result lists (e.g. dense vector
results and sparse BM25 results) into a single ranking without needing to
calibrate or normalize the heterogeneous scores from each retriever. Each
chunk's fused score is the sum over result lists of ``1 / (k + rank)``, where
``rank`` is the chunk's 0-based position within that list.
"""

from __future__ import annotations

import logging

from app.rag.schemas import ScoredChunk

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[ScoredChunk]],
    *,
    k: int = 60,
    top_n: int | None = None,
    source_weights: dict[str, float] | None = None,
) -> list[ScoredChunk]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    For each chunk appearing in any input list, the fused score is::

        score(chunk) = w(source) · Σ_i 1 / (k + rank_i)

    where ``rank_i`` is the chunk's 0-based rank in list ``i`` (chunks absent
    from a list contribute nothing for that list) and ``w(source)`` is an
    optional per-``source_type`` multiplier (default 1.0). Chunks are
    deduplicated by ``chunk.id``; the first-seen :class:`Chunk` object is
    retained as the canonical payload. Results are sorted by descending fused
    score and tagged with ``retriever="rrf"``.

    Args:
        result_lists: One ranked list per retriever. Each list is assumed to be
            already sorted best-first.
        k: RRF damping constant. Larger values flatten the contribution of top
            ranks. Defaults to 60 (the value from the original RRF paper).
        top_n: If given, return only the top ``top_n`` fused results.
        source_weights: Optional ``{source_type: weight}`` map applied as a
            multiplier to each chunk's fused score. Lets a caller boost curated
            sources (e.g. ``resume``/``markdown``) and damp noisy ones (e.g.
            ``github_source``) without changing the rank maths. When ``None``
            (the default) every weight is 1.0 and scores are unchanged.

    Returns:
        A single fused, deduplicated, descending-ranked list of scored chunks.
    """
    if k <= 0:
        # Guard against division-by-zero / negative denominators; fall back to
        # the canonical value rather than raising in a hot path.
        logger.warning("RRF k must be positive; received %d, using 60", k)
        k = 60

    fused_scores: dict[str, float] = {}
    canonical_chunks: dict[str, ScoredChunk] = {}

    for result_list in result_lists:
        for rank, scored in enumerate(result_list):
            chunk_id = scored.chunk.id
            contribution = 1.0 / (k + rank)
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + contribution
            # Keep the first-seen chunk object as canonical.
            if chunk_id not in canonical_chunks:
                canonical_chunks[chunk_id] = scored

    def _weight(chunk_id: str) -> float:
        if not source_weights:
            return 1.0
        return source_weights.get(canonical_chunks[chunk_id].chunk.source_type, 1.0)

    fused: list[ScoredChunk] = [
        ScoredChunk(
            chunk=canonical_chunks[chunk_id].chunk,
            score=score * _weight(chunk_id),
            retriever="rrf",
        )
        for chunk_id, score in fused_scores.items()
    ]
    fused.sort(key=lambda sc: sc.score, reverse=True)

    if top_n is not None:
        fused = fused[:top_n]

    logger.debug(
        "RRF fused %d list(s) into %d unique chunk(s)%s",
        len(result_lists),
        len(fused),
        f" (top_n={top_n})" if top_n is not None else "",
    )
    return fused
