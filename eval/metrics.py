"""Evaluation metrics for the AI Persona system (spec §16.2).

Pure, dependency-light metric functions used by :mod:`eval.run_eval`:

* :func:`precision_at_k` / :func:`recall_at_k` — retrieval quality, using a
  *prefix* match between a retrieved chunk's ``source_id`` and a relevant id.
* :func:`aggregate_latency` — p50 / p95 / mean / max over a list of latency
  values (milliseconds).
* :func:`hallucination_rate` — ``1 - mean(grounded)`` over grounding results.
* :func:`booking_success_rate` — fraction of booking scenarios whose actual
  status matched the expected status.

The retrieval matching rule (shared by precision and recall): a retrieved
source id *matches* a relevant id when they are equal **or** the retrieved id
starts with the relevant id (so a relevant id of ``"owner/repo"`` matches a
retrieved ``"owner/repo:path/to/file.py"``).

This module imports only the standard library plus ``app`` types used purely
for annotations; it performs no network or database I/O.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    from app.brain.llm import LLMClient
    from app.config import Settings
    from app.security.grounding import GroundingResult

logger = logging.getLogger(__name__)


def _matches(retrieved_id: str, relevant_id: str) -> bool:
    """Return ``True`` if a retrieved source id matches a relevant source id.

    Match semantics (spec §16.2): equality or prefix — the retrieved id starts
    with the relevant id. Empty relevant ids never match (they would otherwise
    match everything via ``startswith("")``).

    Args:
        retrieved_id: A retrieved chunk's ``source_id``.
        relevant_id: A gold-item relevant source id.

    Returns:
        Whether the pair is considered a match.
    """

    if not relevant_id:
        return False
    return retrieved_id == relevant_id or retrieved_id.startswith(relevant_id)


def _is_relevant(retrieved_id: str, relevant: Iterable[str]) -> bool:
    """Return ``True`` if ``retrieved_id`` matches *any* relevant id."""

    return any(_matches(retrieved_id, rel) for rel in relevant)


def precision_at_k(
    retrieved_source_ids: list[str], relevant: list[str], k: int
) -> float:
    """Precision@k for a single query.

    Of the first ``k`` retrieved source ids, what fraction are relevant (by the
    prefix-match rule)? Duplicate retrieved ids are counted as they appear in
    the ranked list (the denominator is the number of considered positions).

    Args:
        retrieved_source_ids: Retrieved chunk source ids, in rank order.
        relevant: The gold-item relevant source ids.
        k: Cutoff rank (``k <= 0`` yields ``0.0``).

    Returns:
        Precision in ``[0, 1]``. Returns ``0.0`` when ``k <= 0`` or nothing was
        retrieved.
    """

    if k <= 0:
        return 0.0
    top = retrieved_source_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for source_id in top if _is_relevant(source_id, relevant))
    return hits / len(top)


def recall_at_k(retrieved_source_ids: list[str], relevant: list[str], k: int) -> float:
    """Recall@k for a single query.

    Of the (deduplicated) relevant source ids, what fraction are *covered* by at
    least one of the first ``k`` retrieved ids (prefix match)?

    Args:
        retrieved_source_ids: Retrieved chunk source ids, in rank order.
        relevant: The gold-item relevant source ids.
        k: Cutoff rank (``k <= 0`` yields ``0.0``).

    Returns:
        Recall in ``[0, 1]``. Returns ``0.0`` when ``k <= 0``. Returns ``1.0``
        when there are no relevant ids to find (vacuously complete) so that
        items without relevance labels do not depress the aggregate recall.
    """

    if k <= 0:
        return 0.0
    unique_relevant = [rel for rel in dict.fromkeys(relevant) if rel]
    if not unique_relevant:
        return 1.0
    top = retrieved_source_ids[:k]
    covered = sum(
        1
        for rel in unique_relevant
        if any(_matches(source_id, rel) for source_id in top)
    )
    return covered / len(unique_relevant)


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Return the ``pct`` percentile of ``sorted_values`` (linear interpolation).

    Args:
        sorted_values: Ascending-sorted, non-empty sequence of values.
        pct: Percentile in ``[0, 100]``.

    Returns:
        The interpolated percentile value.
    """

    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = rank - lower_index
    lower = float(sorted_values[lower_index])
    upper = float(sorted_values[upper_index])
    return lower + (upper - lower) * fraction


def aggregate_latency(query_logs_or_values: list[float]) -> dict:
    """Aggregate a list of latency values (milliseconds) into summary stats.

    Args:
        query_logs_or_values: Latency values in milliseconds. Each entry may be
            a raw number, or an object exposing a ``latency_ms`` /
            ``latency_ms_total`` attribute (e.g. a ``BrainResponse`` or
            ``QueryLog``); such objects are reduced to their numeric latency.

    Returns:
        A dict with keys ``p50``, ``p95``, ``mean``, ``max``, and ``count``. All
        statistics are ``0.0`` when the input is empty.
    """

    values = _coerce_latency_values(query_logs_or_values)
    if not values:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "max": 0.0, "count": 0}

    ordered = sorted(values)
    return {
        "p50": _percentile(ordered, 50.0),
        "p95": _percentile(ordered, 95.0),
        "mean": sum(ordered) / len(ordered),
        "max": float(ordered[-1]),
        "count": len(ordered),
    }


def _coerce_latency_values(items: Iterable[object]) -> list[float]:
    """Reduce mixed latency inputs to a flat list of floats (ms).

    Accepts raw numbers as well as objects carrying ``latency_ms`` or
    ``latency_ms_total`` attributes. Non-numeric / unrecognised entries are
    skipped with a debug log so a malformed entry never aborts the run.
    """

    values: list[float] = []
    for item in items:
        if isinstance(item, bool):
            # ``bool`` is a subclass of ``int``; treating it as latency is a bug.
            continue
        if isinstance(item, (int, float)):
            values.append(float(item))
            continue
        latency = getattr(item, "latency_ms", None)
        if latency is None:
            latency = getattr(item, "latency_ms_total", None)
        if isinstance(latency, (int, float)) and not isinstance(latency, bool):
            values.append(float(latency))
        else:
            logger.debug("Skipping unrecognised latency entry: %r", item)
    return values


async def hallucination_rate(
    items_results: list["GroundingResult"],
    llm: "LLMClient | None" = None,
    settings: "Settings | None" = None,
) -> float:
    """Compute the hallucination rate from grounding results.

    Defined as ``1 - mean(grounded)`` across the provided
    :class:`~app.security.grounding.GroundingResult` objects, where a result
    counts as grounded (1.0) or not (0.0) by its ``grounded`` flag.

    The function is ``async`` to match the spec's signature and to allow future
    implementations to run the grounding judge inline; the current
    implementation expects callers (see :mod:`eval.run_eval`) to have already
    produced the grounding verdicts, so ``llm`` / ``settings`` are accepted but
    unused here.

    Args:
        items_results: Per-answer grounding verdicts.
        llm: Optional LLM client (accepted for signature parity; unused).
        settings: Optional settings (accepted for signature parity; unused).

    Returns:
        Hallucination rate in ``[0, 1]``. ``0.0`` when there are no results
        (nothing was judged ungrounded).
    """

    del llm, settings  # accepted for signature parity per spec; not needed here

    if not items_results:
        return 0.0

    grounded_flags = [1.0 if getattr(r, "grounded", False) else 0.0 for r in items_results]
    mean_grounded = sum(grounded_flags) / len(grounded_flags)
    rate = 1.0 - mean_grounded
    logger.debug(
        "hallucination_rate: %d result(s), mean_grounded=%.3f, rate=%.3f",
        len(items_results),
        mean_grounded,
        rate,
    )
    return rate


def booking_success_rate(results: list[tuple[str, str]]) -> float:
    """Fraction of booking scenarios whose actual status matched the expected.

    Args:
        results: A list of ``(expected_status, actual_status)`` pairs. Comparison
            is case-insensitive and whitespace-insensitive.

    Returns:
        Success rate in ``[0, 1]``. ``0.0`` when there are no results.
    """

    if not results:
        return 0.0
    correct = sum(
        1
        for expected, actual in results
        if str(expected).strip().lower() == str(actual).strip().lower()
    )
    return correct / len(results)
