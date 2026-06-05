"""Evaluation harness for the AI Persona system (spec §16).

This package measures the quality and safety of the persona "brain" against a
small gold dataset:

* :mod:`eval.dataset` — the gold-data schema (:class:`~eval.dataset.GoldItem`,
  :class:`~eval.dataset.BookingScenario`) and a loader that falls back to
  generic built-in template examples.
* :mod:`eval.metrics` — retrieval precision/recall (prefix match), latency
  aggregation (p50/p95/mean/max), hallucination rate, and booking success rate.
* :mod:`eval.run_eval` — the orchestrator/CLI that runs retrieval, grounding,
  injection-refusal, and booking checks, writes ``eval/report.json`` plus
  :class:`app.db.models.EvalResult` rows, and prints a summary table.

The import root for this package is ``eval.`` (e.g. ``from eval.metrics import
precision_at_k``). Modules here import from ``app.*`` but the ``app`` layers do
not depend on ``eval`` (spec §20), so there is no import cycle.
"""

from __future__ import annotations

__all__ = [
    "dataset",
    "metrics",
    "run_eval",
]
