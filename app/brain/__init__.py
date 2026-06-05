"""Brain package: the shared LLM client, prompts, and persona orchestration.

This module is an intentional empty marker. The concrete modules ``llm``,
``prompts``, and ``persona`` live alongside it and are imported directly
(e.g. ``from app.brain.llm import LLMClient``) rather than re-exported here, to
avoid import-time side effects and circular imports during application startup.
"""

from __future__ import annotations
