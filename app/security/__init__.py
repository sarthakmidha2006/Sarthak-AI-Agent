"""Security subsystem for the AI Persona "brain".

This package bundles the two defensive layers that protect the persona:

* :mod:`app.security.prompt_guard` — detects and refuses prompt-injection /
  exfiltration attempts and neutralizes untrusted retrieved content so it can be
  embedded safely as *data* inside the model prompt.
* :mod:`app.security.grounding` — an LLM "judge" that verifies the persona's
  answer is actually supported by the retrieved context (hallucination check).

Both modules are import-safe and free of side effects at import time.
"""

from __future__ import annotations

from app.security.grounding import GroundingResult, check_grounding
from app.security.prompt_guard import (
    INJECTION_PATTERNS,
    REFUSAL_MESSAGE,
    GuardResult,
    PromptGuard,
    neutralize_context,
)

__all__ = [
    "GroundingResult",
    "check_grounding",
    "INJECTION_PATTERNS",
    "REFUSAL_MESSAGE",
    "GuardResult",
    "PromptGuard",
    "neutralize_context",
]
