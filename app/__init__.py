"""AI Persona System ("the brain").

A digital persona that answers questions strictly from a retrieved corpus
(resume + GitHub) and can schedule meetings via tool calling. A single shared
backend serves both chat and voice channels.

This package root deliberately imports nothing from submodules to keep import
order predictable and avoid circular imports. Sub-packages are imported on
demand by the application entrypoints.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
