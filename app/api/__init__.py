"""HTTP API layer for the AI Persona system.

This sub-package contains the FastAPI dependency providers (:mod:`app.api.deps`)
and the route modules under :mod:`app.api.routes`. It deliberately imports
nothing at package-import time to keep the import graph acyclic; the application
factory in :mod:`app.main` wires everything together at startup.
"""

from __future__ import annotations

__all__: list[str] = []
