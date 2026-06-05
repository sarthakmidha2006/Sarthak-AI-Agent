"""FastAPI route modules for the AI Persona system.

Each module exposes a module-level ``router`` (:class:`fastapi.APIRouter`) that is
included by the application factory in :mod:`app.main`:

* :mod:`app.api.routes.chat` -- ``POST /chat``
* :mod:`app.api.routes.voice` -- ``POST /voice`` (multipart audio or JSON text)
* :mod:`app.api.routes.availability` -- ``GET /availability``
* :mod:`app.api.routes.booking` -- ``POST /book``
* :mod:`app.api.routes.health` -- ``GET /health`` and ``GET /``

The routers obtain shared singletons (brain, LLM client, vector store, settings)
from ``request.app.state`` and a per-request database session via the
:func:`app.api.deps.get_db` dependency.
"""

from __future__ import annotations

__all__: list[str] = []
