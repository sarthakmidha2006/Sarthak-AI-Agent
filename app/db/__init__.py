"""Database package.

Contains the SQLAlchemy 2.0 engine/session plumbing (:mod:`app.db.database`),
the ORM models (:mod:`app.db.models`), and idempotent seed helpers
(:mod:`app.db.seed`). This is a bare package marker to keep import order
predictable; import the specific submodule you need directly.
"""

from __future__ import annotations
