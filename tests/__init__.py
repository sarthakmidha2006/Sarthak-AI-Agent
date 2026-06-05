"""Test suite for the AI Persona system (spec §17).

All tests run fully offline: there is no OpenAI key requirement and no network
access. The OpenAI boundary is replaced by the :class:`~tests.conftest.FakeLLM`
defined in :mod:`tests.conftest`, which produces scripted chat / tool-call
responses and deterministic embeddings. The database is a throwaway, temp-file
SQLite instance created per test session.
"""

from __future__ import annotations
