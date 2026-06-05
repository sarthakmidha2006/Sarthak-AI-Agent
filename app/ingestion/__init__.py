"""Offline corpus ingestion package (spec §15).

Turns external sources (a resume PDF and a GitHub account) into
:class:`~app.rag.schemas.Document` objects, chunks and embeds them, and
populates the vector store + BM25 index that the runtime RAG layer queries.

This is intentionally a bare package marker. It must NOT import its submodules
(``base``, ``resume``, ``github_source``, ``pipeline``, ``run_ingest``) at
package-import time so that importing the package never pulls in heavy or
optional dependencies (httpx, pypdf, chromadb). Import the specific submodule
you need directly, e.g.::

    from app.ingestion.pipeline import IngestionPipeline
"""

from __future__ import annotations
