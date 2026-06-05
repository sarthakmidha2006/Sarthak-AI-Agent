"""Retrieval-Augmented Generation (RAG) package.

This is intentionally a bare package marker. It must NOT import its submodules
(``schemas``, ``chunking``, ``embeddings``, ``vector_store``, ``bm25_index``,
``hybrid``, ``reranker``, ``retriever``) to avoid import cycles, since several
of those modules depend on layers that in turn import from ``app.rag``. Import
the specific submodule you need directly, e.g.::

    from app.rag.schemas import Chunk
"""

from __future__ import annotations
