"""Token-aware document chunking for the RAG corpus.

This module slices documents into overlapping, token-bounded chunks using
``tiktoken`` so that downstream embedding and lexical indexing operate on
uniformly sized units. Chunk sizes and overlaps are driven by ``Settings``.

The public surface is:

* :func:`chunk_text` -- raw sliding window over a single string.
* :func:`chunk_document` -- turn one :class:`~app.rag.schemas.Document` into
  :class:`~app.rag.schemas.Chunk` objects.
* :func:`chunk_documents` -- chunk a batch of documents using ``Settings``.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import tiktoken

from app.config import Settings
from app.rag.schemas import Chunk, Document

logger = logging.getLogger(__name__)

# Default tiktoken encoding. ``cl100k_base`` is shared by the OpenAI embedding
# and chat models used elsewhere in the system, so token counts here line up
# with what those models actually consume.
DEFAULT_ENCODING = "cl100k_base"


@lru_cache(maxsize=8)
def _get_encoding(model: str) -> "tiktoken.Encoding":
    """Return a cached tiktoken encoding for ``model``.

    ``model`` may be either a model name (e.g. ``"gpt-4o-mini"``) or an
    encoding name (e.g. ``"cl100k_base"``). We try the encoding-name lookup
    first because the spec passes an encoding name by default, then fall back
    to the model-name lookup, and finally to the default encoding so chunking
    never hard-fails on an unknown identifier.
    """
    try:
        return tiktoken.get_encoding(model)
    except (KeyError, ValueError):
        pass
    try:
        return tiktoken.encoding_for_model(model)
    except (KeyError, ValueError):
        logger.warning(
            "Unknown tiktoken model/encoding %r; falling back to %s",
            model,
            DEFAULT_ENCODING,
        )
        return tiktoken.get_encoding(DEFAULT_ENCODING)


def chunk_text(
    text: str,
    *,
    size: int,
    overlap: int,
    model: str = DEFAULT_ENCODING,
) -> list[str]:
    """Split ``text`` into overlapping token windows.

    A sliding window is moved across the token ids of ``text`` with a stride of
    ``size - overlap``. Each window is decoded back into a string. Empty or
    whitespace-only windows are dropped. A short document that fits within a
    single window yields a single chunk.

    Args:
        text: The raw text to chunk.
        size: Maximum number of tokens per chunk. Must be positive.
        overlap: Number of tokens shared between consecutive chunks. Must be
            non-negative and strictly less than ``size``.
        model: tiktoken model or encoding name used to tokenize.

    Returns:
        A list of decoded chunk strings, in order. Empty list if ``text`` is
        blank.

    Raises:
        ValueError: If ``size`` is not positive or ``overlap`` is out of range.
    """
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    if overlap < 0:
        raise ValueError(f"chunk overlap must be non-negative, got {overlap}")
    if overlap >= size:
        raise ValueError(
            f"chunk overlap ({overlap}) must be strictly less than size ({size})"
        )

    if not text or not text.strip():
        return []

    encoding = _get_encoding(model)
    token_ids = encoding.encode(text)
    if not token_ids:
        return []

    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(token_ids), step):
        window = token_ids[start : start + size]
        if not window:
            continue
        decoded = encoding.decode(window).strip()
        if decoded:
            chunks.append(decoded)
        # Once the window reaches the end of the token stream we are done; any
        # further iterations would only re-emit the tail (or empty windows).
        if start + size >= len(token_ids):
            break

    return chunks


def chunk_document(doc: Document, *, size: int, overlap: int) -> list[Chunk]:
    """Chunk a single :class:`Document` into :class:`Chunk` objects.

    Each chunk inherits the document's source metadata and is assigned a stable
    id of the form ``f"{doc.id}::{chunk_index}"``. The document's ``metadata``
    dict is shallow-copied per chunk so callers can safely mutate one chunk's
    metadata without affecting the others.

    Args:
        doc: The source document.
        size: Maximum tokens per chunk.
        overlap: Token overlap between consecutive chunks.

    Returns:
        A list of chunks. May be empty if the document text is blank.
    """
    texts = chunk_text(doc.text, size=size, overlap=overlap)
    if not texts:
        logger.debug("Document %s produced no chunks (empty text)", doc.id)
        return []

    chunks: list[Chunk] = []
    for index, chunk_text_value in enumerate(texts):
        chunks.append(
            Chunk(
                id=f"{doc.id}::{index}",
                document_id=doc.id,
                text=chunk_text_value,
                source_type=doc.source_type,
                source_id=doc.source_id,
                title=doc.title,
                url=doc.url,
                chunk_index=index,
                metadata=dict(doc.metadata),
            )
        )
    logger.debug("Document %s chunked into %d chunk(s)", doc.id, len(chunks))
    return chunks


def chunk_documents(docs: list[Document], *, settings: Settings) -> list[Chunk]:
    """Chunk a batch of documents using sizes from ``settings``.

    Args:
        docs: Documents to chunk.
        settings: Provides ``chunk_size_tokens`` and ``chunk_overlap_tokens``.

    Returns:
        A flat list of chunks across all documents, in document order.
    """
    size = settings.chunk_size_tokens
    overlap = settings.chunk_overlap_tokens

    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, size=size, overlap=overlap))

    logger.info(
        "Chunked %d document(s) into %d chunk(s) (size=%d, overlap=%d)",
        len(docs),
        len(all_chunks),
        size,
        overlap,
    )
    return all_chunks
