"""Tests for the RAG primitives (spec §17).

Covers:

* :func:`app.rag.chunking.chunk_text` -- sliding-window size/overlap counts,
  short-document and empty-input handling, and invalid-argument guards.
* :func:`app.rag.hybrid.reciprocal_rank_fusion` -- RRF ranking maths, dedup by
  ``chunk.id``, ``retriever`` tagging, and ``top_n`` truncation.
* :class:`app.rag.bm25_index.BM25Index` -- build/query ranking, the empty-index
  contract (never crash), tokenisation, and the ``from_chunks`` constructor.

These tests use ``tiktoken`` (offline) and never touch the network.
"""

from __future__ import annotations

import tiktoken

from app.rag.bm25_index import BM25Index, tokenize
from app.rag.chunking import chunk_text
from app.rag.hybrid import reciprocal_rank_fusion
from app.rag.schemas import Chunk, ScoredChunk


def _encoding() -> "tiktoken.Encoding":
    """Return the shared ``cl100k_base`` encoding used by the chunker."""

    return tiktoken.get_encoding("cl100k_base")


def _make_chunk(chunk_id: str, text: str = "placeholder text") -> Chunk:
    """Build a minimal :class:`Chunk` for fusion/index tests."""

    return Chunk(
        id=chunk_id,
        document_id="doc",
        text=text,
        source_type="resume",
        source_id="resume",
        title="Resume",
        url=None,
        chunk_index=0,
        metadata={},
    )


# --------------------------------------------------------------------------- #
# chunk_text
# --------------------------------------------------------------------------- #
def test_chunk_text_window_and_overlap_counts() -> None:
    """A long text is sliced into the expected number of fixed-size windows.

    600 tokens with size=100, overlap=20 → stride 80. Windows start at
    0,80,...,560 → 8 chunks. All but the last hold a full 100 tokens; the last
    holds the 40-token remainder.
    """

    enc = _encoding()
    text = " ".join(f"word{i}" for i in range(300))
    total_tokens = len(enc.encode(text))
    assert total_tokens == 600  # guard the fixture stays deterministic

    chunks = chunk_text(text, size=100, overlap=20)

    assert len(chunks) == 8
    token_lengths = [len(enc.encode(c)) for c in chunks]
    assert token_lengths[:-1] == [100] * 7
    assert token_lengths[-1] == 40


def test_chunk_text_overlap_increases_chunk_count() -> None:
    """A larger overlap shrinks the stride, producing more chunks.

    For a 400-token text and ``size=50``: ``overlap=0`` (stride 50) yields 8
    windows, while ``overlap=10`` (stride 40) yields 10. The extra chunks are
    the observable, deterministic effect of the sliding-window overlap.
    """

    enc = _encoding()
    text = " ".join(f"token{i}" for i in range(200))
    assert len(enc.encode(text)) == 400  # guard the fixture stays deterministic

    no_overlap = chunk_text(text, size=50, overlap=0)
    with_overlap = chunk_text(text, size=50, overlap=10)

    assert len(no_overlap) == 8
    assert len(with_overlap) == 10
    assert len(with_overlap) > len(no_overlap)


def test_chunk_text_short_document_single_chunk() -> None:
    """A document that fits in one window yields exactly one chunk."""

    chunks = chunk_text("a short sentence about engineering", size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0].strip() == "a short sentence about engineering"


def test_chunk_text_empty_and_whitespace() -> None:
    """Empty / whitespace-only text yields no chunks."""

    assert chunk_text("", size=100, overlap=10) == []
    assert chunk_text("   \n\t  ", size=100, overlap=10) == []


def test_chunk_text_invalid_arguments_raise() -> None:
    """Invalid size/overlap combinations raise ``ValueError``."""

    import pytest

    with pytest.raises(ValueError):
        chunk_text("hello world", size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_text("hello world", size=10, overlap=-1)
    with pytest.raises(ValueError):
        chunk_text("hello world", size=10, overlap=10)


# --------------------------------------------------------------------------- #
# reciprocal_rank_fusion
# --------------------------------------------------------------------------- #
def test_rrf_ranking_dedup_and_tagging() -> None:
    """RRF fuses two lists, dedups by id, tags ``rrf``, and ranks correctly.

    list A: c1(rank0), c2(rank1), c3(rank2)
    list B: c2(rank0), c4(rank1), c1(rank2)
    With k=60:
        c2 = 1/61 + 1/60 ≈ 0.033060
        c1 = 1/60 + 1/62 ≈ 0.032796
    so c2 outranks c1, and c4/c3 trail. Four unique chunks survive.
    """

    list_a = [
        ScoredChunk(_make_chunk("c1"), 0.9, "vector"),
        ScoredChunk(_make_chunk("c2"), 0.8, "vector"),
        ScoredChunk(_make_chunk("c3"), 0.7, "vector"),
    ]
    list_b = [
        ScoredChunk(_make_chunk("c2"), 5.0, "bm25"),
        ScoredChunk(_make_chunk("c4"), 4.0, "bm25"),
        ScoredChunk(_make_chunk("c1"), 3.0, "bm25"),
    ]

    fused = reciprocal_rank_fusion([list_a, list_b], k=60)

    ids = [sc.chunk.id for sc in fused]
    assert len(fused) == 4  # dedup: c1,c2 appear in both lists
    assert set(ids) == {"c1", "c2", "c3", "c4"}
    assert ids[0] == "c2"  # highest fused score
    assert ids[1] == "c1"
    assert all(sc.retriever == "rrf" for sc in fused)
    # Scores are strictly descending.
    scores = [sc.score for sc in fused]
    assert scores == sorted(scores, reverse=True)
    # Verify the exact c2 vs c1 maths.
    score_by_id = {sc.chunk.id: sc.score for sc in fused}
    assert score_by_id["c2"] == 1 / 61 + 1 / 60
    assert score_by_id["c1"] == 1 / 60 + 1 / 62
    assert score_by_id["c2"] > score_by_id["c1"]


def test_rrf_top_n_truncation() -> None:
    """``top_n`` limits the fused output to the best N results."""

    list_a = [ScoredChunk(_make_chunk(f"c{i}"), 1.0 / (i + 1), "vector") for i in range(5)]
    fused = reciprocal_rank_fusion([list_a], k=60, top_n=2)
    assert len(fused) == 2
    assert [sc.chunk.id for sc in fused] == ["c0", "c1"]


def test_rrf_keeps_first_seen_chunk_object() -> None:
    """Dedup keeps the first-seen :class:`Chunk` payload for a repeated id."""

    first = _make_chunk("dup", text="first text")
    second = _make_chunk("dup", text="second text")
    fused = reciprocal_rank_fusion(
        [[ScoredChunk(first, 0.9, "vector")], [ScoredChunk(second, 0.1, "bm25")]],
        k=60,
    )
    assert len(fused) == 1
    assert fused[0].chunk.text == "first text"
    # Score accumulates contributions from both lists (rank 0 in each).
    assert fused[0].score == 1 / 60 + 1 / 60


def test_rrf_empty_input() -> None:
    """Fusing empty / no lists yields an empty result."""

    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


# --------------------------------------------------------------------------- #
# tokenize
# --------------------------------------------------------------------------- #
def test_tokenize_lowercases_and_drops_short_tokens() -> None:
    """Tokeniser lowercases, splits on ``\\w+``, and drops length<2 tokens."""

    tokens = tokenize("Hello, A WORLD of x9 codes!")
    assert tokens == ["hello", "world", "of", "x9", "codes"]
    assert "a" not in tokens  # single-character token dropped
    assert tokenize("") == []


# --------------------------------------------------------------------------- #
# BM25Index
# --------------------------------------------------------------------------- #
def test_bm25_build_and_query_ranks_relevant_first() -> None:
    """A built index ranks lexically-relevant chunks first and tags ``bm25``."""

    chunks = [
        _make_chunk("p1", "python machine learning models"),
        _make_chunk("p2", "golang web servers and networking"),
        _make_chunk("p3", "python data pipelines and analytics"),
    ]
    index = BM25Index()
    index.build(chunks)

    assert index.size == 3

    results = index.query("python", top_k=3)
    assert results, "expected at least one BM25 hit for 'python'"
    assert all(r.retriever == "bm25" for r in results)
    top_ids = [r.chunk.id for r in results[:2]]
    # The two python chunks should outrank the golang chunk.
    assert "p1" in top_ids and "p3" in top_ids
    assert results[-1].chunk.id == "p2"
    # Scores are descending floats.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(r.score, float) for r in results)


def test_bm25_respects_top_k() -> None:
    """``top_k`` bounds the number of returned hits."""

    chunks = [_make_chunk(f"c{i}", f"shared keyword chunk number {i}") for i in range(5)]
    index = BM25Index.from_chunks(chunks)
    results = index.query("keyword", top_k=2)
    assert len(results) == 2


def test_bm25_empty_index_returns_empty() -> None:
    """An unbuilt or empty index returns ``[]`` instead of raising."""

    unbuilt = BM25Index()
    assert unbuilt.size == 0
    assert unbuilt.query("anything", top_k=5) == []

    from_empty = BM25Index.from_chunks([])
    assert from_empty.size == 0
    assert from_empty.query("anything", top_k=5) == []


def test_bm25_query_without_usable_tokens_returns_empty() -> None:
    """A query with no usable tokens (all length<2) returns ``[]``."""

    index = BM25Index.from_chunks([_make_chunk("c1", "python engineering")])
    # "a" tokenises to nothing (length < 2).
    assert index.query("a", top_k=5) == []


def test_bm25_save_and_load_roundtrip(tmp_path) -> None:
    """An index round-trips through ``save``/``load`` preserving query behaviour."""

    chunks = [
        _make_chunk("p1", "python machine learning"),
        _make_chunk("p2", "rust systems programming"),
    ]
    index = BM25Index.from_chunks(chunks)
    path = str(tmp_path / "bm25.pkl")
    index.save(path)

    loaded = BM25Index.load(path)
    assert loaded.size == 2
    results = loaded.query("python", top_k=2)
    assert results
    assert results[0].chunk.id == "p1"
    assert results[0].chunk.text == "python machine learning"
