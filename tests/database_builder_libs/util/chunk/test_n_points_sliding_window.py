import pytest

from database_builder_libs.utility.chunk.n_points_fixed_size import FixedSizeChunkingStrategy
from database_builder_libs.utility.chunk.n_points_sliding_window import SlidingWindowChunkingStrategy

from .conftest import DOC, section, MEDIUM, LONG, assert_chunk_contract


def test_invalid_overlap_raises():
    """Constructing the strategy with overlap >= chunk_size raises a ValueError mentioning 'overlap'."""
    with pytest.raises(ValueError, match="overlap"):
        SlidingWindowChunkingStrategy(chunk_size=100, overlap=100)


def test_overlap_equal_to_chunk_size_raises():
    """An overlap exactly equal to chunk_size is also invalid and must raise a ValueError."""
    with pytest.raises(ValueError):
        SlidingWindowChunkingStrategy(chunk_size=200, overlap=200)


def test_empty_sections_returns_empty():
    """Chunking an empty section list produces no chunks."""
    assert SlidingWindowChunkingStrategy().chunk([], document_id=DOC) == []


def test_produces_more_chunks_than_fixed():
    """With the same chunk_size, a sliding window with overlap produces at least as many chunks as fixed-size."""
    strat_fixed   = FixedSizeChunkingStrategy(chunk_size=100)
    strat_sliding = SlidingWindowChunkingStrategy(chunk_size=100, overlap=50)
    fixed   = strat_fixed.chunk([LONG], document_id=DOC)
    sliding = strat_sliding.chunk([LONG], document_id=DOC)
    assert len(sliding) >= len(fixed)


def test_contract():
    """All chunks satisfy the core invariants: correct document_id, monotonic index, non-empty text, empty vector."""
    strat = SlidingWindowChunkingStrategy(chunk_size=100, overlap=40)
    chunks = strat.chunk([MEDIUM, LONG], document_id=DOC)
    assert_chunk_contract(chunks, DOC)


def test_consecutive_chunks_share_content():
    """Adjacent chunks overlap: words near the end of chunk N also appear near the start of chunk N+1."""
    text = " ".join([f"word{i}" for i in range(100)])
    strat = SlidingWindowChunkingStrategy(chunk_size=60, overlap=30)
    chunks = strat.chunk([section("S", text)], document_id=DOC)
    assert len(chunks) >= 2
    # The end of chunk N and the start of chunk N+1 should share at least one word.
    end_of_first    = chunks[0].text.split()[-3:]
    start_of_second = chunks[1].text.split()[:3]
    assert any(w in start_of_second for w in end_of_first)


def test_metadata_has_section_title():
    """Every chunk produced from a section carries that section's title in its metadata."""
    strat = SlidingWindowChunkingStrategy(chunk_size=100, overlap=40)
    chunks = strat.chunk([LONG], document_id=DOC)
    assert all(c.metadata["section_title"] == "Methods" for c in chunks)


def test_ignores_summary_kwarg():
    """Passing a summary keyword argument has no effect on sliding-window chunking."""
    strat   = SlidingWindowChunkingStrategy(chunk_size=100, overlap=40)
    without = strat.chunk([LONG], document_id=DOC)
    with_s  = strat.chunk([LONG], document_id=DOC, summary="ignored")
    assert len(without) == len(with_s)