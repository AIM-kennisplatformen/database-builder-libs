from database_builder_libs.utility.chunk.n_points_fixed_size import FixedSizeChunkingStrategy

from .conftest import DOC, section, MEDIUM, LONG, assert_chunk_contract


def test_empty_sections_returns_empty():
    """Chunking an empty section list produces no chunks."""
    assert FixedSizeChunkingStrategy().chunk([], document_id=DOC) == []


def test_short_text_produces_one_chunk():
    """Text shorter than chunk_size is returned as a single chunk without splitting."""
    chunks = FixedSizeChunkingStrategy(chunk_size=500).chunk([MEDIUM], document_id=DOC)
    assert len(chunks) == 1


def test_long_text_produces_multiple_chunks():
    """Text longer than chunk_size is split into more than one chunk."""
    chunks = FixedSizeChunkingStrategy(chunk_size=100).chunk([LONG], document_id=DOC)
    assert len(chunks) > 1


def test_contract():
    """All chunks satisfy the core invariants: correct document_id, monotonic index, non-empty text, empty vector."""
    chunks = FixedSizeChunkingStrategy(chunk_size=100).chunk([MEDIUM, LONG], document_id=DOC)
    assert_chunk_contract(chunks, DOC)


def test_no_chunk_exceeds_chunk_size():
    """No individual chunk's text length exceeds chunk_size, allowing a small word-boundary slack."""
    strat = FixedSizeChunkingStrategy(chunk_size=150)
    chunks = strat.chunk([LONG], document_id=DOC)
    # Allow one word of slack for whitespace-boundary splits.
    for chunk in chunks:
        assert len(chunk.text) <= 150 + 20, f"chunk too long: {len(chunk.text)}"


def test_drops_windows_below_min_chars():
    """Windows whose text length falls below min_chars are discarded rather than emitted as tiny chunks."""
    strat = FixedSizeChunkingStrategy(chunk_size=1000, min_chars=20)
    tiny = section("T", "x" * 10)  # 10 chars < min_chars=20
    chunks = strat.chunk([tiny], document_id=DOC)
    assert chunks == []


def test_metadata_has_section_title():
    """Every chunk produced from a section carries that section's title in its metadata."""
    chunks = FixedSizeChunkingStrategy(chunk_size=100).chunk([LONG], document_id=DOC)
    assert all(c.metadata["section_title"] == "Methods" for c in chunks)


def test_sections_chunked_independently():
    """Chunks from different sections are kept separate and labelled with their own section title."""
    s1 = section("A", "alpha " * 50)
    s2 = section("B", "beta " * 50)
    chunks = FixedSizeChunkingStrategy(chunk_size=100).chunk([s1, s2], document_id=DOC)
    titles = [c.metadata["section_title"] for c in chunks]
    assert "A" in titles and "B" in titles


def test_ignores_summary_kwarg():
    """Passing a summary keyword argument has no effect on fixed-size chunking."""
    chunks_without = FixedSizeChunkingStrategy(chunk_size=100).chunk([LONG], document_id=DOC)
    chunks_with    = FixedSizeChunkingStrategy(chunk_size=100).chunk([LONG], document_id=DOC, summary="ignored")
    assert len(chunks_without) == len(chunks_with)