from typing import Any, List

import pytest

from database_builder_libs.models.chunk import Chunk
from database_builder_libs.utility.chunk.n_points_section import SectionChunkingStrategy
from database_builder_libs.utility.chunk.n_points_fixed_size import FixedSizeChunkingStrategy
from database_builder_libs.utility.chunk.n_points_sliding_window import SlidingWindowChunkingStrategy
from database_builder_libs.utility.chunk.summary_n_points import SummaryAndNSectionsStrategy


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def section(title: str, text: str, tables: list | None = None):
    """Convenience factory for a RawSection tuple."""
    return (title, text, tables or [])


def fake_table():
    """Stand-in for a DataFrame — any truthy object is sufficient."""
    return object()


DOC = "doc-001"

# A handful of reusable sections with well-known lengths.
SHORT  = section("Intro", "Too short")           # < 20 chars → dropped by min_chars
MEDIUM = section("Background", "A" * 200)        # fits in one fixed chunk
LONG   = section("Methods", " ".join(["word"] * 300))  # forces multiple fixed chunks

def assert_chunk_contract(chunks: list[Chunk], document_id: str) -> None:
    """Assert invariants that every strategy must satisfy."""
    for i, chunk in enumerate(chunks):
        assert chunk.document_id == document_id,    f"chunk {i}: wrong document_id"
        assert chunk.chunk_index == i,              f"chunk {i}: chunk_index not monotonic"
        assert chunk.text.strip(),                  f"chunk {i}: text is empty"
        assert chunk.vector == [],                  f"chunk {i}: vector should be empty"

def test_section_empty_sections_returns_empty():
    assert SectionChunkingStrategy().chunk([], document_id=DOC) == []


def test_section_one_chunk_per_section():
    sections = [section("A", "A" * 50), section("B", "B" * 50)]
    chunks = SectionChunkingStrategy().chunk(sections, document_id=DOC)
    assert len(chunks) == 2


def test_section_contract():
    sections = [section("A", "A" * 50), section("B", "B" * 50), section("C", "C" * 50)]
    chunks = SectionChunkingStrategy().chunk(sections, document_id=DOC)
    assert_chunk_contract(chunks, DOC)


def test_section_drops_short_sections():
    sections = [SHORT, section("Real", "R" * 50)]
    chunks = SectionChunkingStrategy().chunk(sections, document_id=DOC)
    assert len(chunks) == 1
    assert "R" in chunks[0].text


def test_section_min_chars_boundary():
    strat = SectionChunkingStrategy(min_chars=20)
    dropped = SectionChunkingStrategy().chunk([section("X", "x" * 19)], document_id=DOC)
    kept    = SectionChunkingStrategy().chunk([section("X", "x" * 20)], document_id=DOC)
    assert dropped == []
    assert len(kept) == 1


def test_section_metadata_fields():
    tbl = fake_table()
    sections = [section("Intro", "I" * 50, [tbl]), section("Methods", "M" * 50)]
    chunks = SectionChunkingStrategy().chunk(sections, document_id=DOC)
    assert chunks[0].metadata["section_title"] == "Intro"
    assert chunks[0].metadata["has_tables"] is True
    assert chunks[1].metadata["has_tables"] is False


def test_section_include_title_in_text():
    strat = SectionChunkingStrategy(include_title_in_text=True)
    chunks = strat.chunk([section("MyTitle", "body text here padded")], document_id=DOC)
    assert chunks[0].text.startswith("MyTitle\n")


def test_section_without_title_not_prepended():
    strat = SectionChunkingStrategy(include_title_in_text=True)
    chunks = strat.chunk([section("", "body text here padded enough")], document_id=DOC)
    assert chunks[0].text == "body text here padded enough"


def test_section_ignores_summary_kwarg():
    sections = [section("A", "A" * 50)]
    without = SectionChunkingStrategy().chunk(sections, document_id=DOC)
    with_summary = SectionChunkingStrategy().chunk(sections, document_id=DOC, summary="ignored")
    assert len(without) == len(with_summary)


def test_section_chunk_index_stable_across_runs():
    sections = [section("A", "A" * 50), section("B", "B" * 50)]
    run1 = SectionChunkingStrategy().chunk(sections, document_id=DOC)
    run2 = SectionChunkingStrategy().chunk(sections, document_id=DOC)
    assert [c.chunk_index for c in run1] == [c.chunk_index for c in run2]


def test_fixed_empty_sections_returns_empty():
    assert FixedSizeChunkingStrategy().chunk([], document_id=DOC) == []


def test_fixed_short_text_produces_one_chunk():
    chunks = FixedSizeChunkingStrategy(chunk_size=500).chunk([MEDIUM], document_id=DOC)
    assert len(chunks) == 1


def test_fixed_long_text_produces_multiple_chunks():
    chunks = FixedSizeChunkingStrategy(chunk_size=100).chunk([LONG], document_id=DOC)
    assert len(chunks) > 1


def test_fixed_contract():
    chunks = FixedSizeChunkingStrategy(chunk_size=100).chunk([MEDIUM, LONG], document_id=DOC)
    assert_chunk_contract(chunks, DOC)


def test_fixed_no_chunk_exceeds_chunk_size():
    strat = FixedSizeChunkingStrategy(chunk_size=150)
    chunks = strat.chunk([LONG], document_id=DOC)
    # Allow one word of slack for whitespace-boundary splits.
    for chunk in chunks:
        assert len(chunk.text) <= 150 + 20, f"chunk too long: {len(chunk.text)}"


def test_fixed_drops_windows_below_min_chars():
    # One long section followed by a stub that will produce a tiny tail window.
    strat = FixedSizeChunkingStrategy(chunk_size=1000, min_chars=20)
    tiny = section("T", "x" * 10)  # 10 chars < min_chars=20
    chunks = strat.chunk([tiny], document_id=DOC)
    assert chunks == []


def test_fixed_metadata_has_section_title():
    chunks = FixedSizeChunkingStrategy(chunk_size=100).chunk([LONG], document_id=DOC)
    assert all(c.metadata["section_title"] == "Methods" for c in chunks)


def test_fixed_sections_chunked_independently():
    s1 = section("A", "alpha " * 50)
    s2 = section("B", "beta " * 50)
    chunks = FixedSizeChunkingStrategy(chunk_size=100).chunk([s1, s2], document_id=DOC)
    titles = [c.metadata["section_title"] for c in chunks]
    assert "A" in titles and "B" in titles


def test_fixed_ignores_summary_kwarg():
    chunks_without = FixedSizeChunkingStrategy(chunk_size=100).chunk([LONG], document_id=DOC)
    chunks_with    = FixedSizeChunkingStrategy(chunk_size=100).chunk([LONG], document_id=DOC, summary="ignored")
    assert len(chunks_without) == len(chunks_with)


def test_sliding_invalid_overlap_raises():
    with pytest.raises(ValueError, match="overlap"):
        SlidingWindowChunkingStrategy(chunk_size=100, overlap=100)


def test_sliding_overlap_equal_to_chunk_size_raises():
    with pytest.raises(ValueError):
        SlidingWindowChunkingStrategy(chunk_size=200, overlap=200)


def test_sliding_empty_sections_returns_empty():
    assert SlidingWindowChunkingStrategy().chunk([], document_id=DOC) == []


def test_sliding_produces_more_chunks_than_fixed():
    strat_fixed   = FixedSizeChunkingStrategy(chunk_size=100)
    strat_sliding = SlidingWindowChunkingStrategy(chunk_size=100, overlap=50)
    fixed   = strat_fixed.chunk([LONG], document_id=DOC)
    sliding = strat_sliding.chunk([LONG], document_id=DOC)
    assert len(sliding) >= len(fixed)


def test_sliding_contract():
    strat = SlidingWindowChunkingStrategy(chunk_size=100, overlap=40)
    chunks = strat.chunk([MEDIUM, LONG], document_id=DOC)
    assert_chunk_contract(chunks, DOC)


def test_sliding_consecutive_chunks_share_content():
    text = " ".join([f"word{i}" for i in range(100)])
    strat = SlidingWindowChunkingStrategy(chunk_size=60, overlap=30)
    chunks = strat.chunk([section("S", text)], document_id=DOC)
    assert len(chunks) >= 2
    # The end of chunk N and the start of chunk N+1 should share at least one word.
    end_of_first   = chunks[0].text.split()[-3:]
    start_of_second = chunks[1].text.split()[:3]
    assert any(w in start_of_second for w in end_of_first)


def test_sliding_metadata_has_section_title():
    strat = SlidingWindowChunkingStrategy(chunk_size=100, overlap=40)
    chunks = strat.chunk([LONG], document_id=DOC)
    assert all(c.metadata["section_title"] == "Methods" for c in chunks)


def test_sliding_ignores_summary_kwarg():
    strat = SlidingWindowChunkingStrategy(chunk_size=100, overlap=40)
    without = strat.chunk([LONG], document_id=DOC)
    with_s  = strat.chunk([LONG], document_id=DOC, summary="ignored")
    assert len(without) == len(with_s)


def test_summary_n_invalid_n_raises():
    with pytest.raises(ValueError, match="n_sections"):
        SummaryAndNSectionsStrategy(n_sections=0)


def test_summary_n_empty_sections_no_summary_returns_empty():
    assert SummaryAndNSectionsStrategy().chunk([], document_id=DOC) == []


def test_summary_n_empty_sections_with_summary_returns_one_chunk():
    chunks = SummaryAndNSectionsStrategy().chunk([], document_id=DOC, summary="A summary.")
    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_type"] == "summary"


def test_summary_n_summary_is_first_chunk():
    sections = [section("A", "A" * 200), section("B", "B" * 200)]
    chunks = SummaryAndNSectionsStrategy(n_sections=2).chunk(
        sections, document_id=DOC, summary="Overview of the document."
    )
    assert chunks[0].metadata["chunk_type"] == "summary"
    assert chunks[0].text == "Overview of the document."


def test_summary_n_summary_index_is_zero():
    sections = [section("A", "A" * 200)]
    chunks = SummaryAndNSectionsStrategy(n_sections=1).chunk(
        sections, document_id=DOC, summary="Summary text here."
    )
    assert chunks[0].chunk_index == 0


def test_summary_n_body_chunks_follow_summary():
    sections = [section("A", "A" * 500)]
    chunks = SummaryAndNSectionsStrategy(n_sections=3).chunk(
        sections, document_id=DOC, summary="Short summary."
    )
    body_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "body"]
    assert len(body_chunks) == 3


def test_summary_n_produces_exactly_n_body_chunks():
    # Use a text long enough that none of the partitions fall below min_chars.
    text = "word " * 200
    sections = [section("S", text)]
    for n in (1, 3, 5):
        chunks = SummaryAndNSectionsStrategy(n_sections=n).chunk(sections, document_id=DOC)
        assert len(chunks) == n, f"expected {n} chunks, got {len(chunks)}"


def test_summary_n_no_summary_body_starts_at_index_zero():
    sections = [section("A", "A" * 200)]
    chunks = SummaryAndNSectionsStrategy(n_sections=2).chunk(sections, document_id=DOC)
    assert chunks[0].chunk_index == 0
    assert chunks[0].metadata["chunk_type"] == "body"


def test_summary_n_contract_with_summary():
    sections = [section("A", "A" * 300), section("B", "B" * 300)]
    chunks = SummaryAndNSectionsStrategy(n_sections=3).chunk(
        sections, document_id=DOC, summary="Summary here."
    )
    assert_chunk_contract(chunks, DOC)


def test_summary_n_contract_without_summary():
    sections = [section("A", "A" * 300), section("B", "B" * 300)]
    chunks = SummaryAndNSectionsStrategy(n_sections=3).chunk(sections, document_id=DOC)
    assert_chunk_contract(chunks, DOC)


def test_summary_n_body_metadata_chunk_type():
    sections = [section("A", "A" * 300)]
    chunks = SummaryAndNSectionsStrategy(n_sections=2).chunk(sections, document_id=DOC)
    assert all(c.metadata["chunk_type"] == "body" for c in chunks)


def test_summary_n_body_metadata_partition_index_sequential():
    sections = [section("A", "A " * 200)]
    chunks = SummaryAndNSectionsStrategy(n_sections=3).chunk(sections, document_id=DOC)
    partition_indices = [c.metadata["partition_index"] for c in chunks]
    assert partition_indices == list(range(len(chunks)))


def test_summary_n_body_has_tables_flag_propagated():
    tbl = fake_table()
    sections = [section("A", "A " * 100, [tbl]), section("B", "B " * 100)]
    chunks = SummaryAndNSectionsStrategy(n_sections=2).chunk(sections, document_id=DOC)
    # has_tables is True because at least one section has a table.
    assert all(c.metadata["has_tables"] is True for c in chunks)


def test_summary_n_whitespace_only_summary_omitted():
    sections = [section("A", "A" * 200)]
    chunks = SummaryAndNSectionsStrategy(n_sections=1).chunk(
        sections, document_id=DOC, summary="   "
    )
    assert all(c.metadata.get("chunk_type") != "summary" for c in chunks)


def test_summary_n_chunk_index_stable_across_runs():
    sections = [section("A", "word " * 100)]
    run1 = SummaryAndNSectionsStrategy(n_sections=3).chunk(
        sections, document_id=DOC, summary="Summary."
    )
    run2 = SummaryAndNSectionsStrategy(n_sections=3).chunk(
        sections, document_id=DOC, summary="Summary."
    )
    assert [c.chunk_index for c in run1] == [c.chunk_index for c in run2]