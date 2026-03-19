from database_builder_libs.utility.chunk.n_points_section import SectionChunkingStrategy

from .conftest import DOC, section, fake_table, assert_chunk_contract


def test_empty_sections_returns_empty():
    """Chunking an empty section list produces no chunks."""
    assert SectionChunkingStrategy().chunk([], document_id=DOC) == []


def test_one_chunk_per_section():
    """Each section above the minimum length yields exactly one chunk."""
    sections = [section("A", "A" * 50), section("B", "B" * 50)]
    chunks = SectionChunkingStrategy().chunk(sections, document_id=DOC)
    assert len(chunks) == 2


def test_contract():
    """All chunks satisfy the core invariants: correct document_id, monotonic index, non-empty text, empty vector."""
    sections = [section("A", "A" * 50), section("B", "B" * 50), section("C", "C" * 50)]
    chunks = SectionChunkingStrategy().chunk(sections, document_id=DOC)
    assert_chunk_contract(chunks, DOC)


def test_drops_short_sections():
    """Sections whose text falls below min_chars are silently skipped."""
    short = section("Intro", "Too short")
    sections = [short, section("Real", "R" * 50)]
    chunks = SectionChunkingStrategy().chunk(sections, document_id=DOC)
    assert len(chunks) == 1
    assert "R" in chunks[0].text


def test_min_chars_boundary():
    """A section at exactly min_chars is kept; one character below is dropped."""
    dropped = SectionChunkingStrategy().chunk([section("X", "x" * 19)], document_id=DOC)
    kept    = SectionChunkingStrategy().chunk([section("X", "x" * 20)], document_id=DOC)
    assert dropped == []
    assert len(kept) == 1


def test_metadata_fields():
    """Each chunk carries its section title and a boolean flag indicating whether tables were present."""
    tbl = fake_table()
    sections = [section("Intro", "I" * 50, [tbl]), section("Methods", "M" * 50)]
    chunks = SectionChunkingStrategy().chunk(sections, document_id=DOC)
    assert chunks[0].metadata["section_title"] == "Intro"
    assert chunks[0].metadata["has_tables"] is True
    assert chunks[1].metadata["has_tables"] is False


def test_include_title_in_text():
    """When include_title_in_text is True, the section title is prepended to the chunk text."""
    strat = SectionChunkingStrategy(include_title_in_text=True)
    chunks = strat.chunk([section("MyTitle", "body text here padded")], document_id=DOC)
    assert chunks[0].text.startswith("MyTitle\n")


def test_without_title_not_prepended():
    """When the section title is empty, no title prefix is added even with include_title_in_text=True."""
    strat = SectionChunkingStrategy(include_title_in_text=True)
    chunks = strat.chunk([section("", "body text here padded enough")], document_id=DOC)
    assert chunks[0].text == "body text here padded enough"


def test_ignores_summary_kwarg():
    """Passing a summary keyword argument has no effect on section-based chunking."""
    sections = [section("A", "A" * 50)]
    without      = SectionChunkingStrategy().chunk(sections, document_id=DOC)
    with_summary = SectionChunkingStrategy().chunk(sections, document_id=DOC, summary="ignored")
    assert len(without) == len(with_summary)


def test_chunk_index_stable_across_runs():
    """Chunk indices are deterministic: repeated calls on the same input yield identical orderings."""
    sections = [section("A", "A" * 50), section("B", "B" * 50)]
    run1 = SectionChunkingStrategy().chunk(sections, document_id=DOC)
    run2 = SectionChunkingStrategy().chunk(sections, document_id=DOC)
    assert [c.chunk_index for c in run1] == [c.chunk_index for c in run2]