import pytest

from database_builder_libs.utility.chunk.summary_and_sections import SummaryAndSectionsStrategy

from .conftest import DOC, section, fake_table, assert_chunk_contract


def test_empty_sections_no_summary_returns_empty():
    """No sections and no summary yields an empty chunk list."""
    assert SummaryAndSectionsStrategy().chunk([], document_id=DOC) == []


def test_empty_sections_with_summary_returns_one_chunk():
    """A summary provided with no body sections produces exactly one summary chunk."""
    chunks = SummaryAndSectionsStrategy().chunk([], document_id=DOC, summary="A summary.")
    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_type"] == "summary"


def test_summary_is_first_chunk():
    """When a summary is provided it appears as the first chunk, before all body chunks."""
    sections = [section("A", "A" * 200), section("B", "B" * 200)]
    chunks = SummaryAndSectionsStrategy().chunk(
        sections, document_id=DOC, summary="Overview of the document."
    )
    assert chunks[0].metadata["chunk_type"] == "summary"
    assert chunks[0].text == "Overview of the document."


def test_summary_index_is_zero():
    """The summary chunk is always assigned chunk_index 0."""
    sections = [section("A", "A" * 200)]
    chunks = SummaryAndSectionsStrategy().chunk(
        sections, document_id=DOC, summary="Summary text here."
    )
    assert chunks[0].chunk_index == 0


def test_body_chunks_follow_summary():
    """Body chunks appear after the summary chunk with correct count."""
    sections = [section("A", "A" * 200), section("B", "B" * 200), section("C", "C" * 200)]
    chunks = SummaryAndSectionsStrategy().chunk(
        sections, document_id=DOC, summary="Short summary."
    )
    body_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "body"]
    assert len(body_chunks) == 3


def test_produces_one_chunk_per_section():
    """The strategy produces exactly one body chunk per non-empty section."""
    sections = [section("A", "A" * 200), section("B", "B" * 200), section("C", "C" * 200)]
    chunks = SummaryAndSectionsStrategy().chunk(sections, document_id=DOC)
    assert len(chunks) == 3


def test_no_summary_body_starts_at_index_zero():
    """Without a summary the first body chunk is assigned chunk_index 0."""
    sections = [section("A", "A" * 200)]
    chunks = SummaryAndSectionsStrategy().chunk(sections, document_id=DOC)
    assert chunks[0].chunk_index == 0
    assert chunks[0].metadata["chunk_type"] == "body"


def test_contract_with_summary():
    """All chunks satisfy the core invariants when a summary is included."""
    sections = [section("A", "A" * 300), section("B", "B" * 300)]
    chunks = SummaryAndSectionsStrategy().chunk(
        sections, document_id=DOC, summary="Summary here."
    )
    assert_chunk_contract(chunks, DOC)


def test_contract_without_summary():
    """All chunks satisfy the core invariants when no summary is provided."""
    sections = [section("A", "A" * 300), section("B", "B" * 300)]
    chunks = SummaryAndSectionsStrategy().chunk(sections, document_id=DOC)
    assert_chunk_contract(chunks, DOC)


def test_body_metadata_chunk_type():
    """Every body chunk has chunk_type set to 'body' in its metadata."""
    sections = [section("A", "A" * 300), section("B", "B" * 300)]
    chunks = SummaryAndSectionsStrategy().chunk(sections, document_id=DOC)
    assert all(c.metadata["chunk_type"] == "body" for c in chunks)


def test_body_metadata_section_title_preserved():
    """Each body chunk carries the section_title of its originating section."""
    sections = [section("Introduction", "A" * 200), section("Methods", "B" * 200)]
    chunks = SummaryAndSectionsStrategy().chunk(sections, document_id=DOC)
    titles = [c.metadata["section_title"] for c in chunks]
    assert titles == ["Introduction", "Methods"]


def test_body_has_tables_flag_per_section():
    """The has_tables flag reflects the individual section, not the whole document."""
    tbl = fake_table()
    sections = [section("A", "A " * 100, [tbl]), section("B", "B " * 100)]
    chunks = SummaryAndSectionsStrategy().chunk(sections, document_id=DOC)
    assert chunks[0].metadata["has_tables"] is True
    assert chunks[1].metadata["has_tables"] is False


def test_whitespace_only_summary_omitted():
    """A summary consisting entirely of whitespace is treated as absent and no summary chunk is emitted."""
    sections = [section("A", "A" * 200)]
    chunks = SummaryAndSectionsStrategy().chunk(
        sections, document_id=DOC, summary="   "
    )
    assert all(c.metadata.get("chunk_type") != "summary" for c in chunks)


def test_short_sections_dropped():
    """Sections whose text is shorter than min_chars after stripping are silently dropped."""
    sections = [section("A", "A" * 200), section("B", "x")]
    chunks = SummaryAndSectionsStrategy(min_chars=20).chunk(sections, document_id=DOC)
    assert len(chunks) == 1
    assert chunks[0].metadata["section_title"] == "A"


def test_chunk_index_stable_across_runs():
    """Chunk indices are deterministic: repeated calls on the same input yield identical orderings."""
    sections = [section("A", "word " * 100), section("B", "word " * 100)]
    run1 = SummaryAndSectionsStrategy().chunk(sections, document_id=DOC, summary="Summary.")
    run2 = SummaryAndSectionsStrategy().chunk(sections, document_id=DOC, summary="Summary.")
    assert [c.chunk_index for c in run1] == [c.chunk_index for c in run2]


def test_body_chunk_indices_sequential():
    """Body chunk indices increment sequentially after the summary chunk."""
    sections = [section("A", "A" * 200), section("B", "B" * 200), section("C", "C" * 200)]
    chunks = SummaryAndSectionsStrategy().chunk(
        sections, document_id=DOC, summary="Summary."
    )
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))