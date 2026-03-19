import pytest

from database_builder_libs.utility.chunk.summary_n_points import SummaryAndNSectionsStrategy

from .conftest import DOC, section, fake_table, assert_chunk_contract


def test_invalid_n_raises():
    """Constructing the strategy with n_sections=0 raises a ValueError mentioning 'n_sections'."""
    with pytest.raises(ValueError, match="n_sections"):
        SummaryAndNSectionsStrategy(n_sections=0)


def test_empty_sections_no_summary_returns_empty():
    """No sections and no summary yields an empty chunk list."""
    assert SummaryAndNSectionsStrategy().chunk([], document_id=DOC) == []


def test_empty_sections_with_summary_returns_one_chunk():
    """A summary provided with no body sections produces exactly one summary chunk."""
    chunks = SummaryAndNSectionsStrategy().chunk([], document_id=DOC, summary="A summary.")
    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_type"] == "summary"


def test_summary_is_first_chunk():
    """When a summary is provided it appears as the first chunk, before all body chunks."""
    sections = [section("A", "A" * 200), section("B", "B" * 200)]
    chunks = SummaryAndNSectionsStrategy(n_sections=2).chunk(
        sections, document_id=DOC, summary="Overview of the document."
    )
    assert chunks[0].metadata["chunk_type"] == "summary"
    assert chunks[0].text == "Overview of the document."


def test_summary_index_is_zero():
    """The summary chunk is always assigned chunk_index 0."""
    sections = [section("A", "A" * 200)]
    chunks = SummaryAndNSectionsStrategy(n_sections=1).chunk(
        sections, document_id=DOC, summary="Summary text here."
    )
    assert chunks[0].chunk_index == 0


def test_body_chunks_follow_summary():
    """Body chunks appear after the summary chunk and the correct count is produced."""
    sections = [section("A", "A" * 500)]
    chunks = SummaryAndNSectionsStrategy(n_sections=3).chunk(
        sections, document_id=DOC, summary="Short summary."
    )
    body_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "body"]
    assert len(body_chunks) == 3


def test_produces_exactly_n_body_chunks():
    """The strategy always partitions the body text into exactly n_sections chunks regardless of n."""
    text = "word " * 200
    sections = [section("S", text)]
    for n in (1, 3, 5):
        chunks = SummaryAndNSectionsStrategy(n_sections=n).chunk(sections, document_id=DOC)
        assert len(chunks) == n, f"expected {n} chunks, got {len(chunks)}"


def test_no_summary_body_starts_at_index_zero():
    """Without a summary the first body chunk is assigned chunk_index 0."""
    sections = [section("A", "A" * 200)]
    chunks = SummaryAndNSectionsStrategy(n_sections=2).chunk(sections, document_id=DOC)
    assert chunks[0].chunk_index == 0
    assert chunks[0].metadata["chunk_type"] == "body"


def test_contract_with_summary():
    """All chunks satisfy the core invariants when a summary is included."""
    sections = [section("A", "A" * 300), section("B", "B" * 300)]
    chunks = SummaryAndNSectionsStrategy(n_sections=3).chunk(
        sections, document_id=DOC, summary="Summary here."
    )
    assert_chunk_contract(chunks, DOC)


def test_contract_without_summary():
    """All chunks satisfy the core invariants when no summary is provided."""
    sections = [section("A", "A" * 300), section("B", "B" * 300)]
    chunks = SummaryAndNSectionsStrategy(n_sections=3).chunk(sections, document_id=DOC)
    assert_chunk_contract(chunks, DOC)


def test_body_metadata_chunk_type():
    """Every body chunk has chunk_type set to 'body' in its metadata."""
    sections = [section("A", "A" * 300)]
    chunks = SummaryAndNSectionsStrategy(n_sections=2).chunk(sections, document_id=DOC)
    assert all(c.metadata["chunk_type"] == "body" for c in chunks)


def test_body_metadata_partition_index_sequential():
    """Body chunks carry a partition_index that increments sequentially from 0."""
    sections = [section("A", "A " * 200)]
    chunks = SummaryAndNSectionsStrategy(n_sections=3).chunk(sections, document_id=DOC)
    partition_indices = [c.metadata["partition_index"] for c in chunks]
    assert partition_indices == list(range(len(chunks)))


def test_body_has_tables_flag_propagated():
    """If any input section contains tables, all body chunks reflect has_tables=True."""
    tbl = fake_table()
    sections = [section("A", "A " * 100, [tbl]), section("B", "B " * 100)]
    chunks = SummaryAndNSectionsStrategy(n_sections=2).chunk(sections, document_id=DOC)
    assert all(c.metadata["has_tables"] is True for c in chunks)


def test_whitespace_only_summary_omitted():
    """A summary consisting entirely of whitespace is treated as absent and no summary chunk is emitted."""
    sections = [section("A", "A" * 200)]
    chunks = SummaryAndNSectionsStrategy(n_sections=1).chunk(
        sections, document_id=DOC, summary="   "
    )
    assert all(c.metadata.get("chunk_type") != "summary" for c in chunks)


def test_chunk_index_stable_across_runs():
    """Chunk indices are deterministic: repeated calls on the same input yield identical orderings."""
    sections = [section("A", "word " * 100)]
    run1 = SummaryAndNSectionsStrategy(n_sections=3).chunk(
        sections, document_id=DOC, summary="Summary."
    )
    run2 = SummaryAndNSectionsStrategy(n_sections=3).chunk(
        sections, document_id=DOC, summary="Summary."
    )
    assert [c.chunk_index for c in run1] == [c.chunk_index for c in run2]