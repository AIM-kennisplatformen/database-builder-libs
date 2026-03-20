import pytest

from database_builder_libs.models.chunk import Chunk


DOC = "doc-001"
MEDIUM = ("Background", "A" * 200, [])          # fits in one fixed chunk
LONG   = ("Methods", " ".join(["word"] * 300), [])  # forces multiple fixed chunks

def section(title: str, text: str, tables: list | None = None):
    """Convenience factory for a RawSection tuple."""
    return (title, text, tables or [])


def fake_table():
    """Stand-in for a DataFrame — any truthy object is sufficient."""
    return object()


def assert_chunk_contract(chunks: list[Chunk], document_id: str) -> None:
    """Assert invariants that every strategy must satisfy."""
    for i, chunk in enumerate(chunks):
        assert chunk.document_id == document_id,    f"chunk {i}: wrong document_id"
        assert chunk.chunk_index == i,              f"chunk {i}: chunk_index not monotonic"
        assert chunk.text.strip(),                  f"chunk {i}: text is empty"
        assert chunk.vector == [],                  f"chunk {i}: vector should be empty"