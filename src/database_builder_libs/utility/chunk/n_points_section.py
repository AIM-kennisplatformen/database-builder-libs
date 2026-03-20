from __future__ import annotations
 
from dataclasses import dataclass
from typing import Sequence
 
from database_builder_libs.models.chunk import Chunk
from database_builder_libs.models.abstract_chunk_strategy import AbstractChunkingStrategy, RawSection

@dataclass(slots=True)
class SectionChunkingStrategy(AbstractChunkingStrategy):
    """
    Produces exactly one ``Chunk`` per non-empty document section.

    This is the simplest strategy and maps cleanly onto the heading structure
    that Docling extracts.  It is the best default when sections are already
    semantically coherent units (e.g. academic papers, reports).

    Attributes
    ----------
    min_chars : int
        Sections whose text is shorter than this threshold (after stripping)
        are silently dropped.  Prevents index pollution from stub sections
        such as lone headings with no body.  Default: 20.
    include_title_in_text : bool
        When ``True`` the section title is prepended to the chunk text as
        ``"<title>\\n<text>"``.  Useful when the title adds retrieval signal
        that does not appear in the body.  Default: ``False``.
    """

    min_chars: int = 20
    include_title_in_text: bool = False

    def chunk(
        self,
        sections: Sequence[RawSection],
        *,
        document_id: str,
        summary: str | None = None,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0

        for title, text, tables in sections:
            text = text.strip()

            if len(text) < self.min_chars:
                continue

            body = f"{title}\n{text}" if (self.include_title_in_text and title) else text

            chunks.append(
                Chunk(
                    document_id=document_id,
                    chunk_index=index,
                    text=body,
                    vector=[],
                    metadata={
                        "section_title": title,
                        "has_tables": bool(tables),
                    },
                )
            )
            index += 1

        return chunks