from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from database_builder_libs.models.chunk import Chunk
from database_builder_libs.models.abstract_chunk_strategy import AbstractChunkingStrategy, RawSection


@dataclass(slots=True)
class SummaryAndSectionsStrategy(AbstractChunkingStrategy):
    """
    Produces one optional summary chunk followed by one chunk per section.

    Combines the summary prepending of :class:`SummaryAndNSectionsStrategy`
    with the section-preserving behaviour of :class:`SectionChunkingStrategy`.
    Section boundaries, titles, and table flags are all preserved on each body
    chunk.

    Chunk layout (when summary is present)
    ---------------------------------------
    index 0      →  summary text
    index 1..N   →  one chunk per non-empty section, in document order

    Chunk layout (when summary is absent)
    --------------------------------------
    index 0..N   →  one chunk per non-empty section, in document order

    Attributes
    ----------
    min_chars : int
        Sections whose text is shorter than this after stripping are silently
        dropped.  Default: 20.
    """

    min_chars: int = 20

    def chunk(
        self,
        sections: Sequence[RawSection],
        *,
        document_id: str,
        summary: str | None = None,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0

        if summary:
            summary = summary.strip()
            if summary:
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        chunk_index=index,
                        text=summary,
                        vector=[],
                        metadata={"chunk_type": "summary"},
                    )
                )
                index += 1

        for title, text, tables in sections:
            text = text.strip()
            if len(text) < self.min_chars:
                continue

            chunks.append(
                Chunk(
                    document_id=document_id,
                    chunk_index=index,
                    text=text,
                    vector=[],
                    metadata={
                        "chunk_type": "body",
                        "section_title": title,
                        "has_tables": bool(tables),
                    },
                )
            )
            index += 1

        return chunks