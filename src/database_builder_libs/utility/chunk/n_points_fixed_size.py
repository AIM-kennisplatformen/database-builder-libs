from __future__ import annotations
 
from dataclasses import dataclass
from typing import Sequence
 
from database_builder_libs.models.chunk import Chunk
from database_builder_libs.models.abstract_chunk_strategy import AbstractChunkingStrategy, RawSection

@dataclass(slots=True)
class FixedSizeChunkingStrategy(AbstractChunkingStrategy):
    """
    Splits section text into non-overlapping fixed-size character windows.

    Each section may produce one or more chunks depending on its length
    relative to ``chunk_size``.  Splits are made on whitespace boundaries
    wherever possible to avoid cutting words mid-token.

    Attributes
    ----------
    chunk_size : int
        Target maximum number of characters per chunk.  Default: 1000.
    min_chars : int
        Windows shorter than this are dropped (typically the last fragment of
        a short section).  Default: 20.
    """

    chunk_size: int = 1000
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

        for title, text, tables in sections:
            text = text.strip()

            if not text:
                continue

            for window in self._split(text):
                if len(window) < self.min_chars:
                    continue

                chunks.append(
                    Chunk(
                        document_id=document_id,
                        chunk_index=index,
                        text=window,
                        vector=[],
                        metadata={
                            "section_title": title,
                            "has_tables": bool(tables),
                        },
                    )
                )
                index += 1

        return chunks

    def _split(self, text: str) -> list[str]:
        """Split *text* into windows of at most ``chunk_size`` characters."""
        windows: list[str] = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                windows.append(text[start:].strip())
                break

            boundary = text.rfind(" ", start, end)
            if boundary <= start:
                boundary = end

            windows.append(text[start:boundary].strip())
            start = boundary + 1 

        return [w for w in windows if w]

