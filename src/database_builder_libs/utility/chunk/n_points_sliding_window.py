from __future__ import annotations
 
from dataclasses import dataclass
from typing import Sequence
 
from database_builder_libs.models.chunk import Chunk
from database_builder_libs.models.chunk_strategy import ChunkingStrategy, RawSection

@dataclass(slots=True)
class SlidingWindowChunkingStrategy(ChunkingStrategy):
    """
    Produces overlapping character windows across each section's text.

    Overlapping windows preserve cross-boundary context that non-overlapping
    splits lose, at the cost of index size and some retrieval redundancy.
    Useful for dense technical text where important sentences often span what
    would otherwise be a hard split boundary.

    Attributes
    ----------
    chunk_size : int
        Target maximum number of characters per window.  Default: 1000.
    overlap : int
        Number of characters shared between consecutive windows.
        Must be strictly less than ``chunk_size``.  Default: 200.
    min_chars : int
        Windows shorter than this threshold are dropped.  Default: 20.

    Raises
    ------
    ValueError
        If ``overlap >= chunk_size``.
    """

    chunk_size: int = 1000
    overlap: int = 200
    min_chars: int = 20

    def __post_init__(self) -> None:
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be strictly less than "
                f"chunk_size ({self.chunk_size})."
            )

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

            for window in self._slide(text):
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

    def _slide(self, text: str) -> list[str]:
        """Yield overlapping windows of ``chunk_size`` characters."""
        step = self.chunk_size - self.overlap
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

            next_start = start + step
            while next_start < len(text) and text[next_start] == " ":
                next_start += 1
            start = next_start

        return [w for w in windows if w]

