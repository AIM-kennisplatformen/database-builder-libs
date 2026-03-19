from __future__ import annotations
 
from dataclasses import dataclass
from typing import Sequence
 
from database_builder_libs.models.chunk import Chunk
from database_builder_libs.models.abstract_chunk_strategy import AbstractChunkingStrategy, RawSection

@dataclass(slots=True)
class SummaryAndNSectionsStrategy(AbstractChunkingStrategy):
    """
    Produces one summary chunk followed by exactly ``n_sections`` body chunks.

    The summary chunk (index 0) is built from the ``summary`` kwarg passed to
    ``chunk()``.  When no summary is provided the chunk is omitted and body
    chunks start at index 0.

    All section texts are concatenated in document order and then divided as
    evenly as possible into ``n_sections`` fixed-size windows, splitting only
    at whitespace boundaries.  This gives a predictable, bounded index size
    regardless of how many docling sections the document contains — useful
    when downstream retrieval assumes a fixed budget of chunks per document.

    Chunk layout (when summary is present)
    ---------------------------------------
    index 0  →  summary text
    index 1  →  body partition 1  (≈ total_chars / n_sections)
    index 2  →  body partition 2
    ...
    index N  →  body partition N  (absorbs any remainder)

    Attributes
    ----------
    n_sections : int
        Number of body chunks to produce from the merged section text.
        Must be ≥ 1.  Default: 5.
    min_chars : int
        Body windows shorter than this after splitting are silently dropped.
        The final chunk count may therefore be less than ``n_sections`` for
        very short documents.  Default: 20.

    Raises
    ------
    ValueError
        If ``n_sections < 1``.
    """

    n_sections: int = 5
    min_chars: int = 20

    def __post_init__(self) -> None:
        if self.n_sections < 1:
            raise ValueError(f"n_sections must be ≥ 1, got {self.n_sections}.")

    def chunk(
        self,
        sections: Sequence[RawSection],
        *,
        document_id: str,
        summary: str | None = None,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0

        # ── chunk 0: summary ────────────────────────────────────────────────
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

        # ── body: merge all section texts, then split into N even windows ───
        full_text = "\n\n".join(
            text.strip()
            for _, text, _ in sections
            if text.strip()
        )

        if not full_text:
            return chunks

        has_tables = any(bool(tables) for _, _, tables in sections)

        for partition_index, window in enumerate(
            self._even_split(full_text, self.n_sections)
        ):
            window = window.strip()
            if len(window) < self.min_chars:
                continue

            chunks.append(
                Chunk(
                    document_id=document_id,
                    chunk_index=index,
                    text=window,
                    vector=[],
                    metadata={
                        "chunk_type": "body",
                        "partition_index": partition_index,
                        "has_tables": has_tables,
                    },
                )
            )
            index += 1

        return chunks

    def _even_split(self, text: str, n: int) -> list[str]:
        """
        Divide *text* into *n* roughly equal partitions on whitespace
        boundaries.

        The target size of each partition is ``ceil(len(text) / n)``.  If a
        word boundary cannot be found within the target window the split falls
        back to a hard character cut so the method always terminates.  The
        final partition absorbs any remaining characters.
        """
        if not text:
            return []

        target = -(-len(text) // n)
        windows: list[str] = []
        start = 0

        for _ in range(n - 1):
            if start >= len(text):
                break

            end = start + target
            if end >= len(text):
                windows.append(text[start:])
                start = len(text)
                break

            boundary = text.rfind(" ", start, end)
            if boundary <= start:
                boundary = end 

            windows.append(text[start:boundary])
            start = boundary + 1 

        if start < len(text):
            windows.append(text[start:])

        return [w for w in windows if w]