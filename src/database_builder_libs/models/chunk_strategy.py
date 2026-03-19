from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from database_builder_libs.models.chunk import Chunk

# Type alias for the raw section tuples produced by TextStructureExtractor.
# Each entry is (section_title, section_text, list_of_dataframes).
RawSection = tuple[str, str, list[Any]]


# --------------------------------------------------------------------------- #
# Interface                                                                    #
# --------------------------------------------------------------------------- #

class ChunkingStrategy(ABC):
    """
    Interface for all chunking strategies.

    A ChunkingStrategy transforms a sequence of raw document sections into a
    flat list of ``Chunk`` objects that are ready for embedding and indexing.

    Contract
    --------
    - ``chunk()`` is the single method every implementation must provide.
    - Returned ``chunk_index`` values must be monotonically increasing from 0
      and stable across re-indexing runs for identical input.
    - ``text`` on each ``Chunk`` must be non-empty.
    - ``vector`` is left as an empty sequence — embedding is a separate concern.
    - ``metadata`` may carry arbitrary JSON-serialisable fields but must never
      influence chunk identity.

    Implementations
    ---------------
    - :class:`SectionChunkingStrategy`       – one chunk per docling section (default)
    - :class:`FixedSizeChunkingStrategy`      – splits text into fixed-size windows
    - :class:`SlidingWindowChunkingStrategy`  – overlapping fixed-size windows
    - :class:`SummaryAndNSectionsStrategy`    – summary chunk + N evenly-merged body chunks
    """

    @abstractmethod
    def chunk(
        self,
        sections: Sequence[RawSection],
        *,
        document_id: str,
        summary: str | None = None,
    ) -> list[Chunk]:
        """
        Convert raw sections into ``Chunk`` objects.

        Parameters
        ----------
        sections:
            Ordered sequence of ``(title, text, tables)`` tuples as produced
            by ``TextStructureExtractor.extract_sections()``.
        document_id:
            Stable identifier of the parent document.  Passed through
            unchanged into every ``Chunk.document_id``.
        summary:
            Optional pre-extracted summary string (e.g. from
            ``TextMetadata.summary``).  Most strategies ignore this; it is
            consumed by :class:`SummaryAndNSectionsStrategy`.

        Returns
        -------
        list[Chunk]
            Flat, ordered list of chunks.  May be empty if *sections* is empty
            or all sections are blank after cleaning.
        """
        raise NotImplementedError


