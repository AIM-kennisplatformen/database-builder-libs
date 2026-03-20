from __future__ import annotations
from abc import abstractmethod

from pydantic import BaseModel

from database_builder_libs.models.chunk import Chunk


class AbstractChunkEmbedder(BaseModel):
    """
    Interface for all chunk embedders.

    A ChunkEmbedder transforms a list of ``Chunk`` objects (with empty vectors)
    into a list of ``Chunk`` objects with their ``vector`` fields populated,
    ready for indexing into a vector store.

    Contract
    --------
    - ``embed()`` is the single method every implementation must provide.
    - Returned chunks must preserve the original ordering and identity fields
      (``document_id``, ``chunk_index``, ``text``, ``metadata``).
    - The returned list must have the same length as the input list.
    - An empty input list must return an empty list without error.
    - Each returned ``Chunk.vector`` must be non-empty.

    Implementations
    ---------------
    - :class:`OpenAICompatibleChunkEmbedder`  – batched embedding via any /v1/embeddings endpoint
    """

    @abstractmethod
    def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Populate the ``vector`` field of each chunk and return the results.

        Parameters
        ----------
        chunks:
            Ordered list of ``Chunk`` objects whose ``vector`` fields are
            expected to be empty.  May be empty, in which case an empty list
            is returned.

        Returns
        -------
        list[Chunk]
            Flat, ordered list of chunks with ``vector`` populated.  Index *i*
            in the output corresponds to index *i* in *chunks*.
        """
        raise NotImplementedError