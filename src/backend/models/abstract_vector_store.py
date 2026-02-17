from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, List

from backend.models.chunk import Chunk, DocumentId


class AbstractVectorStore(ABC):
    """
    Semantic retrieval storage.

    Stores text chunks with embeddings.
    """

    # ---------------------------------------------------------
    # lifecycle
    # ---------------------------------------------------------

    @abstractmethod
    def connect(self) -> None:
        """Ensure index exists and is reachable."""
        raise NotImplementedError

    # ---------------------------------------------------------
    # write
    # ---------------------------------------------------------

    @abstractmethod
    def store_chunks(self, chunks: List[Chunk]) -> None:
        """
        Insert or overwrite embeddings for a document.
        Must be idempotent.
        """
        raise NotImplementedError

    # ---------------------------------------------------------
    # retrieval
    # ---------------------------------------------------------

    @abstractmethod
    def similarity_search(
        self,
        vector: Sequence[float],
        limit: int = 10,
    ) -> List[Chunk]:
        """Semantic nearest neighbour search."""
        raise NotImplementedError

    @abstractmethod
    def get_document_chunks(self, document_id: DocumentId) -> List[Chunk]:
        """Retrieve all chunks belonging to one document."""
        raise NotImplementedError

    # ---------------------------------------------------------
    # deletion (GDPR critical)
    # ---------------------------------------------------------

    @abstractmethod
    def delete_document(self, document_id: DocumentId) -> int:
        """Remove all vectors for a document."""
        raise NotImplementedError
