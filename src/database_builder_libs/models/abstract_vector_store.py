from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, List

from database_builder_libs.models.chunk import Chunk, DocumentId


class AbstractVectorStore(ABC):
    """
    Semantic retrieval storage for embedding-based search.

    The store persists document chunks together with their embeddings
    and supports nearest-neighbour semantic retrieval.

    The interface is designed to be backend-agnostic and compatible with:
    FAISS, Qdrant, Pinecone, pgvector, Weaviate, Elasticsearch, etc.

    Consistency guarantees
    ----------------------
    Implementations MUST guarantee:

    - Deterministic retrieval for identical index state
    - No duplicate chunks returned
    - Stable chunk identity across writes
    - Full deletion of document vectors (GDPR requirement)

    Embedding contract
    ------------------
    All stored vectors must:
    - Have identical dimensionality
    - Use the same distance metric
    - Be normalized if required by backend

    Mixing embedding models in one index is forbidden.
    """

    def __init__(self) -> None:
        self._connected: bool = False
        self._connecting: bool = False

    def connect(self, config: dict | None = None) -> None:
        """
        Initialize the vector index and verify accessibility.

        This method should:
        - Create index if missing
        - Validate embedding dimensionality
        - Validate distance metric compatibility

        Raises
        ------
        ConnectionError
            Backend unreachable.
        RuntimeError
            Index exists but is incompatible.
        """
        if self._connected:
            return

        self._connecting = True
        try:
            self._connect_impl(config)
            self._connected = True
        finally:
            self._connecting = False

    @abstractmethod
    def _connect_impl(self, config: dict | None) -> None:
        """Backend specific connection logic"""
        raise NotImplementedError

    def _ensure_connected(self) -> None:
        if not (self._connected or self._connecting):
            raise RuntimeError(
                f"{self.__class__.__name__} used before connect() was called"
            )

    @abstractmethod
    def store_chunks(self, chunks: List[Chunk]) -> None:
        """
        Insert or update chunks and their embeddings.

        Behaviour
        ---------
        - Operation must be idempotent
        - Existing chunks with same (document_id, chunk_id) MUST be overwritten
        - Partial document updates are allowed

        Parameters
        ----------
        chunks : List[Chunk]
            Chunks containing text, metadata, and embedding vector.

        Raises
        ------
        RuntimeError
            If called before connect().
        ValueError
            If embedding dimensionality mismatch occurs.
        """
        raise NotImplementedError

    @abstractmethod
    def similarity_search(
        self,
        vector: Sequence[float],
        limit: int = 10,
    ) -> List[Chunk]:
        """
        Perform nearest-neighbour semantic search.

        Parameters
        ----------
        vector : Sequence[float]
            Query embedding. Must match index dimensionality.
        limit : int
            Maximum number of results to return.

        Returns
        -------
        List[Chunk]
            Ordered by similarity descending (most relevant first).

        Guarantees
        ----------
        - At most `limit` results returned
        - No duplicate chunks
        - Ordering must reflect backend similarity score

        Raises
        ------
        ValueError
            If vector dimensionality mismatch.
        RuntimeError
            If store not connected.
        """
        raise NotImplementedError

    @abstractmethod
    def get_document_chunks(self, document_id: DocumentId) -> List[Chunk]:
        """
        Retrieve all chunks belonging to a document.

        Returns
        -------
        List[Chunk]
            All chunks for the document ordered by original document order.

        Raises
        ------
        KeyError
            If document does not exist.
        RuntimeError
            If store not connected.
        """

        raise NotImplementedError

    @abstractmethod
    def delete_document(self, document_id: DocumentId) -> int:
        """
        Permanently remove all vectors belonging to a document.

        This operation must be irreversible and guarantee that the
        document cannot appear in future search results.

        Parameters
        ----------
        document_id : DocumentId
            Identifier of the document to delete.

        Returns
        -------
        int
            Number of deleted chunks.

        GDPR Requirement
        ----------------
        After successful deletion, similarity_search() MUST NOT return
        any chunk originating from this document.

        Raises
        ------
        RuntimeError
            If deletion could not be fully verified.
        """
        raise NotImplementedError
