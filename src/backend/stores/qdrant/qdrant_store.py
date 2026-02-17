from __future__ import annotations

from typing import Final, Sequence, List
from hashlib import blake2b

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from backend.config import settings
from backend.models.chunk import Chunk, DocumentId
from backend.models.abstract_vector_store import AbstractVectorStore


DOC_ID = "document_id"
CHUNK_INDEX = "chunk_index"
TEXT = "text"


class QdrantDatastore(AbstractVectorStore):
    """
    Vector store for semantic retrieval.

    Stores ONLY text chunks.
    Does NOT understand knowledge.
    """

    def __init__(self) -> None:
        self.client: Final[QdrantClient] = QdrantClient(url=settings.QDRANT_URL)
        self.collection: Final[str] = settings.QDRANT_COLLECTION
        self.vector_size: Final[int] = settings.QDRANT_VECTOR_SIZE
        self._initialized = False

    # ---------------------------------------------------------
    # lifecycle
    # ---------------------------------------------------------

    def connect(self) -> None:
        """Ensure collection exists (idempotent)."""
        if self._initialized:
            return

        existing = {c.name for c in self.client.get_collections().collections}

        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

        self._initialized = True

    # ---------------------------------------------------------
    # ID generation
    # ---------------------------------------------------------

    def _point_id(self, document_id: DocumentId, chunk_index: int) -> int:
        digest = blake2b(
            f"{document_id}:{chunk_index}".encode(),
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, "big", signed=False)

    # ---------------------------------------------------------
    # Store
    # ---------------------------------------------------------

    def store_chunks(self, chunks: List[Chunk]) -> None:
        self.connect()

        if not chunks:
            return

        points: List[PointStruct] = []

        for chunk in chunks:
            if not chunk.vector:
                continue

            payload = {
                DOC_ID: chunk.document_id,
                CHUNK_INDEX: chunk.chunk_index,
                TEXT: chunk.text,
                **(chunk.metadata or {}),
            }

            points.append(
                PointStruct(
                    id=self._point_id(chunk.document_id, chunk.chunk_index),
                    vector=list(chunk.vector),
                    payload=payload,
                )
            )

        if points:
            self.client.upsert(collection_name=self.collection, points=points)

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def similarity_search(
        self,
        vector: Sequence[float],
        limit: int = 10,
    ) -> List[Chunk]:

        self.connect()

        response = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        results: List[Chunk] = []

        for point in response.points:
            payload = point.payload or {}

            results.append(
                Chunk(
                    document_id=payload[DOC_ID],
                    chunk_index=payload[CHUNK_INDEX],
                    text=payload.get(TEXT, ""),
                    vector=(),  # never return query vector
                    metadata={
                        k: v for k, v in payload.items()
                        if k not in (DOC_ID, CHUNK_INDEX, TEXT)
                    },
                )
            )

        return results

    # ---------------------------------------------------------
    # Retrieve document
    # ---------------------------------------------------------

    def get_document_chunks(self, document_id: DocumentId) -> List[Chunk]:

        self.connect()

        filt = Filter(
            must=[FieldCondition(key=DOC_ID, match=MatchValue(value=document_id))]
        )

        chunks: List[Chunk] = []
        offset = None

        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=filt,
                with_payload=True,
                with_vectors=False,
                limit=512,
                offset=offset,
            )

            for r in records:
                payload = r.payload or {}

                chunks.append(
                    Chunk(
                        document_id=payload[DOC_ID],
                        chunk_index=payload[CHUNK_INDEX],
                        text=payload.get(TEXT, ""),
                        vector=(),
                        metadata={
                            k: v for k, v in payload.items()
                            if k not in (DOC_ID, CHUNK_INDEX, TEXT)
                        },
                    )
                )

            if offset is None:
                break

        return sorted(chunks, key=lambda c: c.chunk_index)

    # ---------------------------------------------------------
    # Delete (GDPR critical)
    # ---------------------------------------------------------

    def delete_document(self, document_id: DocumentId) -> int:

        self.connect()

        chunks = self.get_document_chunks(document_id)
        if not chunks:
            return 0

        filt = Filter(
            must=[FieldCondition(key=DOC_ID, match=MatchValue(value=document_id))]
        )

        self.client.delete(collection_name=self.collection, points_selector=filt)

        return len(chunks)
