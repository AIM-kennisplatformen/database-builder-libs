from typing import Final, Sequence, Optional
from qdrant_client.conversions.common_types import PointId
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    PointIdsList,
    ScoredPoint,
    Record,
)

from backend.config import settings
from backend.stores.store import Datastore


class QdrantDatastore(Datastore):
    def __init__(self) -> None:
        self.client: Final[QdrantClient] = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=None,
        )
        self.collection: Final[str] = settings.QDRANT_COLLECTION
        self.vector_size: Final[int] = settings.QDRANT_VECTOR_SIZE

        assert self.client is not None, "Qdrant client is not set."
        assert self.collection is not None, "Qdrant collection is not set."
        assert self.vector_size is not None, "Qdrant vector size is not set."

        collections = self.client.get_collections().collections
        existing = {c.name for c in collections}

        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def save(
        self,
        id: str | int,
        vector: Sequence[float],
        payload: Optional[dict] = None,
    ) -> None:
        """Save a single point to the collection."""
        point = PointStruct(
            id=id,
            vector=list(vector),
            payload=payload,
        )
        self.client.upsert(
            collection_name=self.collection,
            points=[point],
        )

    def delete_by_id(self, ids: PointIdsList) -> None:
        """Delete points from the collection based on their IDs."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=ids,
        )

    def delete_by_filter(self, filter: Filter) -> None:
        """Delete points from the collection based on a filter."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=filter,
        )

    def query(
        self,
        query_vector: Sequence[float],
        limit: int = 10,
        filter: Optional[Filter] = None,
    ) -> list[ScoredPoint]:
        """Query the collection for points similar to the query vector."""
        response = self.client.query_points(
            collection_name=self.collection,
            query=[float(v) for v in query_vector],
            limit=limit,
            query_filter=filter,
        )
        return response.points

    def scroll(
        self,
        limit: int = 10,
        filter: Optional[Filter] = None,
    ) -> tuple[list[Record], PointId | None]:
        """Scroll through the collection sorted by ascending ID."""
        return self.client.scroll(
            collection_name=self.collection,
            limit=limit,
            scroll_filter=filter,
        )
