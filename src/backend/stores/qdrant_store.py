from typing import Final, Sequence, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

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

    def save(self,
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


if __name__ == "__main__":
    import random
    import uuid

    ds = QdrantDatastore()

    test_vector = [random.random() for _ in range(ds.vector_size)]
    test_id = str(uuid.uuid4())
    test_payload = {"text": "Hello, Qdrant!"}

    print("Saving point...")
    ds.save(id=test_id, vector=test_vector, payload=test_payload)

    print("Point saved successfully!")
