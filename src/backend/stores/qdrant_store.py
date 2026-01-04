from typing import Final
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

    def connect(self) -> None:
        pass

    def save(self, data: dict) -> None:
        point = PointStruct(
            id=data["id"],
            vector=data["vector"],
            payload=data.get("payload"),
        )
        self.client.upsert(
            collection_name=self.collection,
            points=[point],
        )
        
        
        
if __name__ == "__main__":
    from backend.config import settings
    from qdrant_client import QdrantClient

    store = QdrantDatastore()
    test_point = {
        "id": 1,
        "vector": [0.01] * store.vector_size,
        "payload": {"text": "Hello Qdrant from Docker"}
    }
    store.save(test_point)
    print("Saved successfully!")

    client = QdrantClient(url=settings.QDRANT_URL)
    retrieved = client.retrieve(
        collection_name=settings.QDRANT_COLLECTION,
        ids=[test_point["id"]]
    )
    print("Retrieved point from Qdrant:")
    print(retrieved)
