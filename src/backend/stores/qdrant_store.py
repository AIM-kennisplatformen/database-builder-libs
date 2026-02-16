from __future__ import annotations

from typing import Final, Sequence, List
from hashlib import blake2b

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from backend.config import settings
from backend.models.abstract_store import AbstractStore
from backend.models.node import Node, NodeId


# key stored inside payload to preserve real identity
_NODE_ID_KEY = "_node_id"


class QdrantDatastore(AbstractStore):
    """
    Adapter translating Domain Node <-> Qdrant points.
    Qdrant is treated purely as a vector index.
    """

    def __init__(self) -> None:
        self.client: Final[QdrantClient] = QdrantClient(url=settings.QDRANT_URL)
        self.collection: Final[str] = settings.QDRANT_COLLECTION
        self.vector_size: Final[int] = settings.QDRANT_VECTOR_SIZE

        self._ensure_collection()

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}

        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def connect_to_source(self) -> None:
        """Health check required by AbstractStore"""
        self.client.get_collections()

    # ------------------------------------------------------------------
    # ID mapping
    # ------------------------------------------------------------------

    
    def _to_point_id(self, node_id: str | NodeId) -> int:
        """
        Convert string domain id to stable integer Qdrant id.
        Deterministic hashing avoids collisions in practice.
        """
        node_id = NodeId(str(node_id))
        digest = blake2b(node_id.encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big", signed=False)

    def _payload_with_identity(self, node: Node) -> dict:
        payload = dict(node.payload_data)
        payload[_NODE_ID_KEY] = node.id
        return payload

    def _node_from_record(self, record) -> Node:
        payload = record.payload or {}
        real_id = payload.get(_NODE_ID_KEY, str(record.id))

        return Node(
            id=NodeId(str(real_id)),
            vector_data=tuple(record.vector or ()),
            payload_data={k: v for k, v in payload.items() if k != _NODE_ID_KEY},
            relations=(),
            embedding_model="qdrant",
        )

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store_node(self, node: Node) -> None:
        if not node.vector_data:
            return

        point = PointStruct(
            id=self._to_point_id(node.id),
            vector=list(node.vector_data),
            payload=self._payload_with_identity(node),
        )

        self.client.upsert(collection_name=self.collection, points=[point])

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def get_nodes(self, filter: str | Sequence[float] | None) -> List[Node]:
        # --------------------------------------------------------------
        # vector search
        # --------------------------------------------------------------
        if isinstance(filter, Sequence) and not isinstance(filter, str):
            response = self.client.query_points(
                collection_name=self.collection,
                query=list(filter),
                limit=10,
                with_vectors=True,
            )
            return [self._node_from_record(p) for p in response.points]

        # --------------------------------------------------------------
        # id filter: "id=abc"
        # --------------------------------------------------------------
        if isinstance(filter, str) and filter.startswith("id="):
            node_id = filter.split("=", 1)[1]
            point_id = self._to_point_id(node_id)

            records = self.client.retrieve(
                collection_name=self.collection,
                ids=[point_id],
                with_vectors=True,
            )
            return [self._node_from_record(r) for r in records]

        # --------------------------------------------------------------
        # scroll all
        # --------------------------------------------------------------
        records, _ = self.client.scroll(
            collection_name=self.collection,
            limit=100,
            with_vectors=True,
        )
        return [self._node_from_record(r) for r in records]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def remove_node(self, filter: str) -> Node:
        if not filter.startswith("id="):
            raise ValueError("QdrantDatastore only supports id filter")

        node_id = filter.split("=", 1)[1]
        point_id = self._to_point_id(node_id)

        records = self.client.retrieve(
            collection_name=self.collection,
            ids=[point_id],
            with_vectors=True,
        )
        if not records:
            raise ValueError(f"Node {node_id} not found")

        node = self._node_from_record(records[0])

        self.client.delete(collection_name=self.collection, points_selector=[point_id])

        return node
