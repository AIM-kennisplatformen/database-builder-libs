import socket
import time
from unittest.mock import patch

import pytest
from qdrant_client import QdrantClient
from testcontainers.core.container import DockerContainer

from backend.models.node import Node


def wait_for_port(host: str, port: int, timeout: float = 60.0) -> None:
    start_time = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except (OSError, ConnectionRefusedError) as exc:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Port {port} on {host} did not open within {timeout} seconds."
                ) from exc
            time.sleep(0.5)


@pytest.fixture(scope="module")
def qdrant_container():
    container = DockerContainer("qdrant/qdrant:latest").with_exposed_ports(6333)
    container.start()

    host = "127.0.0.1"
    port = int(container.get_exposed_port(6333))
    url = f"http://{host}:{port}"

    try:
        wait_for_port(host, port, timeout=60)

        client = QdrantClient(url=url)
        for _ in range(30):
            try:
                client.get_collections()
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("Could not connect to Qdrant.")
    except Exception:
        stdout, stderr = container.get_logs()
        print("\n=== CONTAINER LOGS ===")
        print(stdout.decode(errors="ignore"))
        print(stderr.decode(errors="ignore"))
        raise

    yield container, url
    container.stop()


@pytest.fixture
def mock_settings(qdrant_container):
    _, url = qdrant_container
    with patch("backend.config.settings") as settings_mock:
        settings_mock.QDRANT_URL = url
        settings_mock.QDRANT_COLLECTION = "test_collection_vectors"
        settings_mock.QDRANT_VECTOR_SIZE = 4
        yield settings_mock


@pytest.fixture
def store(mock_settings):
    from backend.stores.qdrant_store import QdrantDatastore

    return QdrantDatastore()


@pytest.fixture(autouse=True)
def clean_collection(store, mock_settings):
    # Defensive: delete might fail if it doesn't exist
    try:
        store.client.delete_collection(mock_settings.QDRANT_COLLECTION)
    except Exception:
        pass

    store.client.create_collection(
        collection_name=mock_settings.QDRANT_COLLECTION,
        vectors_config={
            "size": mock_settings.QDRANT_VECTOR_SIZE,
            "distance": "Cosine",
        },
    )
    yield


def test_connect_to_source(store):
    store.connect_to_source()  # should not raise


def test_store_node_workflow(store):
    node = Node(
        id="1001",
        vector_data=(0.5, 0.5, 0.5, 0.5),
        payload_data={"type": "test_node"},
        relations=(),
        embedding_model="test",
    )

    store.store_node(node)

    # Retrieval by payload filter (contract-level)
    results = store.get_nodes("id=1001")

    assert len(results) == 1
    assert results[0].id == "1001"
    assert results[0].payload_data["type"] == "test_node"


def test_get_nodes_by_vector(store):
    node = Node(
        id="1002",
        vector_data=(0.2, 0.2, 0.2, 0.2),
        payload_data={"category": "vector_test"},
        relations=(),
        embedding_model="test",
    )

    store.store_node(node)

    results = store.get_nodes((0.2, 0.2, 0.2, 0.2))

    assert any(n.id == "1002" for n in results)


def test_get_nodes_scroll_all(store):
    # Should return list[Node], possibly empty
    results = store.get_nodes(None)
    assert isinstance(results, list)
    if results:
        assert isinstance(results[0], Node)


def test_remove_node_by_payload_filter(store):
    node = Node(
        id="1003",
        vector_data=(0.7, 0.7, 0.7, 0.7),
        payload_data={"status": "temporary"},
        relations=(),
        embedding_model="test",
    )
    store.store_node(node)

    removed = store.remove_node("id=1003")

    assert isinstance(removed, Node)
    assert removed.id == "1003"

    # Ensure it's gone
    results = store.get_nodes("id=1003")
    assert results == []
