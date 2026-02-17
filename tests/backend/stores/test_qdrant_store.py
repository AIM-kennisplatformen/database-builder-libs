import socket
import time
from unittest.mock import patch

import pytest
from qdrant_client import QdrantClient
from testcontainers.core.container import DockerContainer

from backend.models.chunk import Chunk


# -----------------------------------------------------------
# helpers
# -----------------------------------------------------------

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


# -----------------------------------------------------------
# container
# -----------------------------------------------------------

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


# -----------------------------------------------------------
# fixtures
# -----------------------------------------------------------

@pytest.fixture
def mock_settings(qdrant_container):
    _, url = qdrant_container
    with patch("backend.config.settings") as settings_mock:
        settings_mock.QDRANT_URL = url
        settings_mock.QDRANT_COLLECTION = "test_collection_chunks"
        settings_mock.QDRANT_VECTOR_SIZE = 4
        yield settings_mock


@pytest.fixture
def store(mock_settings):
    from backend.stores.qdrant.qdrant_store import QdrantDatastore
    return QdrantDatastore()


@pytest.fixture(autouse=True)
def clean_collection(store, mock_settings):
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


# -----------------------------------------------------------
# tests
# -----------------------------------------------------------

def test_store_and_retrieve_document_chunks(store):
    chunks = [
        Chunk(document_id="doc1", chunk_index=0, text="hello world", vector=(0.1,0.1,0.1,0.1)),
        Chunk(document_id="doc1", chunk_index=1, text="more text", vector=(0.2,0.2,0.2,0.2)),
    ]

    store.store_chunks(chunks)

    retrieved = store.get_document_chunks("doc1")

    assert len(retrieved) == 2
    assert retrieved[0].chunk_index == 0
    assert retrieved[1].chunk_index == 1
    assert retrieved[0].text == "hello world"


def test_similarity_search(store):
    chunks = [
        Chunk(document_id="doc2", chunk_index=0, text="cats", vector=(0.9,0.0,0.0,0.0)),
        Chunk(document_id="doc3", chunk_index=0, text="dogs", vector=(0.0,0.9,0.0,0.0)),
    ]
    store.store_chunks(chunks)

    results = store.similarity_search((0.9,0.0,0.0,0.0), limit=1)

    assert len(results) == 1
    assert results[0].document_id == "doc2"


def test_delete_document(store):
    chunks = [
        Chunk(document_id="doc4", chunk_index=0, text="temp", vector=(0.3,0.3,0.3,0.3)),
        Chunk(document_id="doc4", chunk_index=1, text="temp2", vector=(0.4,0.4,0.4,0.4)),
    ]

    store.store_chunks(chunks)

    deleted = store.delete_document("doc4")
    assert deleted == 2

    remaining = store.get_document_chunks("doc4")
    assert remaining == []


def test_multiple_documents_isolated(store):
    store.store_chunks([
        Chunk(document_id="docA", chunk_index=0, text="A", vector=(0.1,0.1,0.1,0.1)),
        Chunk(document_id="docB", chunk_index=0, text="B", vector=(0.2,0.2,0.2,0.2)),
    ])

    assert len(store.get_document_chunks("docA")) == 1
    assert len(store.get_document_chunks("docB")) == 1
