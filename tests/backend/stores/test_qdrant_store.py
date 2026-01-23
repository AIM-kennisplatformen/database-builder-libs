import pytest
import time
import socket
from unittest.mock import patch
from testcontainers.core.container import DockerContainer
from qdrant_client import QdrantClient


def wait_for_port(host: str, port: int, timeout: float = 60.0):
    """
    Continuously tries to connect to the host:port until successful.
    """
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
    """
    Starts a Qdrant container and waits for the client to be ready.
    """
    container = DockerContainer("qdrant/qdrant:latest")
    container.with_exposed_ports(6333)
    container.start()

    host = "127.0.0.1"
    port = container.get_exposed_port(6333)
    url = f"http://{host}:{port}"

    try:
        print(f"   [DEBUG] Waiting for Qdrant port {port}...")
        wait_for_port(host, port, timeout=60)

        client = QdrantClient(url=url)
        max_retries = 30
        for _ in range(max_retries):
            try:
                client.get_collections()
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("Could not connect to Qdrant.")

    except Exception as e:
        stdout, stderr = container.get_logs()
        print("\n=== CONTAINER LOGS ===")
        print(stdout.decode())
        print(stderr.decode())
        raise e

    yield container, url
    container.stop()


@pytest.fixture
def mock_settings(qdrant_container):
    container, url = qdrant_container

    with patch("backend.config.settings") as settings_mock:
        settings_mock.QDRANT_URL = url
        settings_mock.QDRANT_COLLECTION = "test_collection_vectors"
        settings_mock.QDRANT_VECTOR_SIZE = 4
        yield settings_mock


@pytest.fixture
def store(mock_settings):
    from backend.stores.qdrant_store import QdrantDatastore

    return QdrantDatastore()


def test_initialization_creates_collection(store, mock_settings):
    collections_info = store.client.get_collections()
    collection_names = [c.name for c in collections_info.collections]
    assert mock_settings.QDRANT_COLLECTION in collection_names


def test_save_and_query_workflow(store):
    vector = [0.1, 0.2, 0.3, 0.4]
    payload = {"city": "Berlin", "country": "Germany"}

    store.save(id=1, vector=vector, payload=payload)

    query_vector = [0.1, 0.2, 0.3, 0.4]
    results = store.query(query_vector=query_vector, limit=1)

    assert len(results) >= 1
    point = results[0]
    assert point.id == 1
    assert point.payload["city"] == "Berlin"


def test_delete_workflow(store):
    vector = [0.9, 0.9, 0.9, 0.9]
    store.save(id=99, vector=vector, payload={"temp": "data"})

    results = store.query(query_vector=vector, limit=1)
    assert len(results) > 0
    assert results[0].id == 99

    store.delete(ids=[99])

    points, _ = store.scroll(limit=100)
    ids = [p.id for p in points]
    assert 99 not in ids


def test_scroll_workflow(store):
    vectors = [
        (10, [0.0, 0.0, 0.0, 0.1], {"idx": 10}),
        (11, [0.0, 0.0, 0.0, 0.2], {"idx": 11}),
        (12, [0.0, 0.0, 0.0, 0.3], {"idx": 12}),
    ]

    for uid, vec, pay in vectors:
        store.save(uid, vec, pay)

    result = store.scroll(limit=10)

    points, next_page_offset = result

    ids = {p.id for p in points}
    assert 10 in ids
    assert 11 in ids
    assert 12 in ids
