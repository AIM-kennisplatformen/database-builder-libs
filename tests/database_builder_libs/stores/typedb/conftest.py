import os
import pytest
import time
import socket
import uuid
from testcontainers.core.container import DockerContainer
from typedb.driver import Credentials, DriverOptions, TypeDB
from database_builder_libs.models.node import Node
from database_builder_libs.stores.typedb.typedb_store import TypeDbDatastore


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
def typedb_container():
    """
    Starts a TypeDB 2.29 container and waits for the driver to be ready.
    """
    container = (
        DockerContainer("typedb/typedb:3.8.0")
        .with_exposed_ports(1729)
        .with_env("JAVA_TOOL_OPTIONS", "-Xmx1g")
    )

    container.start()
    host = "127.0.0.1"
    port = container.get_exposed_port(1729)
    address = f"{host}:{port}"
    credentials = Credentials("admin", "password")
    driver_options = DriverOptions(is_tls_enabled=False)

    try:
        print(f"   [DEBUG] Waiting for TCP port {address}...")
        wait_for_port(host, port, timeout=60)

        max_retries = 30
        for _ in range(max_retries):
            try:
                with TypeDB.driver(address, credentials, driver_options) as driver:
                    driver.databases.all()
                    break
            except Exception as e:
                print(f"   [DEBUG] Driver connection failed: {e}. Retrying...")
                time.sleep(1)
        else:
            raise RuntimeError("Driver could not connect to TypeDB.")

    except Exception as e:
        stdout, stderr = container.get_logs()
        print("\n=== CONTAINER LOGS ===")
        print(stdout.decode())
        print(stderr.decode())
        raise e

    yield address
    container.stop()


@pytest.fixture
def store(typedb_container):
    address = typedb_container

    datastore = TypeDbDatastore()
    schema_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "schema.tql")
    db_name = f"test_db_{uuid.uuid4().hex}"
    datastore.connect(
        {
            "uri": address,
            "database": db_name,
            "schema_path": schema_path,
            "username": "admin",
            "password": "password",
            "tls": False,
        }
    )

    yield datastore

    datastore._entity_attr_cache = {}
    datastore._all_attr_cache = None
    datastore._key_attr_cache = {}

    if datastore.typedb_driver and datastore.typedb_driver.is_open():
        try:
            datastore.typedb_driver.databases.get(db_name).delete()
        except Exception as e:
            print(f"Failed to delete test db {db_name}: {e}")


@pytest.fixture
def friendship_store(store: TypeDbDatastore) -> TypeDbDatastore:
    store.query_schema("""
        define
        relation friendship,
            relates friend,
            relates friend_of;

        person plays friendship:friend;
        person plays friendship:friend_of;
        """)
    return store


@pytest.fixture
def populated_friendship_store(friendship_store: TypeDbDatastore) -> TypeDbDatastore:
    friendship_store.query_write(
        """
        insert
            $a isa person, has name "Alice", has email "alice@test.com";
            $b isa person, has name "Bob", has email "bob@test.com";
            $f isa friendship, links (friend: $a, friend_of: $b);
        """
    )
    return friendship_store


@pytest.fixture
def alice_node() -> Node:
    return Node(
        id="Alice",
        entity_type="person",
        key_attribute="name",
        payload_data={"name": "Alice", "email": "alice@test.com", "age": 25},
        relations=(),
    )


@pytest.fixture
def bob_with_friendship_node() -> Node:
    return Node(
        id="Bob",
        entity_type="person",
        key_attribute="name",
        payload_data={"name": "Bob", "email": "bob@test.com", "age": 30},
        relations=(
            {
                "type": "friendship",
                "roles": {
                    "friend": {
                        "entity_type": "person",
                        "key_attr": "name",
                        "key": "Alice",
                    },
                    "friend_of": {
                        "entity_type": "person",
                        "key_attr": "name",
                        "key": "Bob",
                    },
                },
            },
        ),
    )
