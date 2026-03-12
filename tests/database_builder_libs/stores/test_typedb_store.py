import os
import pytest
import time
import socket
from testcontainers.core.container import DockerContainer
from typedb.driver import ConceptRowIterator, Credentials, DriverOptions, QueryAnswer, Transaction, TypeDB, TransactionType
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
    datastore.connect(
        {
            "uri": address,
            "database": "integration_test_db",
            "schema_path": schema_path,
            "username": "admin",
            "password": "password",
            "tls": False,
        }
    )

    # clean test data AFTER connect
    with datastore._query(TransactionType.WRITE) as tx:
        tx.query("match $x isa person; delete $x isa person;")

    yield datastore


def test_initialization_creates_database_and_schema(store):
    tx: Transaction
    with store._query(TransactionType.READ) as tx:
        rows: ConceptRowIterator = tx.query("match entity $t sub person;").resolve().as_concept_rows()
        row = next(rows, None)
        assert row is not None
        person_type = row.get("t")

        assert person_type is not None
        assert person_type.get_label() == "person"

def test_requires_connect(typedb_container):
    store = TypeDbDatastore()

    with pytest.raises(RuntimeError):
        store.get_nodes("entity=person")


def test_save_and_fetch_workflow(store):
    store.save('insert $p isa person, has name "Alice", has email "alice@test.com";')

    results = store.fetch(
        'match $p isa person, has name "Alice"; fetch $p: name, email;'
    )

    assert len(results) == 1
    data = results[0]["p"]

    assert data["name"][0]["value"] == "Alice"
    assert data["email"][0]["value"] == "alice@test.com"


def test_get_workflow(store):
    store.save('insert $p isa person, has name "Bob", has age 30;')
    results = store.get('match $p isa person, has name "Bob"; get $p;')

    assert len(results) > 0


def test_update_workflow(store):
    store.save('insert $p isa person, has name "Charlie", has age 20;')

    update_query = """
    match 
        $p isa person, has name "Charlie", has age $old_age; 
        $old_age = 20;
    delete 
        $p has $old_age;
    insert 
        $p has age 21;
    """
    store.update(update_query)

    results = store.fetch('match $p isa person, has name "Charlie"; fetch $p: age;')

    assert results[0]["p"]["age"][0]["value"] == 21


def test_delete_workflow(store):
    store.save('insert $p isa person, has name "Dave";')
    assert (
        len(store.fetch('match $p isa person, has name "Dave"; fetch $p: name;')) == 1
    )

    store.delete('match $p isa person, has name "Dave"; delete $p isa person;')

    results = store.fetch('match $p isa person, has name "Dave"; fetch $p: name;')
    assert len(results) == 0


def test_get_nodes_by_keyed_filter(store):
    # Arrange
    store.save(
        'insert $p isa person, has name "Alice", has email "alice@test.com", has age 25;'
    )

    # Act
    results = store.get_nodes("entity=person&name=Alice")

    # Assert
    assert len(results) == 1
    node = results[0]

    assert node.payload_data["name"] == "Alice"
    assert node.payload_data["email"] == "alice@test.com"
    assert node.payload_data["age"] == 25


def test_get_nodes_returns_empty_when_no_match(store):
    results = store.get_nodes("entity=person&name=Nonexistent")
    assert results == []


def test_get_nodes_multiple_matches(store):
    store.save('insert $p isa person, has name "Bob", has age 30;')
    store.save('insert $p isa person, has name "Bob", has age 40;')

    results = store.get_nodes("entity=person&name=Bob")

    assert len(results) == 2
    ages = sorted(n.payload_data["age"] for n in results)
    assert ages == [30, 40]


def test_remove_single_node_by_keyed_filter(store):
    store.save(
        'insert $p isa person, has name "Charlie", has email "charlie@test.com";'
    )

    assert len(store.get_nodes("entity=person&name=Charlie")) == 1

    deleted_count = store.remove_nodes("entity=person&name=Charlie")

    assert deleted_count == 1
    assert store.get_nodes("entity=person&name=Charlie") == []


def test_remove_nodes_refuses_bulk_delete_by_default(store):
    store.save('insert $p isa person, has name "Dave";')
    store.save('insert $p isa person, has name "Eve";')

    with pytest.raises(ValueError):
        store.remove_nodes("entity=person")


def test_remove_nodes_bulk_delete_with_explicit_flag(store):
    store.save('insert $p isa person, has name "Frank";')
    store.save('insert $p isa person, has name "Grace";')

    deleted_count = store.remove_nodes(
        "entity=person",
        allow_multiple=True,
    )

    assert deleted_count >= 2
    assert store.get_nodes("entity=person") == []


def test_remove_nodes_only_affects_target_entities(store):
    store.save('insert $p isa person, has name "Henry";')
    store.save('insert $p isa person, has name "Ivy";')

    store.save(
        'insert $p isa person, has name "Jack";'
    )

    store.remove_nodes("entity=person&name=Jack")

    remaining = store.get_nodes("entity=person")
    names = sorted(n.payload_data["name"] for n in remaining)

    assert names == ["Henry", "Ivy"]


def test_store_node_inserts_entity(store):
    node = Node(
        id="alice@test.com",
        entity_type="person",
        key_attribute="email",
        payload_data={
            "name": "Alice",
            "email": "alice@test.com",
            "age": 25,
        },
        relations=()
    )

    store.store_node(node)

    results = store.fetch(
        'match $p isa person, has email "alice@test.com"; fetch $p: name, email, age;'
    )

    assert len(results) == 1
    person = results[0]["p"]

    assert person["name"][0]["value"] == "Alice"
    assert person["email"][0]["value"] == "alice@test.com"
    assert person["age"][0]["value"] == 25

def test_get_nodes_none_returns_all_nodes(store):
    # Arrange
    store.save('insert $p isa person, has name "Alice";')
    store.save('insert $p isa person, has name "Bob";')
    store.save('insert $p isa person, has name "Charlie";')

    # Act
    results = store.get_nodes(None)

    # Assert
    assert isinstance(results, list)
    assert len(results) == 3

    names = sorted(n.payload_data["name"] for n in results)
    assert names == ["Alice", "Bob", "Charlie"]


def test_store_node_inserts_relation(store):
    alice = Node(
        id="alice@test.com",
        entity_type="person",
        key_attribute="email",
        payload_data={"name": "Alice", "email": "alice@test.com", "age": 25},
        relations=(),
    )

    bob = Node(
        id="bob@test.com",
        entity_type="person",
        key_attribute="email",
        payload_data={"name": "Bob", "email": "bob@test.com", "age": 30},
        relations=(
            {
                "type": "friendship",
                "roles": {
                    "friend": {
                        "entity_type": "person",
                        "key_attr": "email",
                        "key": "alice@test.com",
                    },
                    "friend_of": {
                        "entity_type": "person",
                        "key_attr": "email",
                        "key": "bob@test.com",
                    },
                },
            },
        ),
    )

    # add schema relation
    with store._query(TransactionType.WRITE) as tx:
        tx.query("""
            define
            friendship sub relation,
                relates friend,
                relates friend_of;
        """)

    store.store_node(alice)
    store.store_node(bob)

    results = store.get(
        """
        match
            $a isa person, has email "alice@test.com";
            $b isa person, has email "bob@test.com";
            (friend: $a, friend_of: $b) isa friendship;
        get;
        """
    )

    assert len(results) == 1


def test_store_node_inserts_relation(store):
    alice = Node(
        id="alice@test.com",
        entity_type="person",
        key_attribute="email",
        payload_data={"name": "Alice", "email": "alice@test.com", "age": 25},
        relations=(),
    )

    bob = Node(
        id="bob@test.com",
        entity_type="person",
        key_attribute="email",
        payload_data={"name": "Bob", "email": "bob@test.com", "age": 30},
        relations=(
            {
                "type": "friendship",
                "roles": {
                    "friend": {
                        "entity_type": "person",
                        "key_attr": "email",
                        "key": "alice@test.com",
                    },
                    "friend_of": {
                        "entity_type": "person",
                        "key_attr": "email",
                        "key": "bob@test.com",
                    },
                },
            },
        ),
    )

    # add schema relation
    with store._query(TransactionType.WRITE) as tx:
        tx.query.define("""
        define
        friendship sub relation,
            relates friend,
            relates friend_of;

        person plays friendship:friend;
        person plays friendship:friend_of;
        """)

    store.store_node(alice)
    store.store_node(bob)

    results = store.get(
        """
        match
            $a isa person, has email "alice@test.com";
            $b isa person, has email "bob@test.com";
            (friend: $a, friend_of: $b) isa friendship;
        get;
        """
    )

    assert len(results) == 1


def test_get_nodes_with_relations(store):
    with store._query(TransactionType.WRITE) as tx:
        tx.query.define("""
        define
        friendship sub relation,
            relates friend,
            relates friend_of;

        person plays friendship:friend;
        person plays friendship:friend_of;
        """)

    store.save(
        """
        insert
            $a isa person, has name "Alice", has email "alice@test.com";
            $b isa person, has name "Bob", has email "bob@test.com";
            (friend: $a, friend_of: $b) isa friendship;
        """
    )

    nodes = store.get_nodes("entity=person&email=alice@test.com&include=relations")

    assert len(nodes) == 1
    node = nodes[0]

    assert len(node.relations) == 1
    assert node.relations[0]["type"] == "friendship"


def test_key_inference_from_schema(store):
    with store._query(TransactionType.WRITE) as tx:
        tx.query.define("""
        define
        username sub attribute, value string;

        account sub entity,
            owns username @key,
            owns email;
        """)
    store.save(
        'insert $a isa account, has username "u1", has email "u1@test.com";'
    )

    results = store.get_nodes("entity=account")

    assert len(results) == 1
    node = results[0]

    assert node.key_attribute == "username"
    assert node.id == "u1"


def test_schema_evolution_new_attribute(store):
    with store._query(TransactionType.WRITE) as tx:
        tx.query.define("""
        define
        nickname sub attribute, value string;

        person owns nickname;
        """)

    store.save(
        """
        insert
            $p isa person,
            has name "Alice",
            has email "alice@test.com",
            has nickname "Al";
        """
    )

    nodes = store.get_nodes("entity=person&email=alice@test.com")

    assert len(nodes) == 1
    node = nodes[0]

    assert node.payload_data["nickname"] == "Al"


def test_relation_with_attributes(store):
    with store._query(TransactionType.WRITE) as tx:
        tx.query.define("""
        define
        friendship sub relation,
            relates friend,
            relates friend_of,
            owns since;

        since sub attribute, value long;

        person plays friendship:friend;
        person plays friendship:friend_of;
        """)
    store.save(
        """
        insert
            $a isa person, has name "Alice", has email "alice@test.com";
            $b isa person, has name "Bob", has email "bob@test.com";
            (friend: $a, friend_of: $b) isa friendship, has since 2024;
        """
    )

    nodes = store.get_nodes("entity=person&email=alice@test.com&include=relations")

    rel = nodes[0].relations[0]

    assert rel["type"] == "friendship"
    assert rel["attributes"]["since"] == 2024