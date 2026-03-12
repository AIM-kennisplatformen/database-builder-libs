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
    datastore.query_write("match $x isa person; delete $x;")

    yield datastore


def test_initialization_creates_database_and_schema(store: TypeDbDatastore):
    rows: ConceptRowIterator = store.query_read("match entity $t sub person;").as_concept_rows()
    row = next(rows, None)
    assert row is not None
    person_type = row.get("t")

    assert person_type is not None
    assert person_type.get_label() == "person"

def test_requires_connect(typedb_container):
    store = TypeDbDatastore()

    with pytest.raises(RuntimeError):
        store.get_nodes("entity=person")


def test_save_and_fetch_workflow(store: TypeDbDatastore):
    store.query_write('insert $p isa person, has name_key "Alice", has email "alice@test.com";')

    rows = store.query_read("""
        match 
        $p isa person, has name_key "Alice"; 
        fetch {
            "name_key": $p.name_key,
            "email": $p.email
        };
    """)

    results = list(rows)
    assert len(results) == 1

    assert results[0]["name_key"] == "Alice"
    assert results[0]["email"] == "alice@test.com"


def test_get_workflow(store: TypeDbDatastore):
    store.query_write('insert $p isa person, has name_key "Bob", has age 30;')
    rows = store.query_read('match $p isa person, has name_key "Bob";').as_concept_rows()

    results = list(rows)

    assert len(results) > 0


def test_update_workflow(store: TypeDbDatastore):
    store.query_write('insert $p isa person, has name_key "Charlie", has age 20;')

    update_query = """
    match
    $p isa person, has name_key "Charlie", has age 20;
    update
    $p has age 21;
    """
    store.query_write(update_query)

    rows = store.query_read('match $p isa person, has name_key "Charlie"; fetch { "age": $p.age };').as_concept_documents()
    results = list(rows)

    assert results[0]["age"] == 21


def test_delete_workflow(store: TypeDbDatastore):
    store.query_write('insert $p isa person, has name_key "Dave";')
    assert (
        len(list(
            store.query_read('match $p isa person, has name_key "Dave"; fetch { "name_key": $p.name_key };'))
    ))

    store.query_write('match $p isa person, has name_key "Dave"; delete $p;')

    rows = store.query_read('match $p isa person, has name_key "Dave"; fetch { "name_key": $p.name_key };')
    results = list(rows)

    assert len(results) == 0


def test_get_nodes_by_keyed_filter(store: TypeDbDatastore):
    # Arrange
    store.query_write(
        'insert $p isa person, has name_key "Alice", has email "alice@test.com", has age 25;'
    )

    # Act
    results = store.get_nodes("entity=person&name_key=Alice")

    # Assert
    assert len(results) == 1
    node = results[0]

    assert node.payload_data["name_key"] == "Alice"
    assert node.payload_data["email"] == "alice@test.com"
    assert node.payload_data["age"] == 25


def test_get_nodes_returns_empty_when_no_match(store: TypeDbDatastore):
    results = store.get_nodes("entity=person&name_key=Nonexistent")
    assert results == []


def test_get_nodes_multiple_matches(store: TypeDbDatastore):
    store.query_write('insert $p isa person, has name_key "Bob", has age 30;')
    store.query_write('insert $p isa person, has name_key "Bob", has age 40;')

    results = store.get_nodes("entity=person&name_key=Bob")

    assert len(results) == 2
    ages = sorted(n.payload_data["age"] for n in results)
    assert ages == [30, 40]


def test_remove_single_node_by_keyed_filter(store: TypeDbDatastore):
    store.query_write(
        'insert $p isa person, has name_key "Charlie", has email "charlie@test.com";'
    )

    assert len(store.get_nodes("entity=person&name_key=Charlie")) == 1

    deleted_count = store.remove_nodes("entity=person&name_key=Charlie")

    assert deleted_count == 1
    assert store.get_nodes("entity=person&name_key=Charlie") == []


def test_remove_nodes_refuses_bulk_delete_by_default(store: TypeDbDatastore):
    store.query_write('insert $p isa person, has name_key "Dave";')
    store.query_write('insert $p isa person, has name_key "Eve";')

    with pytest.raises(ValueError):
        store.remove_nodes("entity=person")


def test_remove_nodes_bulk_delete_with_explicit_flag(store: TypeDbDatastore):
    store.query_write('insert $p isa person, has name_key "Frank";')
    store.query_write('insert $p isa person, has name_key "Grace";')

    deleted_count = store.remove_nodes(
        "entity=person",
        allow_multiple=True,
    )

    assert deleted_count >= 2
    assert store.get_nodes("entity=person") == []


def test_remove_nodes_only_affects_target_entities(store: TypeDbDatastore):
    store.query_write('insert $p isa person, has name_key "Henry";')
    store.query_write('insert $p isa person, has name_key "Ivy";')

    store.query_write(
        'insert $p isa person, has name_key "Jack";'
    )

    store.remove_nodes("entity=person&name_key=Jack")

    remaining = store.get_nodes("entity=person")
    name_keys = sorted(n.payload_data["name_key"] for n in remaining)

    assert name_keys == ["Henry", "Ivy"]


def test_store_node_inserts_entity(store: TypeDbDatastore):
    node = Node(
        id="alice@test.com",
        entity_type="person",
        key_attribute="email",
        payload_data={
            "name_key": "Alice",
            "email": "alice@test.com",
            "age": 25,
        },
        relations=()
    )

    store.store_node(node)

    results = store.query_read(
        'match $p isa person, has email "alice@test.com"; fetch $p: name_key, email, age;'
    )

    assert len(results) == 1
    person = results[0]["p"]

    assert person["name_key"][0]["value"] == "Alice"
    assert person["email"][0]["value"] == "alice@test.com"
    assert person["age"][0]["value"] == 25

def test_get_nodes_none_returns_all_nodes(store: TypeDbDatastore):
    # Arrange
    store.query_write('insert $p isa person, has name_key "Alice";')
    store.query_write('insert $p isa person, has name_key "Bob";')
    store.query_write('insert $p isa person, has name_key "Charlie";')

    # Act
    results = store.get_nodes(None)

    # Assert
    assert isinstance(results, list)
    assert len(results) == 3

    name_keys = sorted(n.payload_data["name_key"] for n in results)
    assert name_keys == ["Alice", "Bob", "Charlie"]


def test_store_node_inserts_relation(store: TypeDbDatastore):
    alice = Node(
        id="alice@test.com",
        entity_type="person",
        key_attribute="email",
        payload_data={"name_key": "Alice", "email": "alice@test.com", "age": 25},
        relations=(),
    )

    bob = Node(
        id="bob@test.com",
        entity_type="person",
        key_attribute="email",
        payload_data={"name_key": "Bob", "email": "bob@test.com", "age": 30},
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
    store.query_write("""
            define
            friendship sub relation,
                relates friend,
                relates friend_of;
        """)

    store.store_node(alice)
    store.store_node(bob)

    results = store.query_read(
        """
        match
            $a isa person, has email "alice@test.com";
            $b isa person, has email "bob@test.com";
            (friend: $a, friend_of: $b) isa friendship;
        get;
        """
    )

    assert len(results) == 1


def test_store_node_inserts_relation(store: TypeDbDatastore):
    alice = Node(
        id="alice@test.com",
        entity_type="person",
        key_attribute="email",
        payload_data={"name_key": "Alice", "email": "alice@test.com", "age": 25},
        relations=(),
    )

    bob = Node(
        id="bob@test.com",
        entity_type="person",
        key_attribute="email",
        payload_data={"name_key": "Bob", "email": "bob@test.com", "age": 30},
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
    store.query_write("""
        define
        friendship sub relation,
            relates friend,
            relates friend_of;

        person plays friendship:friend;
        person plays friendship:friend_of;
        """)

    store.store_node(alice)
    store.store_node(bob)

    results = store.query(
        """
        match
            $a isa person, has email "alice@test.com";
            $b isa person, has email "bob@test.com";
            (friend: $a, friend_of: $b) isa friendship;
        get;
        """
    )

    assert len(results) == 1


def test_get_nodes_with_relations(store: TypeDbDatastore):
    store.query_write("""
        define
        friendship sub relation,
            relates friend,
            relates friend_of;

        person plays friendship:friend;
        person plays friendship:friend_of;
        """)

    store.query_write(
        """
        insert
            $a isa person, has name_key "Alice", has email "alice@test.com";
            $b isa person, has name_key "Bob", has email "bob@test.com";
            (friend: $a, friend_of: $b) isa friendship;
        """
    )

    nodes = store.get_nodes("entity=person&email=alice@test.com&include=relations")

    assert len(nodes) == 1
    node = nodes[0]

    assert len(node.relations) == 1
    assert node.relations[0]["type"] == "friendship"


def test_key_inference_from_schema(store: TypeDbDatastore):
    store.query_write("""
        define
        username_key sub attribute, value string;

        account sub entity,
            owns username_key @key,
            owns email;
        """)
    store.query_write(
        'insert $a isa account, has username_key "u1", has email "u1@test.com";'
    )

    results = store.get_nodes("entity=account")

    assert len(results) == 1
    node = results[0]

    assert node.key_attribute == "username_key"
    assert node.id == "u1"


def test_schema_evolution_new_attribute(store: TypeDbDatastore):
    store.query_write("""
        define
        nickname_key sub attribute, value string;

        person owns nickname_key;
        """)

    store.query_write(
        """
        insert
            $p isa person,
            has name_key "Alice",
            has email "alice@test.com",
            has nickname_key "Al";
        """
    )

    nodes = store.get_nodes("entity=person&email=alice@test.com")

    assert len(nodes) == 1
    node = nodes[0]

    assert node.payload_data["nickname_key"] == "Al"


def test_relation_with_attributes(store: TypeDbDatastore):
    store.query_write("""
        define
        friendship sub relation,
            relates friend,
            relates friend_of,
            owns since;

        since sub attribute, value long;

        person plays friendship:friend;
        person plays friendship:friend_of;
        """)
    
    store.query_write(
        """
        insert
            $a isa person, has name_key "Alice", has email "alice@test.com";
            $b isa person, has name_key "Bob", has email "bob@test.com";
            (friend: $a, friend_of: $b) isa friendship, has since 2024;
        """
    )

    nodes = store.get_nodes("entity=person&email=alice@test.com&include=relations")

    rel = nodes[0].relations[0]

    assert rel["type"] == "friendship"
    assert rel["attributes"]["since"] == 2024