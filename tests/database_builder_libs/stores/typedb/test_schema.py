import pytest
from database_builder_libs.stores.typedb.typedb_store import TypeDbDatastore

def test_key_inference_from_schema(store: TypeDbDatastore):
    store.query_schema("""
        define
        attribute username, value string;

        entity account,
            owns username @key,
            owns email;
        """)
    store.query_write(
        'insert $a isa account, has username "u1", has email "u1@test.com";'
    )

    results = store.get_nodes("entity=account")

    assert len(results) == 1
    node = results[0]

    assert node.key_attribute == "username"
    assert node.id == "u1"


def test_schema_evolution_new_attribute(store: TypeDbDatastore):
    store.query_schema("""
        define
        attribute nickname, value string;

        person owns nickname;
        """)

    store.query_write(
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


def test_relation_with_attributes(store: TypeDbDatastore):
    tx = store.query_schema("""
        define
        relation friendship,
            relates friend,
            relates friend_of,
            owns since;

        attribute since, value integer;

        person plays friendship:friend;
        person plays friendship:friend_of;
        """)
    
    store.query_write(
        """
        insert
            $a isa person, has name "Alice", has email "alice@test.com";
            $b isa person, has name "Bob", has email "bob@test.com";
            $f isa friendship, links (friend: $a, friend_of: $b), has since 2024;
        """
    )

    nodes = store.get_nodes("entity=person&email=alice@test.com&include=relations")

    rel = nodes[0].relations[0]

    assert rel["type"] == "friendship"
    assert rel["attributes"]["since"] == 2024
