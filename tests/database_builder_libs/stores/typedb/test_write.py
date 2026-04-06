import pytest
from database_builder_libs.models.node import Node
from database_builder_libs.stores.typedb.typedb_store import TypeDbDatastore

def test_save_and_fetch_workflow(store: TypeDbDatastore):
    store.query_write('insert $p isa person, has name "Alice", has email "alice@test.com";')

    rows = store.query_read("""
        match 
        $p isa person, has name "Alice"; 
        fetch {
            "name": $p.name,
            "email": $p.email
        };
    """).as_concept_documents()

    results = list(rows)
    assert len(results) == 1

    assert results[0]["name"] == "Alice"
    assert results[0]["email"] == "alice@test.com"


def test_update_workflow(store: TypeDbDatastore):
    store.query_write('insert $p isa person, has name "Charlie", has age 20;')

    update_query = """
    match
    $p isa person, has name "Charlie", has age 20;
    update
    $p has age 21;
    """
    store.query_write(update_query)

    rows = store.query_read('match $p isa person, has name "Charlie"; fetch { "age": $p.age };').as_concept_documents()
    results = list(rows)

    assert results[0]["age"] == 21


def test_store_node_inserts_entity(store: TypeDbDatastore, alice_node: Node):
    store.store_node(alice_node)

    results = store.query_read("""
        match $p isa person, has email "alice@test.com"; 
        fetch { $p.* };
    """).as_concept_documents()

    rows = list(results)

    assert len(rows) == 1
    person = rows[0]

    assert person["name"] == "Alice"
    assert person["email"] == "alice@test.com"
    assert person["age"] == 25


def test_store_node_inserts_relation(friendship_store: TypeDbDatastore, alice_node: Node, bob_with_friendship_node: Node):
    friendship_store.store_node(alice_node)
    friendship_store.store_node(bob_with_friendship_node)

    results = friendship_store.query_read(
        """
        match
            $a isa person, has email "alice@test.com";
            $b isa person, has email "bob@test.com";
            $f isa friendship, links (friend: $a, friend_of: $b);
        """
    ).as_concept_rows()

    assert len(list(results)) == 1


def test_store_node_relation_idempotent(friendship_store: TypeDbDatastore, alice_node: Node, bob_with_friendship_node: Node):
    """
    Store the same node again to verify that the operation is idempotent.
    Re-inserting an existing node should not create duplicates or duplicate relations. 
    This ensures that calling `store_node` multiple times with the same data leaves the graph in the same state.
    """
    friendship_store.store_node(alice_node)
    friendship_store.store_node(bob_with_friendship_node)

    # insert again
    friendship_store.store_node(bob_with_friendship_node)

    nodes = friendship_store.get_nodes("entity=person&name=Bob&include=relations")

    assert len(nodes[0].relations) == 1
