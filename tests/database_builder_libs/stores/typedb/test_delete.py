import pytest
from database_builder_libs.stores.typedb.typedb_store import TypeDbDatastore

def test_delete_workflow(store: TypeDbDatastore):
    store.query_write('insert $p isa person, has name "Dave";')
    assert (
        len(list(
            store.query_read('match $p isa person, has name "Dave"; fetch { "name": $p.name };'))
    ))

    store.query_write('match $p isa person, has name "Dave"; delete $p;')

    rows = store.query_read('match $p isa person, has name "Dave"; fetch { "name": $p.name };')
    results = list(rows)

    assert len(results) == 0


def test_remove_single_node_by_keyed_filter(store: TypeDbDatastore):
    store.query_write(
        'insert $p isa person, has name "Charlie", has email "charlie@test.com";'
    )

    assert len(store.get_nodes("entity=person&name=Charlie")) == 1

    deleted_count = store.remove_nodes("entity=person&name=Charlie")

    assert deleted_count == 1
    assert store.get_nodes("entity=person&name=Charlie") == []


def test_remove_nodes_refuses_bulk_delete_by_default(store: TypeDbDatastore):
    store.query_write('insert $p isa person, has name "Dave";')
    store.query_write('insert $p isa person, has name "Eve";')

    with pytest.raises(ValueError):
        store.remove_nodes("entity=person")


def test_remove_nodes_bulk_delete_with_explicit_flag(store: TypeDbDatastore):
    store.query_write('insert $p isa person, has name "Frank";')
    store.query_write('insert $p isa person, has name "Grace";')

    deleted_count = store.remove_nodes(
        "entity=person",
        allow_multiple=True,
    )

    assert deleted_count >= 2
    assert store.get_nodes("entity=person") == []


def test_remove_nodes_only_affects_target_entities(store: TypeDbDatastore):
    store.query_write('insert $p isa person, has name "Henry";')
    store.query_write('insert $p isa person, has name "Ivy";')

    store.query_write(
        'insert $p isa person, has name "Jack";'
    )

    store.remove_nodes("entity=person&name=Jack")

    remaining = store.get_nodes("entity=person")
    names = sorted(n.payload_data["name"] for n in remaining)

    assert names == ["Henry", "Ivy"]
