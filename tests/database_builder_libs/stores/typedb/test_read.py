import pytest
from database_builder_libs.stores.typedb.typedb_store import TypeDbDatastore

def test_get_workflow(store: TypeDbDatastore):
    store.query_write('insert $p isa person, has name "Bob", has age 30;')
    rows = store.query_read('match $p isa person, has name "Bob";').as_concept_rows()

    results = list(rows)

    assert len(results) > 0


def test_get_nodes_by_keyed_filter(store: TypeDbDatastore):
    # Arrange
    store.query_write(
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


def test_get_nodes_returns_empty_when_no_match(store: TypeDbDatastore):
    results = store.get_nodes("entity=person&name=Nonexistent")
    assert results == []


def test_get_nodes_multiple_matches(store: TypeDbDatastore):
    store.query_write('insert $p isa person, has name "Bob1", has age 30;')
    store.query_write('insert $p isa person, has name "Bob2", has age 30;')

    results = store.get_nodes("entity=person&age=30")

    assert len(results) == 2
    ages = sorted(n.payload_data["age"] for n in results)
    assert ages == [30, 30]


def test_get_nodes_none_returns_all_nodes(store: TypeDbDatastore):
    # Arrange
    store.query_write('insert $p isa person, has name "Alice";')
    store.query_write('insert $p isa person, has name "Bob";')
    store.query_write('insert $p isa person, has name "Charlie";')

    # Act
    results = store.get_nodes(None)

    # Assert
    assert isinstance(results, list)
    assert len(results) == 3

    names = sorted(n.payload_data["name"] for n in results)
    assert names == ["Alice", "Bob", "Charlie"]


def test_get_nodes_with_relations(populated_friendship_store: TypeDbDatastore):
    nodes = populated_friendship_store.get_nodes("entity=person&email=alice@test.com&include=relations")

    assert len(nodes) == 1
    node = nodes[0]

    assert len(node.relations) == 1
    assert node.relations[0]["type"] == "friendship"


def test_relation_single_role_player_is_loaded(store: TypeDbDatastore):
    store.query_schema("""
    define
    relation tagged,
        relates item;

    person plays tagged:item;
    """)

    store.query_write("""
        insert
            $p isa person, has name "Alice", has email "alice@test.com";
            (item: $p) isa tagged;
    """)

    nodes = store.get_nodes("entity=person&email=alice@test.com&include=relations")

    assert len(nodes) == 1
    node = nodes[0]

    assert len(node.relations) == 1
    assert node.relations[0]["type"] == "tagged"


def test_relation_attributes_are_loaded(store: TypeDbDatastore):
    store.query_schema("""
        define
        relation collaboration,
            relates contributor,
            relates project,
            owns contribution_role;

        attribute contribution_role, value string;

        person plays collaboration:contributor;
        person plays collaboration:project;
        """)

    store.query_write("""
        insert
            $a isa person, has name "Alice", has email "alice@test.com";
            $b isa person, has name "Bob", has email "bob@test.com";
            (contributor: $a, project: $b) isa collaboration, has contribution_role "author";
    """)

    nodes = store.get_nodes("entity=person&email=alice@test.com&include=relations")

    rel = nodes[0].relations[0]

    assert rel["attributes"]["contribution_role"] == "author"


def test_relations_not_duplicated(populated_friendship_store: TypeDbDatastore):
    nodes = populated_friendship_store.get_nodes("entity=person&email=alice@test.com&include=relations")

    assert len(nodes[0].relations) == 1


def test_multiple_relations_retrieval(store: TypeDbDatastore):
    store.query_schema("""
        define
        relation friendship,
            relates friend,
            relates friend_of;

        relation tagged,
            relates item;

        person plays friendship:friend;
        person plays friendship:friend_of;
        person plays tagged:item;
        """)

    store.query_write("""
        insert
            $a isa person, has name "Alice", has email "alice@test.com";
            $b isa person, has name "Bob", has email "bob@test.com";
            $f isa friendship, links (friend: $a, friend_of: $b);
            (item: $a) isa tagged;
    """)


    nodes = store.get_nodes("entity=person&email=alice@test.com&include=relations")

    assert len(nodes) == 1
    node = nodes[0]

    assert len(node.relations) == 2
    relation_types = sorted(r["type"] for r in node.relations)
    assert relation_types == ["friendship", "tagged"]


def test_get_relations_returns_all_when_none(populated_friendship_store: TypeDbDatastore):
    relations = populated_friendship_store.get_relations(None)
    assert len(relations) >= 1
    types = [r["type"] for r in relations]
    assert "friendship" in types


def test_get_relations_by_type(populated_friendship_store: TypeDbDatastore):
    relations = populated_friendship_store.get_relations("relation=friendship")
    assert len(relations) == 1
    rel = relations[0]
    assert rel["type"] == "friendship"
    assert "friend" in rel["roles"]
    assert "friend_of" in rel["roles"]
    assert rel["roles"]["friend"]["key"] == "Alice"
    assert rel["roles"]["friend_of"]["key"] == "Bob"


def test_get_relations_with_attributes(store: TypeDbDatastore):
    store.query_schema("""
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

    relations = store.get_relations("relation=friendship&since=2024")
    assert len(relations) == 1
    rel = relations[0]
    assert rel["type"] == "friendship"
    assert rel["attributes"]["since"] == 2024


def test_get_relations_returns_empty_when_no_match(store: TypeDbDatastore):
    store.query_schema("""
        define
        relation friendship,
            relates friend,
            owns since;

        attribute since, value integer;

        person plays friendship:friend;
        """)
        
    store.query_write(
        """
        insert
            $a isa person, has name "Alice", has email "alice@test.com";
            $f isa friendship, links (friend: $a), has since 2024;
        """
    )
    
    relations = store.get_relations("relation=friendship&since=2099")
    assert relations == []
