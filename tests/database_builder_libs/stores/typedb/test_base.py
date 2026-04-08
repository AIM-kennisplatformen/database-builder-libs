import pytest
from typedb.driver import ConceptRowIterator
from database_builder_libs.stores.typedb.typedb_store import TypeDbDatastore

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
