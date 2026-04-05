from database_builder_libs.stores.typedb._types import (
    RelationRef,
    RelationData,
    EagerQueryAnswer,
)
from database_builder_libs.stores.typedb._base import TypeDbBase
from database_builder_libs.stores.typedb._schema import TypeDbSchemaMixin
from database_builder_libs.stores.typedb._query import TypeDbQueryMixin
from database_builder_libs.stores.typedb._read import TypeDbReadMixin
from database_builder_libs.stores.typedb._write import TypeDbWriteMixin
from database_builder_libs.stores.typedb._delete import TypeDbDeleteMixin

class TypeDbDatastore(
    TypeDbDeleteMixin,
    TypeDbWriteMixin,
    TypeDbReadMixin,
    TypeDbQueryMixin,
    TypeDbSchemaMixin,
    TypeDbBase,
):
    """
    TypeDB-backed implementation of the AbstractStore knowledge persistence layer.

    This adapter maps canonical `Node` objects onto a TypeDB schema.

    Mapping rules
    -------------
    Node → TypeDB Entity
        node.entity_type      → entity type
        node.id               → key_attribute value
        node.payload_data     → attributes
        node.relations        → relations

    Identity semantics
    ------------------
    A Node is uniquely identified by id.

    Storing the same node twice MUST NOT create duplicates.
    The adapter performs existence checks before insertion.

    Filter language
    ---------------
    All read and delete operations use a URL-query style filter string:

        "entity=<type>&<attribute>=<value>&<attribute>=<value>"

    Examples
        entity=person&email=a@b.com
        entity=document&title=Report.pdf

    Behaviour
    ---------
    - get_nodes() returns normalized Node objects reconstructed from TypeDB
    - store_node() performs upsert semantics
    - remove_node() deletes exactly one node (safety enforced)
    - remove_nodes() allows bulk deletion when explicitly enabled

    Safety guarantees
    -----------------
    - Refuses full-entity deletion without explicit permission
    - Deduplicates overlapping attribute matches
    - Ensures relations are not duplicated
    - Schema is automatically applied at initialization

    Transactions
    ------------
    Each operation runs in its own transaction.
    Writes are committed automatically when successful.

    Notes
    -----
    This adapter assumes the schema defines a key attribute for each entity type.
    """
    pass

__all__ = [
    "TypeDbDatastore",
    "RelationRef",
    "RelationData",
    "EagerQueryAnswer",
]
