from typing import Mapping, cast

from database_builder_libs.models.node import Node
from database_builder_libs.stores.typedb._base import TypeDbBase
from database_builder_libs.stores.typedb._types import RelationData
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from database_builder_libs.stores.typedb.typedb_store import TypeDbDatastore


class TypeDbWriteMixin(TypeDbBase):
    def _insert_entity(
        self: "TypeDbDatastore",
        entity_type: str,
        key_attr: str,
        key_value: str,
        payload: Mapping[str, object],
    ) -> None:
        """Insert a node entity into the TypeDB database."""
        attrs = self._format_attributes(payload)

        query = f"""
        insert
            $e isa {entity_type},
            has {key_attr} "{key_value}"
            {", " if attrs else ""}{attrs};
        """

        self.query_write(query)

    def _insert_relation(self: "TypeDbDatastore", rel: RelationData) -> None:
        """Insert a relation and link it to its role players."""
        attributes = rel.get("attributes", {})

        if self._relation_exists(rel):
            return

        match_roles = [self._match_relation_ref(role, ref) for role, ref in rel["roles"].items()]
        insert_roles = [f"{role}: ${role}" for role in rel["roles"]]

        attrs = self._format_attributes(attributes)

        query = f"""
        match
            {"".join(match_roles)}
        insert
            ({", ".join(insert_roles)}) isa {rel["type"]}
            {", " + attrs if attrs else ""};
        """

        self.query_write(query)

    def store_node(self: "TypeDbDatastore", node: Node) -> None:
        """
        Insert a Node into TypeDB if it does not already exist, and ensure its relations exist.

        Behaviour
        ---------
        - Creates the entity if it does not exist
        - Leaves existing entity attributes unchanged (no merge or update of payload_data)
        - Inserts missing relations only
        - Operation is idempotent

        Identity rule
        -------------
        A node is considered existing if an entity with
        (entity_type, key_attribute, id) exists.

        Relations are inserted only if not already present.
        """
        self._ensure_connected()
        if not self._entity_exists(
            node.entity_type,
            node.key_attribute,
            node.id,
        ):
            self._insert_entity(
                node.entity_type,
                node.key_attribute,
                node.id,
                node.payload_data,
            )

        for rel in node.relations:
            self._insert_relation(cast(RelationData, rel))
