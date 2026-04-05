from database_builder_libs.models.node import Node
from database_builder_libs.stores.typedb._base import TypeDbBase
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from database_builder_libs.stores.typedb.typedb_store import TypeDbDatastore


class TypeDbDeleteMixin(TypeDbBase):
    def remove_nodes(self: "TypeDbDatastore", filter: str, allow_multiple: bool = False) -> int:
        """
        Delete nodes matching the filter.

        Safety rules
        ------------
        - Refuses deleting entire entity type unless allow_multiple=True
        - Requires at least one attribute filter by default

        Returns
        -------
        int
            Number of deleted nodes

        This operation is irreversible.
        """
        if not filter:
            raise ValueError("TypeDB datastore requires a keyed filter string")

        parsed = self._parse_filter(filter)
        entity_type: str = parsed["entity_type"]
        attrs: dict[str, str] = parsed["attributes"]
        self._ensure_connected()

        if not allow_multiple and not attrs:
            raise ValueError(
                f"Refusing to delete all instances of '{entity_type}'. "
                "Provide a keyed filter or set allow_multiple=True."
            )

        match_block = self._build_match(entity_type, attrs)

        delete_query = f"""
        match
            {match_block};
        delete
            $e;
        """

        nodes = self.get_nodes(filter)
        count = len(nodes)

        if count == 0:
            return 0

        self.query_write(delete_query)

        return count

    def remove_node(self: "TypeDbDatastore", filter: str) -> Node:
        """
        Delete exactly one node and return it.

        Raises
        ------
        ValueError
            No match or multiple matches
        RuntimeError
            Deletion inconsistency detected
        """
        nodes = self.get_nodes(filter)

        if not nodes:
            raise ValueError(f"No node found for filter: {filter}")

        if len(nodes) > 1:
            raise ValueError(
                "remove_node() matched multiple nodes. "
                "Use remove_nodes(..., allow_multiple=True) instead."
            )

        removed = nodes[0]

        deleted_count = self.remove_nodes(filter, allow_multiple=False)

        if deleted_count != 1:
            raise RuntimeError("TypeDB deletion inconsistency detected")

        return removed
