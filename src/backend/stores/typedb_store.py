from typing import Final, Mapping, Optional, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typedb.driver import (
    SessionType,
    TransactionType,
    TypeDB,
    TypeDBDriver,
    TypeDBOptions,
    TypeDBSession,
    TypeDBTransaction,
)

from backend.config import settings
from backend.models.abstract_store import AbstractStore
from backend.models.node import EntityType, KeyAttribute, Node, NodeId
from urllib.parse import parse_qs
from typing import TypedDict, cast

class RelationRef(TypedDict):
    entity_type: str
    key_attr: str
    key: str

class RelationData(TypedDict, total=False):
    type: str
    roles: Mapping[str, RelationRef]
    attributes: Mapping[str, object]


class TypeDbDatastore(AbstractStore):
    """
    TypeDbDatastore is a concrete implementation of the Datastore abstract base class for
    interacting with a TypeDB database.

    Data in TypeDB can be inserted, queried, and managed using TypeDB's schema and query language.

    Due to the implementation of the typedb-driver for TypeDB 2.x this datastore needs separate
    methods for inserting, fetching, deleting, and updating data.
    """

    def __init__(self) -> None:
        self.typedb_driver: Final[TypeDBDriver] = TypeDB.core_driver(
            address=settings.TYPEDB_URI
        )
        self.database: Final[str] = settings.TYPEDB_DATABASE

        assert self.typedb_driver is not None
        assert self.database is not None

        if not self.typedb_driver.databases.contains(self.database):
            self.typedb_driver.databases.create(self.database)

        current_dir = Path(__file__).parent
        schema_path = current_dir / "schema.tql"

        # ---- DEFINE SCHEMA ----
        with self._query(SessionType.SCHEMA, TransactionType.WRITE) as transaction:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = f.read()
                transaction.query.define(schema)

        # ---- CRITICAL FIX ----
        # TypeDB requires reopening driver after schema changes
        self.typedb_driver.close()
        self.typedb_driver = TypeDB.core_driver(address=settings.TYPEDB_URI)


    @contextmanager
    def _query(self, session_type: SessionType, transaction_type: TransactionType):
        session: TypeDBSession
        transaction: TypeDBTransaction
        with self.typedb_driver.session(self.database, session_type) as session:
            with session.transaction(transaction_type) as transaction:
                try:
                    yield transaction
                finally:
                    if transaction.is_open() and transaction_type.is_write():
                        transaction.commit()

    def save(self, query: str, options: Optional[TypeDBOptions] = None) -> None:
        with self._query(SessionType.DATA, TransactionType.WRITE) as transaction:
            transaction.query.insert(query, options)

    def delete(self, query: str, options: Optional[TypeDBOptions] = None) -> None:
        with self._query(SessionType.DATA, TransactionType.WRITE) as transaction:
            transaction.query.delete(query, options)

    def fetch(self, query: str, options: Optional[TypeDBOptions] = None) -> list[Any]:
        with self._query(SessionType.DATA, TransactionType.READ) as transaction:
            iterator = transaction.query.fetch(query, options)
            results = list(iterator)
            return results

    def get(self, query: str, options: Optional[TypeDBOptions] = None) -> list[Any]:
        with self._query(SessionType.DATA, TransactionType.READ) as transaction:
            iterator = transaction.query.get(query, options)
            results = [result.map for result in iterator]
            return results

    def update(self, query: str, options: Optional[TypeDBOptions] = None) -> None:
        with self._query(SessionType.DATA, TransactionType.WRITE) as transaction:
            transaction.query.update(query, options)

    def _format_attributes(self, payload: Mapping[str, object]) -> str:
        clauses = []

        for attr, value in payload.items():
            if value is None:
                continue

            if isinstance(value, str):
                clauses.append(f'has {attr} "{value}"')
            elif isinstance(value, bool):
                clauses.append(f"has {attr} {str(value).lower()}")
            elif isinstance(value, int):
                clauses.append(f"has {attr} {value}")
            elif isinstance(value, float):
                clauses.append(f"has {attr} {value}")
            else:
                raise TypeError(f"Unsupported value for attribute {attr}: {type(value)}")

        return ", ".join(clauses)

    def _entity_exists(self, entity_type: str, key_attr: str, key_value: str) -> bool:
        query = f"""
        match
            $e isa {entity_type},
            has {key_attr} "{key_value}";
        fetch $e : name;
        """
        return bool(self.fetch(query))
    def _insert_entity(
        self,
        entity_type: str,
        key_attr: str,
        key_value: str,
        payload: Mapping[str, object],
    ) -> None:
        attrs = self._format_attributes(payload)

        query = f"""
        insert
            $e isa {entity_type},
            has {key_attr} "{key_value}"
            {", " if attrs else ""}{attrs};
        """

        self.save(query)

    def connect_to_source(self) -> None:
        """Abstract method to connect to the sink."""
        self.__init__()
    

    def _relation_exists(self, rel_type: str, role_map: Mapping[str, RelationRef]) -> bool:
        match_roles = []

        for role, ref in role_map.items():
            match_roles.append(
                f"""
                ${role} isa {ref["entity_type"]},
                    has {ref["key_attr"]} "{ref["key"]}";
                """
            )

        query = f"""
        match
            {''.join(match_roles)}
            ({', '.join(f'{r}: ${r}' for r in role_map)}) isa {rel_type};
        fetch;
        """

        return bool(self.fetch(query))


    def _insert_relation(self, rel: RelationData) -> None:
        if self._relation_exists(rel["type"], rel["roles"]):
            return

        match_roles = []
        insert_roles = []

        for role, ref in rel["roles"].items():
            match_roles.append(
                f"""
                ${role} isa {ref["entity_type"]},
                    has {ref["key_attr"]} "{ref["key"]}";
                """
            )
            insert_roles.append(f"{role}: ${role}")

        attrs = self._format_attributes(rel.get("attributes", {}))

        query = f"""
        match
            {''.join(match_roles)}
        insert
            ({', '.join(insert_roles)}) isa {rel["type"]}
            {", " if attrs else ""}{attrs};
        """

        self.save(query)

    def store_node(self, node: Node) -> None:
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
    
    def _parse_filter(self, filter: str) -> dict:
        parsed = parse_qs(filter, keep_blank_values=False)

        # flatten single values
        params = {k: v[0] for k, v in parsed.items()}

        if "entity" not in params:
            raise ValueError("Missing required 'entity' filter")

        entity_type = params.pop("entity")
        include_relations = params.pop("include", None) == "relations"

        return {
            "entity_type": entity_type,
            "attributes": params,
            "include_relations": include_relations,
        }

    def _format_attribute_match(self, attrs: dict[str, str]) -> str:
        clauses = []

        for attr, value in attrs.items():
            clauses.append(f'has {attr} "{value}"')

        return ",\n       ".join(clauses)
    def _unwrap_value(self, v: Any) -> Any:
        # Most common: list of value dicts
        if isinstance(v, list) and v:
            first = v[0]
            if isinstance(first, dict) and "value" in first:
                return first["value"]
            return first

        # Sometimes already a dict
        if isinstance(v, dict) and "value" in v:
            return v["value"]

        # Fallback: already a primitive
        return v
    def _get_entity_attribute_labels(self, entity_type: str) -> list[str]:
        query = f"""
        match
            $t type {entity_type};
            $t owns $a;
        get
            $a;
        """
        results = self.get(query)
        return sorted({r["a"].get_label().name for r in results})
    
    def get_nodes(self, filter: str | Sequence[float] | None) -> list[Node]:
        if not isinstance(filter, str):
            raise TypeError("TypeDB datastore requires string filter")

        if not filter:
            raise ValueError("TypeDB datastore requires a keyed filter string")

        parsed = self._parse_filter(filter)
        entity_type = parsed["entity_type"]
        attrs = parsed["attributes"]

        match_clauses = [f"$e isa {entity_type}"]
        if attrs:
            match_clauses.append(self._format_attribute_match(attrs))

        match_block = ", ".join(match_clauses)

        attr_labels = self._get_entity_attribute_labels(entity_type)
        if not attr_labels:
            return []

        fetch_block = ", ".join(attr_labels)

        query = f"""
        match
            {match_block};
        fetch
            $e: {fetch_block};
        """

        nodes: list[Node] = []

        def is_subset(a: Mapping[str, object], b: Mapping[str, object]) -> bool:
            return all(k in b and b[k] == v for k, v in a.items())

        with self._query(SessionType.DATA, TransactionType.READ) as tx:
            for res in tx.query.fetch(query):
                entity_data = res["e"]

                payload: dict[str, object] = {}
                for attr_name, values in entity_data.items():
                    if attr_name == "type" or not values:
                        continue
                    payload[attr_name] = values[0]["value"]

                # derive key attribute (first attribute)
                if not payload:
                    continue

                key_attr = next(iter(payload))
                key_val = payload[key_attr]

                new_node = Node(
                    id=NodeId(str(key_val)),
                    entity_type=EntityType(entity_type),
                    key_attribute=KeyAttribute(key_attr),
                    payload_data=payload,
                    relations=(),
                    vector_data=(),
                    embedding_model="typedb",
                )

                merged = False
                for i, existing in enumerate(nodes):
                    if is_subset(payload, existing.payload_data):
                        merged = True
                        break

                    if is_subset(existing.payload_data, payload):
                        nodes[i] = new_node
                        merged = True
                        break

                if not merged:
                    nodes.append(new_node)

        return nodes
    
    def _ensure_safe_delete(
        self,
        entity_type: str,
        attrs: dict[str, str],
        allow_multiple: bool,
    ) -> None:
        if allow_multiple:
            return

        # Require at least one attribute filter (ideally @key)
        if not attrs:
            raise ValueError(
                f"Refusing to delete all instances of '{entity_type}'. "
                "Provide a keyed filter or set allow_multiple=True."
            )


    def remove_nodes(self, filter: str, allow_multiple: bool = False) -> int:
        if not filter:
            raise ValueError("TypeDB datastore requires a keyed filter string")

        parsed = self._parse_filter(filter)
        entity_type: str = parsed["entity_type"]
        attrs: dict[str, str] = parsed["attributes"]

        if not allow_multiple and not attrs:
            raise ValueError(
                f"Refusing to delete all instances of '{entity_type}'. "
                "Provide a keyed filter or set allow_multiple=True."
            )

        match_clauses = [f"$e isa {entity_type}"]
        if attrs:
            match_clauses.append(self._format_attribute_match(attrs))
        match_block = ",\n       ".join(match_clauses)

        delete_query = f"""
        match
            {match_block};
        delete
            $e isa {entity_type};
        """

        nodes = self.get_nodes(filter)
        count = len(nodes)

        if count == 0:
            return 0

        with self._query(SessionType.DATA, TransactionType.WRITE) as tx:
            tx.query.delete(delete_query)

        return count
    
    def remove_node(self, filter: str) -> Node:
        """
        Remove exactly one node and return it.
        Adapter around remove_nodes() to satisfy AbstractStore.
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

        # perform actual deletion using your existing logic
        deleted_count = self.remove_nodes(filter, allow_multiple=False)

        if deleted_count != 1:
            raise RuntimeError("TypeDB deletion inconsistency detected")

        return removed
