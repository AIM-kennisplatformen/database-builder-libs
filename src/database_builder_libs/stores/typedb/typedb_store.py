from typing import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typedb.driver import (
    ConceptRow,
    Credentials,
    Driver,
    DriverOptions,
    QueryAnswer,
    Transaction,
    TransactionType,
    TypeDB,
)

from database_builder_libs.models.abstract_store import AbstractStore
from database_builder_libs.models.node import EntityType, KeyAttribute, Node, NodeId
from urllib.parse import parse_qs
from typing import TypedDict, cast
from dataclasses import replace


class RelationRef(TypedDict):
    entity_type: str
    key_attr: str
    key: str


class RelationData(TypedDict, total=False):
    type: str
    roles: Mapping[str, RelationRef]
    attributes: Mapping[str, object]


class EagerQueryAnswer:
    """
    Eagerly evaluates a TypeDB QueryAnswer to prevent 'concurrent transaction close' errors
    when evaluating iterator wrappers outside of the transaction block.
    """
    def __init__(self, answer: QueryAnswer):
        self._is_docs = answer.is_concept_documents()
        self._is_rows = answer.is_concept_rows()
        if self._is_docs:
            self._docs = list(answer.as_concept_documents())
        elif self._is_rows:
            self._rows = list(answer.as_concept_rows())
        else:
            self._answer = answer

    def as_concept_documents(self):
        if not self._is_docs:
            raise TypeError("Query did not return concept documents")
        return iter(self._docs)

    def as_concept_rows(self):
        if not self._is_rows:
            raise TypeError("Query did not return concept rows")
        return iter(self._rows)

    def as_raw(self):
        if not self._answer:
            raise TypeError("Query is already evaluated as documents or rows")
        return self._answer

    def __iter__(self):
        if self._is_docs:
            return iter(self._docs)
        if self._is_rows:
            return iter(self._rows)
        return iter([])


class TypeDbDatastore(AbstractStore):
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

    def __init__(self) -> None:
        super().__init__()
        self.typedb_driver: Driver | None = None
        self.database: str | None = None
        self._entity_attr_cache: dict[str, list[str]] = {}
        self._all_attr_cache: list[str] | None = None
        self._key_attr_cache: dict[str, str | None] = {}

    def _connect_impl(self, config: dict | None) -> None:
        if not config:
            raise ValueError("TypeDB requires configuration")

        uri = config.get("uri")
        database = config.get("database")
        schema_path = config.get("schema_path")

        if not uri or not database:
            raise ValueError("TypeDB config requires 'uri' and 'database'")

        credentials = Credentials(config.get("username"), config.get("password"))
        driver_options = DriverOptions(is_tls_enabled=config.get("tls"))

        self.typedb_driver = TypeDB.driver(address=uri, credentials=credentials, driver_options=driver_options)
        self.database = database

        # create database if missing
        if not self.typedb_driver.databases.contains(database):
            self.typedb_driver.databases.create(database)

        # apply schema if provided
        if schema_path:
            path = Path(schema_path)
            with self.transaction(TransactionType.SCHEMA) as tx:
                tx.query(path.read_text(encoding="utf-8")).resolve()
                tx.commit()

            # required by TypeDB after schema change
            self.typedb_driver.close()
            self.typedb_driver = TypeDB.driver(address=uri, credentials=credentials, driver_options=driver_options)

    def _build_match(
        self, entity_type: str | None, attrs: Mapping[str, str] | None
    ) -> str:
        clauses = []

        if entity_type:
            clauses.append(f"$e isa {entity_type}")
        else:
            clauses.append("$e isa entity")

        if attrs:
            clauses.append(self._format_attribute_match(dict(attrs)))

        return ", ".join(clauses)

    @contextmanager
    def transaction(self, transaction_type: TransactionType) -> Generator[Transaction, None, None]:
        self._ensure_connected()

        with self.typedb_driver.transaction(database_name=self.database, transaction_type=transaction_type) as transaction:
            try:
                yield transaction
            except Exception:
                # Do NOT commit — let TypeDB abort on close
                raise
            else:
                if (transaction_type.is_write() or transaction_type.is_schema()) and transaction.is_open():
                    transaction.commit()

    def query_read(self, query: str) -> EagerQueryAnswer:
        with self.transaction(TransactionType.READ) as tx:
            return EagerQueryAnswer(tx.query(query).resolve())

    def query_write(self, query: str) -> EagerQueryAnswer:
        with self.transaction(TransactionType.WRITE) as tx:
            return EagerQueryAnswer(tx.query(query).resolve())

    def query_schema(self, query: str) -> EagerQueryAnswer:
        with self.transaction(TransactionType.SCHEMA) as tx:
            return EagerQueryAnswer(tx.query(query).resolve())

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
                raise TypeError(
                    f"Unsupported value for attribute {attr}: {type(value)}"
                )

        return ", ".join(clauses)

    def _entity_exists(self, entity_type: str, key_attr: str, key_value: str) -> bool:

        query = f"""
        match
            $e isa {entity_type},
                has {key_attr} "{key_value}";
        limit 1;
        """

        result = self.query_read(query).as_concept_rows()

        return bool(len(list(result)) == 1)

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

        self.query_write(query)

    def _match_relation_ref(self, role: str, ref: RelationRef) -> str:
        return f"""
        ${role} isa {ref["entity_type"]},
            has {ref["key_attr"]} "{ref["key"]}";
        """

    def _relation_exists(
        self, rel: RelationData
    ) -> bool:
        role_map = rel["roles"]
        attributes = rel.get("attributes", {})

        match_roles = [
            self._match_relation_ref(role, ref)
            for role, ref in role_map.items()
        ]

        attr_match = ""
        if attributes is not {}:
            attr_match = ", " + self._format_attributes(attributes)

        # Generate the relation line
        role_bindings = ", ".join([f"{role}: ${role}" for role in role_map.keys()])

        query = f"""
        match
            {" ".join(match_roles)}
            $r ({", ".join(f"{r}: ${r}" for r in role_map)}) isa {rel["type"]}
            {attr_match};
        limit 1;
        """

        rows = self.query_read(query).as_concept_rows()

        return bool(len(list(rows)) > 0)

    def _insert_relation(self, rel: RelationData) -> None:
        attributes = rel.get("attributes", {})

        if self._relation_exists(rel):
            return

        match_roles = [
            self._match_relation_ref(role, ref)
            for role, ref in rel["roles"].items()
        ]
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

    def store_node(self, node: Node) -> None:
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

    def _get_key_attribute(
        self, entity_type: str, payload: Mapping[str, object]
    ) -> str:
        # cached?
        if entity_type in self._key_attr_cache:
            key = self._key_attr_cache[entity_type]
            return key if key else sorted(payload.keys())[0]

        query = f"""
        match
            {entity_type} owns $a;
            $a sub key;
        fetch {{
            'a': $a
        }};
        """
        rows = self.query_read(query).as_concept_documents()
        result = list(rows)

        if result:
            key = result[0]["a"].get("label")
            self._key_attr_cache[entity_type] = key
            return key

        # remember: no schema key
        self._key_attr_cache[entity_type] = None
        return sorted(payload.keys())[0]

    def _get_key_attr_for_type(self, entity_type: str) -> str | None:

        if entity_type in self._key_attr_cache:
            return self._key_attr_cache[entity_type]

        query = f"""
        match
            {entity_type} owns $a;
            $a sub key;
        fetch {{
            'a': $a
        }};
        """

        result = self.query_read(query)

        if result:
            key = result[0]["a"].get_label().name
            self._key_attr_cache[entity_type] = key
            return key

        self._key_attr_cache[entity_type] = None
        return None

    def _get_entity_key_value(self, entity_type: str, key_attr: str, thing) -> str:
        # best effort: read all attributes and pick the key_attr
        for a in thing.get_has():
            if a.get_type().get_label().name == key_attr:
                return str(a.get_value())
        # fallback: re-query by iid if needed, but usually not necessary
        return ""

    def _get_relations_for_entity(
        self,
        tx,
        entity_type: str,
        key_attr: str,
        key_value: str,
    ) -> list[RelationData]:

        relations: list[RelationData] = []

        query = f"""
        match
            $e isa {entity_type},
                has {key_attr} "{key_value}";
            $r ($role: $e, $other_role: $x) isa $rel_type;
        get $r, $x, $rel_type;
        """
        seen_relations: set[str] = set()
        for result in tx.query.get(query):
            mapping = result.map
            rel = mapping["r"]

            rel_id = rel.get_iid()

            if rel_id in seen_relations:
                continue

            seen_relations.add(rel_id)
            rel_type = rel.get_type().get_label().name

            role_players = rel.get_players(tx)

            roles: dict[str, RelationRef] = {}

            for role_type, players in role_players.items():
                role_name = role_type.get_label().name

                for player in players:
                    player_type = player.get_type().get_label().name

                    key_attr_name = self._get_key_attr_for_type(player_type)

                    if key_attr_name is None:
                        continue

                    key_val = None

                    for attr in player.get_has(tx):
                        if attr.get_type().get_label().name == key_attr_name:
                            key_val = attr.get_value()
                            break

                    if key_val is None:
                        continue

                    roles[role_name] = {
                        "entity_type": player_type,
                        "key_attr": key_attr_name,
                        "key": str(key_val),
                    }

            attributes: dict[str, object] = {}

            for attr in rel.get_has(tx):
                attr_name = attr.get_type().get_label().name
                attributes[attr_name] = attr.get_value()

            relations.append(
                RelationData(
                    type=rel_type,
                    roles=roles,
                    attributes=attributes,
                )
            )

        return relations

    def _build_entity_relation_query(
            self,
            node: Node,
            relation_player_counts: list[int]
    ) -> str:
        """
        Build a query that matches relations for `node` where the relation has
        any of the given numbers of *other* players.

        Example:
            relation_player_counts = [1, 2]
            -> matches relations with:
            - queried entity + 1 other player
            - queried entity + 2 other players
        """
        branches = []

        for count in relation_player_counts:
            other_players = []
            fetch_players = []

            for i in range(1, count + 1):
                other_players.append(f"$rp{i} isa $rp{i}_type;")
                fetch_players.append(f"""
                    "relation_player_{i}": {{
                        'type': $rp{i}_type,
                        'role': $rp{i}_role,
                        'data': {{ $rp{i}.* }}
                    }}
                """)

            # relation links queried entity + all other players
            links = ", ".join(
                ["$e"] + [f"$rp{i}: $rp{i}_role" for i in range(1, count + 1)]
            )

            branch = f"""
            {{
                {' '.join(other_players)}
                $rel isa $rel_type, links ({links});
                fetch {{
                    {",".join(fetch_players)},
                    "relation": {{
                        'type': $rel_type,
                        'data': {{ $rel.* }}
                    }}
                }};
            }}
            """
            branches.append(branch)

        query = f"""
        match
            $e isa {node.entity_type}, has {node.key_attribute} "{node.id}";
        {chr(10).join(f"or {b}" if i > 0 else b for i, b in enumerate(branches))}
        """

        return query.strip()

    def _load_relations_batch(
        self,
        tx: Transaction,
        nodes: list[Node],
    ) -> dict[str, list[RelationData]]:
        """
        Load relations for nodes.

        Works with relations of any number of role players.
        """
        relations_by_node: dict[str, list[RelationData]] = {}

        for node in nodes:
            node_relations: list[RelationData] = relations_by_node.get(str(node.id), [])
            type_lookup = {r["type"]: r for r in node_relations}           
   
            query = f"""
            match
            # 1. Identify the person (add 'has name_key "Name"' here to filter)
            $p isa { node.entity_type }, has {node.key_attribute} "{node.id}"; 

            # 2. Find any relation that this person plays a role in
            $rel links ($p);
            $rel isa! $rel_type;

            fetch {{
                # 3. Fetch the relation's type and its owned data
                'relation': {{
                    'type': $rel_type,
                    'data': {{ $rel.* }}
                }},

                # 4. Open a sub-query to fetch all players in that specific relation
                'players': [
                    match 
                    $r links ($role: $player);
                    $player isa! $player_type;
                    
                    fetch {{
                        # 5. Fetch the player's role, exact type, and owned data
                        'role': $role,
                        'type': $player_type,
                        'data': {{ $player.* }}
                    }};
                ]
            }};
            """

            rows =tx.query(query).resolve().as_concept_documents()
            rows = list(rows)

            for row in rows:
                rel_type = row.get("relation").get('type').get('label')
                rel_data = row.get("relation").get('data')

                for player in row.get("players", []):
                    role = player.get("role").get('label')
                    type = player.get("type").get('label')
                    data = player.get("data")

                    key_attr = self._get_key_attr_for_type(type)
                    key_val = None

                    for other_attr, other_attr_val in data.items():
                        if other_attr == key_attr:
                            key_val = other_attr_val
                            break

                    if key_val is None:
                        continue

                    if rel_type not in type_lookup:
                        roles = {role: RelationRef(
                            entity_type=type,
                            key_attr=key_attr,
                            key=str(key_val),
                        )}

                        rel_attributes = {}

                        for rel_attr, rel_attr_val in rel_data.items():
                            rel_attributes[rel_attr] = rel_attr_val

                        entry = RelationData(type=rel_type, roles=roles, attributes=rel_attributes)

                        node_relations.append(entry)
                        type_lookup[rel_type] = entry
                    else:
                        existing = type_lookup[rel_type]

                        roles = existing["roles"]
                        roles[role] = RelationRef(
                            entity_type=type,
                            key_attr=key_attr,
                            key=str(key_val),
                        )
            
            relations_by_node[str(node.id)] = node_relations

        return relations_by_node

    def _fetch_to_nodes(
        self, rows: list[ConceptRow], include_relations: bool = False
    ) -> list[Node]:
        nodes: list[Node] = []

        with self.transaction(TransactionType.READ) as tx:
            raw_nodes = []

            for row in rows:
                entity_type = row.get("entity_type", {})
                entity_data = row.get("data", {})

                payload = {}

                for attr_name, values in entity_data.items():
                    if attr_name == "type" or not values:
                        continue
                    payload[attr_name] = values[0]["value"] if isinstance(values, list) else values

                if not payload:
                    continue

                key_attr = self._get_key_attribute(entity_type, payload)
                key_val = payload[key_attr]

                raw_nodes.append(
                    Node(
                        id=NodeId(str(key_val)),
                        entity_type=EntityType(entity_type),
                        key_attribute=KeyAttribute(key_attr),
                        payload_data=payload,
                        relations=(),
                    )
                )

            if include_relations:
                relations_map = self._load_relations_batch(tx, raw_nodes)

                nodes = [
                    replace(
                        node,
                        relations=tuple(relations_map.get(node.id, [])),
                    )
                    for node in raw_nodes
                ]

            else:
                nodes = raw_nodes

        return nodes

    def _deduplicate(self, nodes: list[Node]) -> list[Node]:
        """Merge overlapping nodes based on payload subset semantics."""

        merged: list[Node] = []

        def is_subset(a: Mapping[str, object], b: Mapping[str, object]) -> bool:
            return all(k in b and b[k] == v for k, v in a.items())

        for new in nodes:
            replaced = False
            for i, existing in enumerate(merged):
                if is_subset(new.payload_data, existing.payload_data):
                    replaced = True
                    break

                if is_subset(existing.payload_data, new.payload_data):
                    merged[i] = new
                    replaced = True
                    break

            if not replaced:
                merged.append(new)

        return merged

    def _get_all_attribute_labels(self) -> list[str]:
        if self._all_attr_cache is not None:
            return self._all_attr_cache

        query = """
        match
            $p owns $a;
        fetch {
            "a": $a
        };
        """

        rows = self.query_read(query).as_concept_documents()
        self._all_attr_cache = sorted({r.get("a").get('label') for r in list(rows)})
        return self._all_attr_cache

    def _get_all_nodes(self) -> list[Node]:
        self._ensure_connected()

        attr_labels = self._get_all_attribute_labels()
        if not attr_labels:
            return []

        query = """
        match
            entity $e;
        fetch {
            'e': $e,
        };
        """

        entity_rows = self.query_read(query).as_concept_documents()
        result = list(entity_rows)

        rows = []

        for row in result:
            entity_label = row.get("e").get("label")

            query = f"""
            match
                $e isa {entity_label};
            fetch {{
                'data': {{ $e.* }},
                'entity_type': '{entity_label}',
            }};
            """

            rows.extend(list(self.query_read(query).as_concept_documents()))

        return self._deduplicate(self._fetch_to_nodes(rows))

    def _get_entity_attribute_labels(self, entity_type: str) -> list[str]:
        if entity_type in self._entity_attr_cache:
            return self._entity_attr_cache[entity_type]

        query = f"""
        match
            {entity_type} owns $a;
        fetch {{
            "a": $a
        }};
        """

        results = self.query_read(query)
        labels = sorted({r["a"]["label"] for r in results})
        self._entity_attr_cache[entity_type] = labels
        return labels

    def get_nodes(self, filter: str | None) -> list[Node]:
        """
        Retrieve nodes using a filter query.

        Filter syntax
        -------------
        URL query string:

            entity=<entity_type>&<attr>=<value>&...

        Example
            entity=person&email=john@doe.com

        Behaviour
        ---------
        - Returns normalized Node objects
        - Deduplicates overlapping matches
        - Automatically infers key_attribute from schema

        Raises
        ------
        TypeError
            If filter is not a string
        ValueError
            If filter missing required entity parameter
        """

        if filter is None:
            return self._get_all_nodes()

        if not isinstance(filter, str):
            raise TypeError("filter must be a string or None")

        if not filter:
            raise ValueError("filter cannot be empty; use None to fetch all nodes")

        self._ensure_connected()

        parsed = self._parse_filter(filter)
        entity_type = parsed["entity_type"]
        attrs = parsed["attributes"]
        include_relations = parsed["include_relations"]

        match_block = self._build_match(entity_type, attrs)

        attr_labels = self._get_entity_attribute_labels(entity_type)
        if not attr_labels:
            return []

        query = f"""
        match
            {match_block};
        fetch {{
            'data': {{$e.*}},
            'entity_type': '{entity_type}',
        }};
        """

        rows = self.query_read(query).as_concept_documents()

        return self._fetch_to_nodes(list(rows), include_relations=include_relations)

    def remove_nodes(self, filter: str, allow_multiple: bool = False) -> int:
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

    def remove_node(self, filter: str) -> Node:
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
