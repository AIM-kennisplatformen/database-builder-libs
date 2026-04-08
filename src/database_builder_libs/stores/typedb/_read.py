from dataclasses import replace

from typedb.driver import Transaction, TransactionType

from database_builder_libs.models.node import EntityType, KeyAttribute, Node, NodeId
from database_builder_libs.stores.typedb._base import TypeDbBase
from database_builder_libs.stores.typedb._types import RelationData, RelationRef
from typing import Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from database_builder_libs.stores.typedb.typedb_store import TypeDbDatastore


class TypeDbReadMixin(TypeDbBase):
    def _entity_exists(self, entity_type: str, key_attr: str, key_value: str) -> bool:
        """Check if an entity with the given key attribute and value already exists."""

        query = f"""
        match
            $e isa {entity_type},
                has {key_attr} "{key_value}";
        limit 1;
        """

        result = self.query_read(query).as_concept_rows()

        return bool(len(list(result)) == 1)

    def _relation_exists(self: "TypeDbDatastore", rel: RelationData) -> bool:
        """Check if a specific relation linking exactly the same role players exists."""
        role_map = rel["roles"]
        attributes = rel.get("attributes", {})

        match_roles = [self._match_relation_ref(role, ref) for role, ref in role_map.items()]

        attr_match = ""
        if attributes != {}:
            attr_match = ", " + self._format_attributes(attributes)

        query = f"""
        match
            {" ".join(match_roles)}
            $r ({", ".join(f"{r}: ${r}" for r in role_map)}) isa {rel["type"]}
            {attr_match};
        limit 1;
        """

        rows = self.query_read(query).as_concept_rows()

        return bool(len(list(rows)) > 0)

    def _get_entity_key_value(self, entity_type: str, key_attr: str, thing) -> str:
        """Extract the key attribute value from a TypeDB Thing object."""
        # best effort: read all attributes and pick the key_attr
        for a in thing.get_has():
            if a.get_type().get_label().name == key_attr:
                return str(a.get_value())
        # fallback: re-query by iid if needed, but usually not necessary
        return ""

    def _get_relations_for_entity(
        self: "TypeDbDatastore",
        tx,
        entity_type: str,
        key_attr: str,
        key_value: str,
    ) -> list[RelationData]:
        """Fetch all relations linked to an entity, and resolve their role players."""

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

    def _load_relations_batch(
        self: "TypeDbDatastore",
        tx: Transaction,
        nodes: list[Node],
    ) -> dict[str, list[RelationData]]:
        """Batch load relations for a list of nodes using a single query per node."""
        relations_by_node: dict[str, list[RelationData]] = {}

        for node in nodes:
            node_relations: list[RelationData] = []

            query = f"""
            match
                $p isa {node.entity_type}, has {node.key_attribute} "{node.id}";
                $rel links ($p);
                $rel isa! $rel_type;
            fetch {{
                'relation': {{
                    'type': $rel_type,
                    'data': {{ $rel.* }}
                }},
                'players': [
                    match
                    $rel links ($role: $player);
                    $player isa! $player_type;
                    fetch {{
                        'role': $role,
                        'type': $player_type,
                        'data': {{ $player.* }}
                    }};
                ]
            }};
            """

            rows = list(tx.query(query).resolve().as_concept_documents())

            for row in rows:
                rel_type = row.get("relation").get("type").get("label")
                rel_data = row.get("relation").get("data")

                roles: dict[str, RelationRef] = {}

                for player in row.get("players", []):
                    role_label = player.get("role").get("label")
                    role = role_label.split(":")[-1] if ":" in role_label else role_label
                    player_type = player.get("type").get("label")
                    data = player.get("data")

                    key_attr = self._get_key_attr_for_type(player_type)
                    if key_attr is None:
                        continue

                    key_val = data.get(key_attr)
                    if key_val is None:
                        continue

                    roles[role] = RelationRef(
                        entity_type=player_type,
                        key_attr=key_attr,
                        key=str(key_val),
                    )

                rel_attributes: dict[str, object] = dict(rel_data.items())

                node_relations.append(
                    RelationData(
                        type=rel_type,
                        roles=roles,
                        attributes=rel_attributes,
                    )
                )

            relations_by_node[str(node.id)] = node_relations

        return relations_by_node

    def _fetch_to_nodes(self: "TypeDbDatastore", rows: list[dict], include_relations: bool = False) -> list[Node]:
        """Convert raw concept document rows to canonical Node objects."""
        nodes: list[Node] = []

        with self.transaction(TransactionType.READ) as tx:
            raw_nodes = []

            for row in rows:
                entity_type = row.get("entity_type")
                if not isinstance(entity_type, str):
                    continue
                entity_data = row.get("data")
                if not isinstance(entity_data, dict):
                    continue

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

    def _fetch_to_relations(self: "TypeDbDatastore", rows: list[dict]) -> list[RelationData]:
        """Convert raw concept document rows to RelationData objects."""
        relations: list[RelationData] = []

        for row in rows:
            rel = row.get("relation")
            if not rel:
                continue

            rel_type_obj = rel.get("type", {})
            rel_type = rel_type_obj.get("label") if isinstance(rel_type_obj, dict) else str(rel_type_obj)
            rel_data = rel.get("data", {})

            roles: dict[str, RelationRef] = {}

            for player in row.get("players", []):
                role_label = player.get("role", {}).get("label")
                if not role_label:
                    continue
                role = role_label.split(":")[-1] if ":" in role_label else role_label
                player_type = player.get("type", {}).get("label")
                data = player.get("data", {})

                key_attr = self._get_key_attr_for_type(player_type)
                if key_attr is None:
                    continue

                key_val = data.get(key_attr)
                if key_val is None:
                    continue

                roles[role] = RelationRef(
                    entity_type=player_type,
                    key_attr=key_attr,
                    key=str(key_val),
                )

            rel_attributes: dict[str, object] = dict(rel_data.items())

            relations.append(
                RelationData(
                    type=rel_type,
                    roles=roles,
                    attributes=rel_attributes,
                )
            )

        return relations

    def _deduplicate(self, nodes: list[Node]) -> list[Node]:
        """Merge overlapping nodes based on payload subset semantics."""

        merged: list[Node] = []

        def is_subset(a: Mapping, b: Mapping) -> bool:
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

    def _get_all_nodes(self: "TypeDbDatastore") -> list[Node]:
        """Retrieve and reconstruct all Node objects from the database."""
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

    def _get_all_relations(self: "TypeDbDatastore") -> list[RelationData]:
        """Retrieve and reconstruct all relations from the database."""
        self._ensure_connected()

        query = """
        match
            relation $t;
        fetch {
            't': $t,
        };
        """

        type_rows = self.query_read(query).as_concept_documents()
        result = list(type_rows)

        rows = []

        for row in result:
            rel_label = row.get("t").get("label")

            if rel_label == 'relation':
                continue

            query = f"""
            match
                $rel isa {rel_label};
                $rel isa! $rel_type;
            fetch {{
                'relation': {{
                    'type': $rel_type,
                    'data': {{ $rel.* }}
                }},
                'players': [
                    match
                    $rel links ($role: $player);
                    $player isa! $player_type;
                    fetch {{
                        'role': $role,
                        'type': $player_type,
                        'data': {{ $player.* }}
                    }};
                ]
            }};
            """
            rows.extend(list(self.query_read(query).as_concept_documents()))

        return self._fetch_to_relations(rows)

    def get_nodes(self: "TypeDbDatastore", filter: str | None) -> list[Node]:
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

    def get_relations(self: "TypeDbDatastore", filter: str | None) -> list[RelationData]:
        """
        Retrieve relations using a filter query.

        Filter syntax
        -------------
        URL query string:

            relation=<relation_type>&<attr>=<value>&...

        Example
        -------
            relation=friendship&since=2024

        Behaviour
        ---------
        - Returns normalized RelationData objects

        Raises
        ------
        TypeError
            If filter is not a string
        ValueError
            If filter missing required relation parameter
        """
        if filter is None:
            return self._get_all_relations()

        if not isinstance(filter, str):
            raise TypeError("filter must be a string or None")

        if not filter:
            raise ValueError("filter cannot be empty; use None to fetch all relations")

        self._ensure_connected()

        parsed = self._parse_relation_filter(filter)
        relation_type = parsed["relation_type"]
        attrs = parsed["attributes"]

        match_block = self._build_relation_match(relation_type, attrs)

        query = f"""
        match
            {match_block};
            $rel isa! $rel_type;
        fetch {{
            'relation': {{
                'type': $rel_type,
                'data': {{ $rel.* }}
            }},
            'players': [
                match
                $rel links ($role: $player);
                $player isa! $player_type;
                fetch {{
                    'role': $role,
                    'type': $player_type,
                    'data': {{ $player.* }}
                }};
            ]
        }};
        """

        rows = self.query_read(query).as_concept_documents()

        return self._fetch_to_relations(list(rows))
