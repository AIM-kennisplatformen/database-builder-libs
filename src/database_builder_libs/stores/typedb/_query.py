from typing import Mapping
from urllib.parse import parse_qs

from database_builder_libs.models.node import Node
from database_builder_libs.stores.typedb._base import TypeDbBase
from database_builder_libs.stores.typedb._types import RelationRef


class TypeDbQueryMixin(TypeDbBase):
    def _build_match(self, entity_type: str | None, attrs: Mapping[str, str] | None) -> str:
        """Build a TypeQL match block for a specific entity type and its attributes."""
        clauses = []

        if entity_type:
            clauses.append(f"$e isa {entity_type}")
        else:
            clauses.append("$e isa entity")

        if attrs:
            clauses.append(self._format_attribute_match(dict(attrs)))

        return ", ".join(clauses)

    def _build_relation_match(self, relation_type: str | None, attrs: Mapping[str, str] | None) -> str:
        """Build a TypeQL match block for a specific relation type and its attributes."""
        clauses = []

        if relation_type:
            clauses.append(f"$rel isa {relation_type}")
        else:
            clauses.append("$rel isa relation")

        if attrs:
            clauses.append(self._format_attribute_match(dict(attrs)))

        return ", ".join(clauses)

    def _format_attributes(self, payload: Mapping[str, object]) -> str:
        """Format a payload mapping into a TypeQL attribute injection string."""
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

    def _format_attribute_match(self, attrs: dict[str, str]) -> str:
        """Format an attribute mapping into a TypeQL match attribute string, inferring types."""
        clauses = []

        for attr, value in attrs.items():
            try:
                int_val = int(value)
                clauses.append(f"has {attr} {int_val}")
            except (ValueError, TypeError):
                try:
                    float_val = float(value)
                    clauses.append(f"has {attr} {float_val}")
                except (ValueError, TypeError):
                    clauses.append(f'has {attr} "{value}"')

        return ",\n       ".join(clauses)

    def _parse_filter(self, filter: str) -> dict:
        """Parse an entity filter string into entity type, attributes, and options."""
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

    def _parse_relation_filter(self, filter: str) -> dict:
        """Parse a relation filter string into relation type and attributes."""
        parsed = parse_qs(filter, keep_blank_values=False)

        # flatten single values
        params = {k: v[0] for k, v in parsed.items()}

        if "relation" not in params:
            raise ValueError("Missing required 'relation' filter")

        relation_type = params.pop("relation")

        return {
            "relation_type": relation_type,
            "attributes": params,
        }

    def _match_relation_ref(self, role: str, ref: RelationRef) -> str:
        """Format a TypeQL match block for a specific relation role reference."""
        return f"""
        ${role} isa {ref["entity_type"]},
            has {ref["key_attr"]} "{ref["key"]}";
        """

    def _build_entity_relation_query(self, node: Node, relation_player_counts: list[int]) -> str:
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
