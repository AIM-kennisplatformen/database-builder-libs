import re
from typing import Mapping

from database_builder_libs.stores.typedb._base import TypeDbBase


class TypeDbSchemaMixin(TypeDbBase):
    def _load_key_attrs_from_schema(self) -> dict[str, str]:
        assert self.typedb_driver is not None
        assert self.database is not None

        schema_text = self.typedb_driver.databases.get(self.database).schema()

        # First pass: map each type to its parent
        parent_map: dict[str, str] = {}
        key_map: dict[str, str] = {}
        current_type = None

        for line in schema_text.splitlines():
            line = line.strip()
            type_match = re.match(r'^(?:entity|relation)\s+([\w\-]+)', line)
            if type_match:
                current_type = type_match.group(1)

            sub_match = re.search(r'sub\s+([\w\-]+)', line)
            if sub_match and current_type:
                parent = sub_match.group(1)
                if parent not in ("entity", "relation", "attribute"):
                    parent_map[current_type] = parent

            if current_type:
                key_match = re.search(r'owns\s+([\w\-]+)\s+@key', line)
                if key_match:
                    key_map[current_type] = key_match.group(1)

        # Second pass: propagate @key from parent to subtypes that don't have one
        def resolve_key(type_name: str, depth: int = 0) -> str | None:
            if depth > 20:
                return None
            if type_name in key_map:
                return key_map[type_name]
            parent = parent_map.get(type_name)
            if parent:
                return resolve_key(parent, depth + 1)
            return None

        all_types = set(key_map.keys()) | set(parent_map.keys())
        for type_name in all_types:
            if type_name not in key_map:
                inherited = resolve_key(type_name)
                if inherited:
                    key_map[type_name] = inherited

        return key_map

    def _get_key_attribute(self, entity_type: str, payload: Mapping[str, object]) -> str:
        key = self._key_attr_cache.get(entity_type)
        return key if key else sorted(payload.keys())[0]

    def _get_key_attr_for_type(self, entity_type: str) -> str | None:
        return self._key_attr_cache.get(entity_type)

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
