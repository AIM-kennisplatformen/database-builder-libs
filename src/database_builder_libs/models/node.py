from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence, NewType

NodeId = NewType("NodeId", str)
"""
Globally stable identifier of a knowledge entity.

Properties
----------
- Uniquely identifies the same real-world entity across all systems
- Must be deterministic across ingestion runs
- Must not encode storage-specific information (database IDs, row numbers)
- Safe for use as foreign key in relations

Examples
--------
"user:42"
"doi:10.1000/182"
"sharepoint:file:abc123"
"""


EntityType = NewType("EntityType", str)
"""
Logical category of a Node.

Represents ontology class — NOT a storage label.

Constraints
-----------
- Stable for a given NodeId
- Low cardinality
- Used for indexing and filtering

Examples
--------
"person"
"document"
"organization"
"concept"
"""


KeyAttribute = NewType("KeyAttribute", str)
"""
Name of the primary human-readable identifier inside a Node payload.

This is a presentation hint, not identity.

Examples
--------
"name"
"title"
"email"
"filename"
"""


Payload = Mapping[str, object]
"""
Structured attributes describing a Node.

Requirements
------------
- JSON-serializable
- Order-independent
- Deterministic for identical source data
- Values must be immutable or safely copyable

This data may be merged across updates.
"""


Relation = Mapping[str, object]
"""
Edge description between two Nodes.

Minimum required keys
---------------------
"type" : str
    Relationship type
"target" : NodeId
    Identifier of the related node

Optional keys
-------------
Any additional metadata describing the relation.

Constraints
-----------
- Must not encode storage-specific fields
- Must be JSON-serializable
- Duplicate relations should be treated as identical
"""


@dataclass(slots=True, frozen=True)
class Node:
    """
    Canonical storage-agnostic representation of a knowledge entity.

    A Node is the normalized form of structured information extracted from
    external sources. All database adapters MUST translate their internal
    records into this structure before persistence or retrieval.

    Identity
    --------
    The node identity is defined exclusively by `id`.

    Nodes with identical `id` MUST represent the same real-world entity.
    Stores must overwrite existing nodes instead of creating duplicates.

    Fields
    ------
    id : NodeId
        Globally stable identifier of the entity.
        Must remain constant across synchronization runs and storage backends.

    payload_data : Mapping[str, object]
        Structured attributes describing the entity (properties).

        Requirements:
        - JSON-serializable
        - Deterministic for identical source state
        - Order-independent
        - Safe to merge across updates

    relations : Sequence[Relation]
        Outgoing relationships from this node to other nodes.

        Each relation mapping should minimally contain:
            {
                "type": <relation name>,
                "target": <NodeId>
            }

        Constraints:
        - Must not contain cyclic self-references unless meaningful
        - Order does not carry semantic meaning
        - Duplicate relations should be ignored by stores

    entity_type : EntityType
        Logical category of the entity (e.g., "person", "document", "concept").
        Used for indexing, filtering and schema interpretation.
        Must remain stable for a given node id.

    key_attribute : KeyAttribute
        Name of the primary human-readable identifier inside payload_data
        (e.g., "email", "title", "name").

    """
    id: NodeId
    payload_data: Payload = field(default_factory=dict)
    relations: Sequence[Relation] = field(default_factory=tuple)

    entity_type: EntityType = EntityType("node")
    key_attribute: KeyAttribute = KeyAttribute("id")