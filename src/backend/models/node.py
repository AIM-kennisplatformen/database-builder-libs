from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence, NewType

# --- Strong semantic types ---

NodeId = NewType("NodeId", str)
EntityType = NewType("EntityType", str)
KeyAttribute = NewType("KeyAttribute", str)

Payload = Mapping[str, object]
Relation = Mapping[str, object]


# --- Domain model ---

@dataclass(slots=True, frozen=True)
class Node:
    """
    Canonical domain representation of a stored knowledge object.

    This object MUST be storage-agnostic.
    All graph database adapters normalize their records into this structure.
    """

    id: NodeId
    payload_data: Payload = field(default_factory=dict)
    relations: Sequence[Relation] = field(default_factory=tuple)

    # semantic graph fields (used by TypeDB / KG stores)
    entity_type: EntityType = EntityType("node")
    key_attribute: KeyAttribute = KeyAttribute("id")

    # embedding metadata (future-proofing)
    embedding_model: str = "unknown"
