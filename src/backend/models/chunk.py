from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any, Sequence


DocumentId = str


@dataclass(slots=True)
class Chunk:
    """
    Smallest retrievable semantic unit.

    Stored ONLY in vector DB.
    """

    # global identity (connects to TypeDB)
    document_id: DocumentId

    # stable order inside document
    chunk_index: int

    # actual retrievable text
    text: str

    # embedding vector
    vector: Sequence[float]

    # optional lightweight metadata (page, section, etc)
    metadata: Mapping[str, Any] | None = None
