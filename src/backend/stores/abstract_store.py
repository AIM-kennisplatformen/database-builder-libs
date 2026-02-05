from abc import ABC, abstractmethod

from pydantic import BaseModel
from typing import Sequence, Optional, List

class Node(BaseModel):
    id: str
    relations: list[str]
    vector_data: Sequence[float]
    payload_data: dict



class AbstractStore(ABC):
    """
    Abstract base class that should be implemented by datasources to save entities and relations to various datastores.
    """

    type: str

    @abstractmethod
    def connect_to_source(self) -> None:
        """Abstract method to connect to the sink."""
        ...
    
    @abstractmethod
    def store_node(self, node: Node) -> None:
        """Abstract method to store a node."""
        ...

    def get_nodes(self, filter: str | Sequence[float] | None) -> List[Node]:
        """Abstract method to retrieve a list of nodes."""
        ...

    def remove_node(self, filter: str) -> Node:
        """Abstract method to remove a node."""
        ...