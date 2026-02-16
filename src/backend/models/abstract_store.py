from abc import ABC, abstractmethod
from typing import Sequence, List
from backend.models.node import Node


class AbstractStore(ABC):

    @abstractmethod
    def connect_to_source(self) -> None:
        ...

    @abstractmethod
    def store_node(self, node: Node) -> None:
        ...

    @abstractmethod
    def get_nodes(self, filter: str | Sequence[float] | None) -> List[Node]:
        ...

    @abstractmethod
    def remove_node(self, filter: str) -> Node:
        ...
