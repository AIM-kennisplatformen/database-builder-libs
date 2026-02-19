from abc import ABC, abstractmethod
from typing import Sequence, List
from database_builder_libs.models.node import Node


class AbstractStore(ABC):
    """
    Abstract persistence layer for storing and retrieving Nodes.

    A Store represents a backend capable of persisting structured nodes and
    retrieving them via identifier lookup, textual filtering, or vector similarity.

    Typical implementations:
    - SQL/NoSQL database
    - Vector database (e.g., embeddings search)
    - Graph database
    - In-memory index

    Consistency requirements
    ------------------------
    Implementations must ensure:
    - Stable node identity across reads
    - Deterministic retrieval for identical queries
    - Idempotent storage: storing the same Node twice must not create duplicates
    """

    @abstractmethod
    def connect_to_source(self) -> None:
        """
        Initialize connection to the storage backend.

        This method should:
        - Open database/session connection
        - Validate schema or index availability
        - Prepare search indexes if necessary

        Raises
        ------
        ConnectionError
            If the backend cannot be reached.
        RuntimeError
            If the storage backend is misconfigured.
        """
        ...

    @abstractmethod
    def store_node(self, node: Node) -> None:
        """
        Persist a Node into the store.

        Behaviour
        ---------
        - If the node already exists (same unique identifier), it must be updated.
        - Operation must be idempotent.

        Parameters
        ----------
        node : Node
            The node to persist.

        Raises
        ------
        RuntimeError
            If called before connect_to_source().
        ValueError
            If the node is invalid for this backend.
        """
        ...

    @abstractmethod
    def get_nodes(self, filter: str | None) -> List[Node]:
        """
        Retrieve nodes matching a query filter.

        Parameters
        ----------
        filter : str | Sequence[float] | None
            Query selector defining retrieval mode:

            - str → keyword / identifier / query string search
            - None → return all stored nodes

        Returns
        -------
        List[Node]
            Matching nodes ordered by relevance:
            - text search → relevance score descending
            - vector search → similarity descending
            - None → implementation-defined but deterministic ordering

        Guarantees
        ----------
        - No duplicate nodes returned
        - Same query produces stable ordering if backend unchanged

        Raises
        ------
        RuntimeError
            If called before connect_to_source().
        """
        ...

    @abstractmethod
    def remove_node(self, filter: str) -> Node:
        """
        Remove a single node identified by filter.

        Parameters
        ----------
        filter : str
            Unique identifier of the node to remove.

        Returns
        -------
        Node
            The removed node.

        Behaviour
        ---------
        - Must remove exactly one node
        - Must fail if multiple or zero matches

        Raises
        ------
        KeyError
            If no node matches the filter.
        RuntimeError
            If more than one node matches.
        """
        ...