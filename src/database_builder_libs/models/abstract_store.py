from abc import ABC, abstractmethod
from typing import List
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

    def __init__(self) -> None:
        self._connected: bool = False
        self._connecting: bool = False
    
    def connect(self, config: dict | None = None) -> None:
            """
            Establish connection to the backend.

            This method is idempotent. Calling it multiple times must be safe.

            Parameters
            ----------
            config : Any | None
                Backend-specific configuration object.

            Raises
            ------
            ConnectionError
                Backend unreachable.
            RuntimeError
                Backend misconfigured.
            """
            if self._connected:
                return

            self._connecting = True
            try:
                self._connect_impl(config)
                self._connected = True
            finally:
                self._connecting = False

    
    @abstractmethod
    def _connect_impl(self, config: dict | None) -> None:
        """Backend-specific connection logic."""
        ...

    def _ensure_connected(self) -> None:
        if not (self._connected or self._connecting):
            raise RuntimeError(
                f"{self.__class__.__name__} used before connect() was called"
            )

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