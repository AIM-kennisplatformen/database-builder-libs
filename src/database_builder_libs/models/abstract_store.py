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
        raise NotImplementedError

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
        raise NotImplementedError

    @abstractmethod
    def get_nodes(self, filter: str | None) -> List[Node]:
        """
        Retrieve nodes from the store.

        Retrieval Modes
        ---------------
        filter is interpreted as:

        - str  → Selection query
            Returns nodes that directly match stored records.
            Multiple results representing different stored entities
            MUST be preserved (no merging/deduplication).

        - None → Reconstruction query
            Returns the canonical set of nodes represented by the backend.
            Implementations MUST merge overlapping representations and
            return a normalized, duplicate-free set of Nodes.

        Returns
        -------
        List[Node]
            Deterministically ordered list of nodes.

        Guarantees
        ----------
        - Stable ordering for identical queries if backend unchanged
        - filter=None returns a duplicate-free canonical node set
        - filter=str preserves multiplicity of stored entities

        Raises
        ------
        RuntimeError
            If called before connect().
        """
        raise NotImplementedError

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
        ValueError
            If multiple nodes match the filter.
        """
        raise NotImplementedError