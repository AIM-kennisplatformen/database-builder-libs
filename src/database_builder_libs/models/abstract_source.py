from abc import ABC, abstractmethod
from datetime import datetime
from typing import Mapping, Optional

from pydantic import BaseModel


class Content(BaseModel):
    """
    Representation of a single artefact retrieved from a source.

    An artefact corresponds to a uniquely identifiable entity in the external
    system (e.g., SharePoint document, Zotero item, database record).

    Attributes
    ----------
    date : datetime
        Last modification timestamp of the artefact in the source system.
        Must be timezone-aware (UTC recommended).
    id_ : str
        Stable unique identifier of the artefact in the source.
        This identifier MUST remain constant across synchronizations.
    content : dict
        Normalized payload retrieved from the source.

        The structure is implementation specific but must be JSON-serializable
        and deterministic: identical source state must produce identical dict.
    """
    date: datetime
    id_: str
    content: dict


class AbstractSource(ABC, BaseModel):
    """
    Abstract interface describing a synchronizable external data source.

    A Source implementation is responsible for:
    1. Establishing a connection to a remote system
    2. Discovering which artefacts changed since a timestamp
    3. Retrieving normalized content for those artefacts

    The interface is designed for incremental synchronization workflows.

    Lifecycle
    ---------
    connect_to_source() MUST be called before any other method.

    Consistency guarantees
    ----------------------
    Implementations must ensure:
    - Stable artefact identifiers across runs
    - Monotonic modification timestamps per artefact
    - Deterministic content serialization

    Typical implementations:
    SharePoint, Zotero, REST APIs, file repositories, databases.
    """


    _connected: bool = False
    _config: Mapping[str, str] | None = None
    """
        Configuration
    -------------
    `config` is a mapping containing backend-specific parameters.

    Examples
    --------
    Zotero:
        {"library_id": "...", "api_key": "..."}
    Filesystem:
        {"path": "/data"}
    REST API:
        {"url": "...", "token": "..."}

    Consistency guarantees
    ----------------------
    Implementations must ensure:
    - Stable artefact identifiers across runs
    - Monotonic modification timestamps
    - Deterministic content serialization
    """
    
    def connect(self, config: Mapping[str, str] | None = None) -> None:
        """
        Establish connection to the external source.

        Idempotent: safe to call multiple times.

        Raises
        ------
        ConnectionError
        PermissionError
        ValueError
        """
        if self._connected:
            return

        self._connect_impl(config or {})
        self._config = config
        self._connected = True

    @abstractmethod
    def _connect_impl(self, config: Mapping[str, str]) -> None:
        """Backend-specific connection logic."""
        ...

    def _ensure_connected(self) -> None:
        """Raise if the source is used before connect()."""
        if not self._connected:
            raise RuntimeError(
                f"{self.__class__.__name__} used before connect() was called"
            )

    @abstractmethod
    def get_list_artefacts(
        self, last_synced: Optional[datetime]
    ) -> list[tuple[str, datetime]]:
        """
        Return identifiers of artefacts modified since `last_synced`.

        Parameters
        ----------
        last_synced : datetime | None
            UTC timestamp of last successful synchronization.
            If None, the implementation must return ALL available artefacts.

        Returns
        -------
        list[tuple[str, datetime]]
            A list of (artefact_id, last_modified_timestamp).

        Requirements
        ------------
        - Returned timestamps must be timezone-aware.
        - Each artefact_id must appear at most once.
        - The list should be ordered by timestamp ascending if possible.

        Raises
        ------
        RuntimeError
            If called before connect_to_source().
        """

    @abstractmethod
    def get_content(self, artefacts: list[tuple[str, datetime]]) -> list[Content]:
        """
        Retrieve normalized content for provided artefacts.

        Parameters
        ----------
        artefacts : list[tuple[str, datetime]]
            Artefacts returned from get_list_artefacts().

        Returns
        -------
        list[Content]
            Content objects corresponding to requested artefacts.

        Guarantees
        ----------
        - One Content object per artefact_id
        - Returned content.date must match the provided timestamp
          unless the source updated during retrieval.

        Notes
        -----
        Implementations should batch requests where possible.

        Raises
        ------
        RuntimeError
            If called before connect_to_source().
        KeyError
            If an artefact no longer exists.
        """
