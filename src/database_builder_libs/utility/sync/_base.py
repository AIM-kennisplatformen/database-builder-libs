from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

Artifact = tuple[str, datetime]
ConflictItem = str


class AbstractSyncTracker(ABC):
    """
    Abstract interface for tracking synchronization state across data sources.

    A SyncTracker is responsible for:
    1. Recording when each source was last synchronized
    2. Tracking individual artifact modification times
    3. Detecting conflicts where the same artifact has different
       modification times across sources

    Lifecycle
    ---------
    Call ``start_sync(source_name)`` before fetching artifacts.
    Call ``finish_sync(source_name, artifacts)`` after storing them.

    Typical workflow::

        tracker = SqliteSyncTracker()
        last_sync = tracker.start_sync("Zotero")
        artifacts = source.get_list_artefacts(last_synced=last_sync)
        conflicts = tracker.finish_sync("Zotero", artifacts)
        tracker.close()
    """

    @abstractmethod
    def start_sync(self, source_name: str) -> Optional[float]:
        """
        Return the last successful sync timestamp for a source.

        Parameters
        ----------
        source_name : str
            Unique identifier for the data source.

        Returns
        -------
        float | None
            Unix timestamp of the last sync, or None if the source
            has never been synchronized.

        Notes
        -----
        On first call for an unknown source the implementation should
        register the source and return None.
        """
        ...

    @abstractmethod
    def finish_sync(
        self,
        source_name: str,
        artifacts: list[Artifact],
    ) -> list[ConflictItem]:
        """
        Record artifacts that were just synchronized and detect conflicts.

        Parameters
        ----------
        source_name : str
            Unique identifier for the data source.
        artifacts : list[Artifact]
            List of (item_key, modification_time) tuples returned
            by the source.

        Returns
        -------
        list[ConflictItem]
            Item keys whose modification times differ across sources.
            Empty list means no conflicts were detected.

        Notes
        -----
        The implementation must update the source's last_sync_time
        to the current time after recording the artifacts.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """
        Release any resources held by the tracker.

        Behaviour
        ---------
        Must be safe to call multiple times. After close() the tracker
        should not be used for further operations.
        """
        ...
