from datetime import datetime
from pathlib import Path
import shutil
from typing import Any, List, Optional
from pydantic import PrivateAttr
from pyzotero import zotero

from loguru import logger
from database_builder_libs.models.abstract_source import AbstractSource, Content
from database_builder_libs.config import Settings


class ZoteroSource(AbstractSource):
    """
    Zotero implementation of AbstractSource.

    Provides incremental synchronization of a Zotero library and exposes items
    as canonical `Content` objects.

    Mapping
    -------
    Zotero item            → Content
    item.key               → Content.id_
    item.data              → Content.content
    item.dateModified      → Content.date

    Synchronization semantics
    -------------------------
    - get_list_artefacts() performs incremental sync using Zotero `since`
    - Returned timestamps are UTC
    - Identifiers are stable across runs
    - Deleted items are NOT reported (Zotero API limitation)

    Attachment handling
    -------------------
    download_zotero_item() retrieves the first attachment:
    - Prefers local Zotero storage when available
    - Falls back to API download

    Lifecycle
    ---------
    Connection is automatically established after model initialization.
    """

    _zotero: Optional[zotero.Zotero] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Called automatically by Pydantic after initialization."""
        self.connect_to_source()

    def connect_to_source(self) -> None:
        """
        Initialize Zotero API client.

        Idempotent: calling multiple times does not recreate the client.

        Raises
        ------
        RuntimeError
            If credentials are missing or invalid.
        """

        if self._zotero is not None:
            return  # already connected

        self._zotero = zotero.Zotero(
            library_id=Settings.ZOTERO_LIBRARY_ID,
            library_type=Settings.ZOTERO_LIBRARY_TYPE,
            api_key=Settings.ZOTERO_API_KEY,
        )

    def get_all_documents_metadata(self, collection_id: str) -> List[dict[str, Any]]:
        """Retrieve the metadata of all documents within collection

        This function calls the zotero collection items api:
        'https://api.zotero.org/users/<library_id>/collections/<collection_id>/items/top'
        Using the pyzotero library and returns a list containing dictionaries of metadata.
        Keep in mind, that the structures returned by this function are large and take some time to retrieve.

        Args:
            `collection_id`: The collection to retrieve document metadata from
                            (should be visible in WebURL when using zotero webportal)

        Yields:
            List containing document-metadata dict for all documents in the library (one dict per document).
            The dict output closely resembles the dict output format of pyzotero:
            https://pyzotero.readthedocs.io/en/latest/#zotero.Zotero.collection_items_top
        """
        if self._zotero is None:
            raise RuntimeError("Zotero client not initialized")

        return self._zotero.everything(
            self._zotero.collection_items_top(collection_id, limit=None)
        )

    def download_zotero_item(
        self,
        *,
        item_id: str,
        download_path: str,
    ) -> None:
        """Download the first attachment of specified zotero item to specified path

        This function is a wrapper around the dump api to provide a means to download attachments of zotero items using
        local & cloud api. As the default (at this time) dump api_call only provides cloud download functionality.

        Args:
           `item_id`: The specific item_id of the item to get the attachment/pdf from (`key` attribute from above mentioned zotero dict)
           `download_path`: The folder to download the item to, the file_path will be -> <download_path>/<item_id>.pdf
        """
        if self._zotero is None:
            raise RuntimeError("Zotero client not initialized")

        logger.debug("Fetching File: %s", item_id)

        children = self._zotero.children(item_id)
        if not children:
            logger.warning("No child attachments found for item %s", item_id)
            return

        attachments = [
            c for c in children if c.get("data", {}).get("itemType") == "attachment"
        ]
        if not attachments:
            logger.warning("No attachment-type children for item %s", item_id)
            return

        attachment = attachments[0]
        data = attachment.get("data", {})
        local_path = data.get("path")

        target = Path(download_path) / f"{item_id}.pdf"

        if local_path and Path(local_path).exists():
            logger.info("Copying local attachment from %s", local_path)
            shutil.copy(local_path, target)
            return

        logger.info("Local attachment not found, downloading via Zotero API")

        self._zotero.dump(
            itemkey=attachment["key"],
            filename=target.name,
            path=download_path,
        )

    def get_list_artefacts(
        self, last_synced: Optional[datetime]
    ) -> list[tuple[str, datetime]]:
        """
        Return Zotero items modified after `last_synced`.

        Parameters
        ----------
        last_synced : datetime | None
            UTC timestamp of last successful sync.
            If None, all items are returned.

        Returns
        -------
        list[(item_key, modified_time)]

        Sync guarantees
        ---------------
        - item_key is stable across runs
        - timestamps are timezone-aware UTC
        - includes newly created and modified items
        - DOES NOT include deleted items (Zotero limitation)

        Notes
        -----
        Zotero `since` uses server modification time, not file change time.
        """
        if self._zotero is None:
            raise RuntimeError("Zotero client not initialized")

        items = self._zotero.everything(
            self._zotero.items(since=int(last_synced.timestamp()))
            if last_synced
            else self._zotero.items()
        )

        artefacts: list[tuple[str, datetime]] = []

        for item in items:
            data = item.get("data", {})
            key = data.get("key")
            modified = data.get("dateModified")

            if not key or not modified:
                continue

            artefacts.append(
                (
                    key,
                    datetime.fromisoformat(modified.replace("Z", "+00:00")),
                )
            )

        return artefacts

    def get_content(self, artefacts: list[tuple[str, datetime]]) -> list[Content]:
        """
        Fetch normalized content for Zotero items.

        Each artefact is retrieved individually and converted to `Content`.

        Guarantees
        ----------
        - One Content object per artefact
        - Content.date matches modification timestamp from list phase
        - Content.content contains raw Zotero `data` field

        This method does not download attachments.
        """
        if self._zotero is None:
            raise RuntimeError("Zotero client not initialized")

        contents: list[Content] = []

        for item_key, modified in artefacts:
            item = self._zotero.item(item_key)
            data = item.get("data", {})

            contents.append(
                Content(
                    id_=item_key, 
                    date=modified,
                    content=data,
                )
            )

        return contents

