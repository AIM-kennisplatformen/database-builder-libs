from datetime import datetime
from pathlib import Path
import shutil
from typing import Any, List, Mapping, Optional
from pydantic import PrivateAttr, BaseModel
from pyzotero import zotero

from loguru import logger
from database_builder_libs.models.abstract_source import AbstractSource, Content
from datetime import timezone
from dateutil.parser import isoparse

def _to_utc_ts(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.astimezone(timezone.utc).timestamp())

class ZoteroConfig(BaseModel):
    library_id: str
    library_type: str
    api_key: str
    collection: str | None = None

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
    `connect()` must be called before using the source.
    """

    _zotero: Optional[zotero.Zotero] = PrivateAttr(default=None)
    _config: Optional[ZoteroConfig] = PrivateAttr(default=None)

    def _connect_impl(self, config: Mapping[str, Any]) -> None:
        self._config = ZoteroConfig(**config)
        self._zotero = zotero.Zotero(**self._config.model_dump(exclude={"collection"}))

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
        self._ensure_connected()
        assert self._zotero is not None

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
        self._ensure_connected()
        assert self._zotero is not None

        logger.debug("Fetching File: {}", item_id)

        children = self._zotero.children(item_id)
        if not children:
            logger.warning("No child attachments found for item {}", item_id)
            return

        attachments = [
            c for c in children if c.get("data", {}).get("itemType") == "attachment"
        ]
        if not attachments:
            logger.warning("No attachment-type children for item {}", item_id)
            return

        attachment = attachments[0]
        data = attachment.get("data", {})
        local_path = data.get("path")

        download_dir = Path(download_path)
        download_dir.mkdir(parents=True, exist_ok=True)
        target = download_dir / f"{item_id}.pdf"
        if local_path and Path(local_path).exists():
            logger.info("Copying local attachment from {}", local_path)
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
        self._ensure_connected()
        assert self._zotero is not None
        assert self._config is not None

        if self._config.collection:
            items_iter = self._zotero.collection_items_top(self._config.collection, limit=None)
        else:
            items_iter = self._zotero.items()

        items = list(self._zotero.everything(items_iter))

        artefacts: list[tuple[str, datetime]] = []

        # If no cursor → epoch
        if last_synced is None:
            last_synced = datetime(1970, 1, 1, tzinfo=timezone.utc)

        for item in items:
            data = item.get("data", {})
            key = data.get("key")

            # Zotero reality: sometimes only dateAdded exists
            modified_str = data.get("dateModified") or data.get("dateAdded")
            if not key or not modified_str:
                continue

            modified = isoparse(modified_str).astimezone(timezone.utc)

            if modified > last_synced:
                artefacts.append((key, modified))

        return artefacts

    def get_content(self, artefacts: list[tuple[str, datetime]]) -> list[Content]:
        """
        Fetch normalized content for Zotero items.

        Each artefact is retrieved individually and converted to `Content`.

        Guarantees
        ----------
        - One Content object per artefact
        - Content.date reflects the modification timestamp observed during listing.
        - Content.content may represent a newer revision if the item changed during retrieval.
        - Content.content contains raw Zotero `data` field

        This method does not download attachments.
        """
        self._ensure_connected()
        assert self._zotero is not None

        contents: list[Content] = []

        for item_key, modified in artefacts:
            item = self._zotero.item(item_key)
            if not item:
                continue
            data = item.get("data", {})

            contents.append(
                Content(
                    id_=item_key, 
                    date=modified,
                    content=data,
                )
            )

        return contents

