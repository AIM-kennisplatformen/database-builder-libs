from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any, List, Mapping, Optional
from pydantic import PrivateAttr, BaseModel
from pyzotero import zotero

from loguru import logger
from database_builder_libs.models.abstract_source import AbstractSource, Content
from dateutil.parser import isoparse


@dataclass(frozen=True)
class _FileTypeInfo:
    mime: str
    extension: str  # includes leading dot


FILE_TYPES: dict[str, _FileTypeInfo] = {
    "pdf":  _FileTypeInfo("application/pdf",                                                          ".pdf"),
    "epub": _FileTypeInfo("application/epub+zip",                                                     ".epub"),
    "docx": _FileTypeInfo("application/vnd.openxmlformats-officedocument.wordprocessingml.document",  ".docx"),
    "doc":  _FileTypeInfo("application/msword",                                                       ".doc"),
    "txt":  _FileTypeInfo("text/plain",                                                               ".txt"),
    "html": _FileTypeInfo("text/html",                                                                ".html"),
}

# Reverse lookup: MIME type → extension, derived from FILE_TYPES
MIME_TO_EXT: dict[str, str] = {info.mime: info.extension for info in FILE_TYPES.values()}


class FileType:
    """Pre-defined file type groups for convenient reference."""

    PDF       = ["pdf"]
    EPUB      = ["epub"]
    EBOOKS    = ["epub", "pdf"]       # E-book formats (EPUB preferred)
    DOCUMENTS = ["pdf", "docx", "doc"]  # Document formats
    TEXT      = ["txt", "html"]       # Plain text formats
    OFFICE    = ["docx", "doc"]       # Office documents
    ALL       = list(FILE_TYPES)      # All types in insertion-order priority


class ZoteroConfig(BaseModel):
    library_id: str
    library_type: str
    api_key: str
    collection: str | None = None


class ZoteroSource(AbstractSource[ZoteroConfig]):
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
    download_zotero_item() retrieves the best attachment:
    - File type preferences (pdf, epub, docx, doc, txt, html)
    - Configurable priority order via accept_types parameter
    - Intelligent fallback when preferred types unavailable
    - Prefers local Zotero storage when available
    - Falls back to API download
    - Saves with correct file extension based on content type

    File Type Configuration
    -----------------------
    Use the FileType class for convenient file type groups:

    >>> # EPUB before PDF (for e-books)
    >>> zotero.download_zotero_item(
    ...     item_id="ABC123",
    ...     download_path="./downloads/",
    ...     accept_types=FileType.EBOOKS
    ... )

    >>> # PDF only, strict mode
    >>> zotero.download_zotero_item(
    ...     item_id="ABC123",
    ...     download_path="./downloads/",
    ...     accept_types=FileType.PDF,
    ...     allow_fallback=False
    ... )

    >>> # Documents (PDF, DOCX, DOC), no plain text
    >>> zotero.download_zotero_item(
    ...     item_id="ABC123",
    ...     download_path="./downloads/",
    ...     accept_types=FileType.DOCUMENTS
    ... )

    Lifecycle
    ---------
    `connect()` must be called before using the source.
    """

    _zotero: Optional[zotero.Zotero] = PrivateAttr(default=None)
    _config: Optional[ZoteroConfig] = PrivateAttr(default=None)

    def _connect_impl(self, config: Mapping[str, Any]) -> None:
        self._config = ZoteroConfig(**config)
        self._zotero = zotero.Zotero(**self._config.model_dump(exclude={"collection"}))

    def _get_zotero(self) -> zotero.Zotero:
        self._ensure_connected()
        if self._zotero is None:
            raise RuntimeError("Zotero client not initialized after connect()")
        return self._zotero

    def _get_config(self) -> ZoteroConfig:
        self._ensure_connected()
        if self._config is None:
            raise RuntimeError("Zotero config not initialized after connect()")
        return self._config

    def _select_best_attachment(
        self,
        attachments: list[dict],
        accept_types: list[str],
        allow_fallback: bool = True,
    ) -> Optional[dict]:
        """
        Select the best attachment from a list, ranked by preferred file types.

        Tries each type in `accept_types` in order, returning the first match.
        If nothing matches and `allow_fallback` is True, returns the first
        attachment regardless of type. Returns None if the list is empty or
        no match is found without fallback.

        Args:
            attachments:    Attachment dicts from the Zotero API.
            accept_types:   File-type priority order, e.g. ["pdf", "epub"].
                            Valid values are keys of FILE_TYPES.
            allow_fallback: When True, return any attachment if no preferred
                            type is found. When False, return None instead.
        """
        if not attachments:
            return None

        def content_type(a: dict) -> str:
            return a.get("data", {}).get("contentType", "")

        for file_type in accept_types:
            type_info = FILE_TYPES.get(file_type)
            if type_info is None:
                logger.warning("Unknown file type '{}', skipping", file_type)
                continue
            if match := next((a for a in attachments if content_type(a) == type_info.mime), None):
                logger.debug("Selected {} attachment for item", file_type)
                return match

        if allow_fallback:
            logger.debug("No preferred type found (wanted: {}), using first attachment", accept_types)
            return attachments[0]

        logger.warning("No acceptable attachment found (wanted: {})", accept_types)
        return None

    def get_all_documents_metadata(self, collection_id: str) -> List[dict[str, Any]]:
        """Retrieve the metadata of all documents within collection.

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
        z = self._get_zotero()
        return z.everything(z.collection_items_top(collection_id, limit=None))

    def download_zotero_item(
        self,
        *,
        item_id: str,
        download_path: str,
        accept_types: list[str] | None = None,
        allow_fallback: bool = True,
    ) -> bool:
        """
        Download the best attachment of a Zotero item to the specified path.

        Parameters
        ----------
        item_id : str
            The specific item_id of the item to get the attachment from.

        download_path : str
            The folder to download the item to.
            File will be saved as <download_path>/<item_id>.<ext>.

        accept_types : list[str] | None, default=None
            List of acceptable file types in priority order.

            Supported types:
            - "pdf"  : PDF documents (application/pdf)
            - "epub" : EPUB ebooks (application/epub+zip)
            - "docx" : Word documents (modern format)
            - "doc"  : Word documents (legacy format)
            - "txt"  : Text files (text/plain)
            - "html" : HTML files (text/html)

            If None, defaults to FileType.ALL (all types, PDF preferred).

            Examples:
            - ["pdf"]            : Only PDFs, fail if not found (with allow_fallback=False)
            - ["epub", "pdf"]    : Try EPUB first, then PDF
            - FileType.EBOOKS    : Use pre-defined group (EPUB, PDF)
            - FileType.DOCUMENTS : Use pre-defined group (PDF, DOCX, DOC)

        allow_fallback : bool, default=True
            If True: accept other file types if no matching type found.
            If False: skip item if no matching type found.

        Returns
        -------
        bool
            True if the file was downloaded successfully.
            False if no matching attachment was found or no attachments exist.

        Notes
        -----
        - Local storage is checked first, then API fallback.
        - File extension is determined by actual content type.

        Examples
        --------
        >>> zotero = ZoteroSource()
        >>> zotero.connect(config)

        # Default: try all formats, prefer PDF
        >>> zotero.download_zotero_item(item_id="ABC123", download_path="./downloads/")
        True

        # Only PDFs, strict mode
        >>> zotero.download_zotero_item(
        ...     item_id="ABC123",
        ...     download_path="./downloads/",
        ...     accept_types=["pdf"],
        ...     allow_fallback=False,
        ... )
        True  # Only if PDF found

        # EPUB-first policy
        >>> zotero.download_zotero_item(
        ...     item_id="XYZ789",
        ...     download_path="./downloads/",
        ...     accept_types=FileType.EBOOKS,
        ... )
        True
        """
        z = self._get_zotero()

        logger.debug("Fetching file: {}", item_id)

        children = z.children(item_id)
        if not children:
            logger.warning("No child attachments found for item {}", item_id)
            return False

        attachments = [
            c for c in children if c.get("data", {}).get("itemType") == "attachment"
        ]
        if not attachments:
            logger.warning("No attachment-type children for item {}", item_id)
            return False

        attachment = self._select_best_attachment(
            attachments,
            accept_types=accept_types if accept_types is not None else FileType.ALL,
            allow_fallback=allow_fallback,
        )

        if not attachment:
            logger.warning("Could not find acceptable attachment for item {}", item_id)
            return False

        data = attachment.get("data", {})
        content_type = data.get("contentType", "application/pdf")
        ext = MIME_TO_EXT.get(content_type, ".pdf")

        download_dir = Path(download_path)
        download_dir.mkdir(parents=True, exist_ok=True)
        target = download_dir / f"{item_id}{ext}"

        local_path = data.get("path")
        if local_path and Path(local_path).exists():
            logger.info("Copying local attachment from {}", local_path)
            shutil.copy(local_path, target)
            return True

        logger.info("Local attachment not found, downloading via Zotero API")
        z.dump(
            itemkey=attachment["key"],
            filename=target.name,
            path=download_path,
        )

        return True

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
        z = self._get_zotero()
        config = self._get_config()

        items_iter = (
            z.collection_items_top(config.collection, limit=None)
            if config.collection else z.items()
        )
        items = list(z.everything(items_iter))

        # If no cursor → epoch
        if last_synced is None:
            last_synced = datetime(1970, 1, 1, tzinfo=timezone.utc)

        artefacts: list[tuple[str, datetime]] = []

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
        z = self._get_zotero()

        contents: list[Content] = []

        for item_key, modified in artefacts:
            item = z.item(item_key)
            if not item:
                continue
            contents.append(
                Content(
                    id_=item_key,
                    date=modified,
                    content=item.get("data", {}),
                )
            )

        return contents