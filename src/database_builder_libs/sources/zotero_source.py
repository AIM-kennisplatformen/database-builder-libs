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


# File type constants for convenience
class FileType:
    """Pre-defined file type groups for convenient reference"""
    
    # Single types
    PDF = ["pdf"]
    EPUB = ["epub"]
    DOCX = ["docx"]
    DOC = ["doc"]
    TXT = ["txt"]
    HTML = ["html"]
    
    # Type groups
    EBOOKS = ["epub", "pdf"]  # E-book formats (EPUB preferred)
    DOCUMENTS = ["pdf", "docx", "doc"]  # Document formats
    TEXT = ["txt", "html"]  # Plain text formats
    OFFICE = ["docx", "doc"]  # Office documents
    
    # All types in order of preference
    ALL = ["pdf", "epub", "docx", "doc", "txt", "html"]


# Mapping of file types to MIME types
FILE_TYPE_MIME_MAP = {
    "pdf": "application/pdf",
    "epub": "application/epub+zip",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "txt": "text/plain",
    "html": "text/html",
}


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

    def _select_best_attachment(self, attachments: list[dict]) -> Optional[dict]:
        """
        Select the best attachment for download from a list of attachments.
        
        Priority order:
        1. PDF attachments (contentType: application/pdf)
        2. EPUB attachments (contentType: application/epub+zip)
        3. Other document formats (.docx, .doc, .txt, .html)
        4. Any attachment (last resort)
        
        This ensures we get PDFs even if they're not the first attachment.
        Used as fallback when no preferred types match.
        
        Args:
            attachments: List of attachment dictionaries from Zotero API
        
        Returns:
            Best matching attachment dict, or None if no attachments available
        """
        if not attachments:
            return None
        
        # Priority 1: PDF files
        pdf_attachments = [
            a for a in attachments
            if a.get("data", {}).get("contentType") == "application/pdf"
        ]
        if pdf_attachments:
            logger.debug("Found {} PDF attachment(s), selecting first", len(pdf_attachments))
            return pdf_attachments[0]
        
        # Priority 2: EPUB files
        epub_attachments = [
            a for a in attachments
            if a.get("data", {}).get("contentType") == "application/epub+zip"
        ]
        if epub_attachments:
            logger.debug("No PDF found, selecting EPUB attachment")
            return epub_attachments[0]
        
        # Priority 3: Other document formats
        doc_types = {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
            "application/msword",  # .doc
            "text/plain",  # .txt
            "text/html",  # .html
        }
        doc_attachments = [
            a for a in attachments
            if a.get("data", {}).get("contentType") in doc_types
        ]
        if doc_attachments:
            logger.debug("No PDF/EPUB found, selecting document attachment")
            return doc_attachments[0]
        
        # Fallback: any attachment
        logger.debug("No standard format found, selecting any available attachment")
        return attachments[0]

    def _select_best_attachment_by_type(
        self,
        attachments: list[dict],
        accept_types: list[str],
        allow_fallback: bool
    ) -> Optional[dict]:
        """
        Select attachment matching preferred file types in priority order.
        
        Algorithm:
        1. For each file type in accept_types (in order):
           - Find all attachments matching that type
           - Return first match if found
        2. If no preferred types match and allow_fallback=True:
           - Return _select_best_attachment() (smart fallback)
        3. Otherwise:
           - Return None (no acceptable attachment found)
        
        Args:
            attachments: List of attachment dictionaries from Zotero API
            accept_types: List of acceptable file types in priority order
                         Valid types: "pdf", "epub", "docx", "doc", "txt", "html"
            allow_fallback: If True, use smart selection if no preferred types found
                           If False, only accept specified types
        
        Returns:
            Best matching attachment dict, or None if no match and no fallback
        """
        if not attachments:
            return None
        
        # Try each accepted type in priority order
        for file_type in accept_types:
            mime_type = FILE_TYPE_MIME_MAP.get(file_type)
            if not mime_type:
                logger.warning("Unknown file type: {}", file_type)
                continue
            
            # Find all attachments matching this type
            matching = [
                a for a in attachments
                if a.get("data", {}).get("contentType") == mime_type
            ]
            
            if matching:
                logger.debug(
                    "Found {} {} attachment(s), selecting first",
                    len(matching),
                    file_type
                )
                return matching[0]
        
        # No preferred types found
        if allow_fallback:
            logger.debug(
                "No preferred types found (wanted: {}), using smart fallback",
                ", ".join(accept_types)
            )
            return self._select_best_attachment(attachments)
        
        logger.warning(
            "No acceptable file types found (wanted: {})",
            ", ".join(accept_types)
        )
        return None

    def _get_file_extension(self, content_type: str) -> str:
        """
        Get file extension from MIME content type.
        
        Maps common MIME types to file extensions.
        
        Args:
            content_type: MIME type string (e.g., "application/pdf")
        
        Returns:
            File extension including dot (e.g., ".pdf")
        """
        mime_to_ext = {
            "application/pdf": ".pdf",
            "application/epub+zip": ".epub",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "application/msword": ".doc",
            "text/plain": ".txt",
            "text/html": ".html",
        }
        
        return mime_to_ext.get(content_type, ".pdf")  # Default to .pdf

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
        accept_types: list[str] | None = None,
        allow_fallback: bool = True,
    ) -> bool:
        """
        Download the best attachment of specified zotero item to specified path.
        
        Smart attachment selection based on file type preferences.
        
        Parameters
        ----------
        item_id : str
            The specific item_id of the item to get the attachment from
        
        download_path : str
            The folder to download the item to
            File will be saved as <download_path>/<item_id>.<ext>
        
        accept_types : list[str] | None, default=None
            List of acceptable file types in priority order.
            
            Supported types:
            - "pdf" : PDF documents (application/pdf)
            - "epub" : EPUB ebooks (application/epub+zip)
            - "docx" : Word documents (modern format)
            - "doc" : Word documents (legacy format)
            - "txt" : Text files (text/plain)
            - "html" : HTML files (text/html)
            
            If None, defaults to FileType.ALL (all types, PDF preferred)
            
            Examples:
            - ["pdf"] : Only PDFs, fail if not found (with allow_fallback=False)
            - ["epub", "pdf"] : Try EPUB first, then PDF
            - ["pdf", "epub"] : Try PDF first, then EPUB
            - FileType.EBOOKS : Use pre-defined group (EPUB, PDF)
            - FileType.DOCUMENTS : Use pre-defined group (PDF, DOCX, DOC)
        
        allow_fallback : bool, default=True
            If True: Accept other file types if no matching type found
            If False: Skip item if no matching type found
            
            Examples:
            - accept_types=["pdf"], allow_fallback=True:
              Tries PDF first, but accepts EPUB, DOCX, etc. if no PDF
            
            - accept_types=["pdf"], allow_fallback=False:
              Only accepts PDF, skips item if not found
        
        Returns
        -------
        bool
            True if file was downloaded successfully
            False if no matching attachment found (with allow_fallback=False)
                or no attachments at all
        
        Notes
        -----
        - Local storage is checked first, then API fallback
        - File extension is determined by actual content type
        - Detailed logging shows attachment selection process
        
        Examples
        --------
        >>> zotero = ZoteroSource()
        >>> zotero.connect(config)
        
        # Default: try all formats, prefer PDF
        >>> zotero.download_zotero_item(
        ...     item_id="ABC123",
        ...     download_path="./downloads/"
        ... )
        True
        
        # Only PDFs, strict mode
        >>> zotero.download_zotero_item(
        ...     item_id="ABC123",
        ...     download_path="./downloads/",
        ...     accept_types=["pdf"],
        ...     allow_fallback=False
        ... )
        True  # Only if PDF found
        
        # EPUB-first policy for ebook items
        >>> zotero.download_zotero_item(
        ...     item_id="XYZ789",
        ...     download_path="./downloads/",
        ...     accept_types=FileType.EBOOKS
        ... )
        True
        
        # Accept documents only (PDF, DOCX, DOC)
        >>> zotero.download_zotero_item(
        ...     item_id="DOC456",
        ...     download_path="./downloads/",
        ...     accept_types=FileType.DOCUMENTS,
        ...     allow_fallback=False
        ... )
        True  # Only if PDF, DOCX, or DOC found
        """
        self._ensure_connected()
        assert self._zotero is not None

        logger.debug("Fetching File: {}", item_id)

        children = self._zotero.children(item_id)
        if not children:
            logger.warning("No child attachments found for item {}", item_id)
            return False

        attachments = [
            c for c in children if c.get("data", {}).get("itemType") == "attachment"
        ]
        if not attachments:
            logger.warning("No attachment-type children for item {}", item_id)
            return False

        # Default accept_types to all if not specified
        if accept_types is None:
            accept_types = FileType.ALL

        # Select best attachment based on file type preferences
        attachment = self._select_best_attachment_by_type(
            attachments,
            accept_types,
            allow_fallback
        )
        
        if not attachment:
            logger.warning("Could not find acceptable attachment for item {}", item_id)
            return False

        data = attachment.get("data", {})
        local_path = data.get("path")
        
        # Get correct file extension based on content type
        content_type = data.get("contentType", "application/pdf")
        ext = self._get_file_extension(content_type)
        
        download_dir = Path(download_path)
        download_dir.mkdir(parents=True, exist_ok=True)
        target = download_dir / f"{item_id}{ext}"
        
        if local_path and Path(local_path).exists():
            logger.info("Copying local attachment from {}", local_path)
            shutil.copy(local_path, target)
            return True

        logger.info("Local attachment not found, downloading via Zotero API")

        self._zotero.dump(
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
        self._ensure_connected()
        assert self._zotero is not None
        assert self._config is not None

        if self._config.collection:
            items_iter = self._zotero.collection_items_top(
                self._config.collection, limit=None
            )
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