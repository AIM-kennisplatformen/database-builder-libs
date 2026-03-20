# Zotero source

## Overview

The Zotero source allows you to retrieve documents and metadata from a Zotero database using its API. It implements the AbstractSource interface to provide incremental synchronization of Zotero library items.

## Design Notes

### Interaction Patterns

The ZoteroSource follows these interaction patterns:

1. **Connection Pattern**:
    - Initialize with library credentials
    - Create pyzotero client instance
    - Optional collection filtering

2. **Synchronization Pattern**:
    - Query items modified since last sync
    - Convert Zotero timestamps to UTC
    - Return stable item keys

3. **Content Retrieval Pattern**:
    - Fetch full item metadata
    - Normalize to Content objects
    - Preserve Zotero data structure

4. **Attachment Download Pattern**:
    - Check for local file availability first
    - Fall back to API download if needed
    - Save as `{item_id}.pdf`

### Implementation Details

* **Timestamp Handling**: All timestamps converted to UTC for consistency
* **Deletion Limitation**: Zotero API doesn't report deleted items in sync
* **Attachment Priority**: Prefers local Zotero storage over API downloads for performance
* **Error Handling**: Gracefully handles missing attachments and continues processing

## Docstring
::: database_builder_libs.sources.zotero_source
    handler: python