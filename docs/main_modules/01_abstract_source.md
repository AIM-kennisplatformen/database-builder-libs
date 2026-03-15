# Abstract source

## Overview
The **AbstractSource** class defines the contract that all concrete data‑source adapters must implement. It encapsulates the lifecycle of a synchronizable external system, providing a clear separation between connection handling, artefact discovery, and content retrieval. By adhering to this interface, different back‑ends (e.g., Zotero, SharePoint, REST APIs) can be swapped interchangeably while the rest of the pipeline remains agnostic to the underlying source.

## Design notes

### Interaction Pattern

The AbstractSource follows a three-phase interaction pattern:

1. **Connection Phase**: Establish connection to external system using backend-specific configuration
2. **Discovery Phase**: Query for artefacts modified since last synchronization timestamp
3. **Retrieval Phase**: Fetch normalized content for discovered artefacts

This design enables efficient incremental synchronization while maintaining consistency through stable identifiers and deterministic content serialization.

## Docstring
::: database_builder_libs.models.abstract_source
    handler: python

