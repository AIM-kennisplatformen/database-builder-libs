# Qdrant store

## Overview
`QdrantDatastore` is a concrete `AbstractVectorStore` implementation backed by the Qdrant vector database. It stores document `Chunk` objects, each identified by a deterministic hash‑derived point ID, and provides fast cosine‑similarity search.

## Design notes

### Implementation Patterns

1. **Identity Management**:
    - Blake2b hash of `document_id:chunk_index`
    - 64-bit unsigned integer point IDs
    - Enables idempotent upserts

2. **Vector Storage**:
    - Validates embedding dimensionality
    - Skips chunks without vectors
    - Stores text and metadata in payload

3. **Retrieval Patterns**:
    - Similarity search returns chunks without embeddings
    - Document retrieval uses scrolling for large results
    - Results sorted by chunk_index for reconstruction

4. **GDPR Compliance**:
    - Complete document deletion
    - Verification of deletion status
    - Returns count of deleted chunks

### Configuration Example

```python
config = {
    "url": "http://localhost:6333",
    "collection": "knowledge_base",
    "vector_size": 768  # Must match embedding model output
}
```

## Docstring
::: database_builder_libs.stores.qdrant.qdrant_store
    handler: python
