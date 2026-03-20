# Qdrant store

## Overview
`QdrantDatastore` is a concrete `AbstractVectorStore` implementation backed by the Qdrant vector database. It stores document `Chunk` objects, each identified by a deterministic hash‑derived point ID, and provides fast cosine‑similarity search.

## Design notes

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
