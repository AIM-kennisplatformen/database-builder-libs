# Abstract chunk embedder

## Overview

The `AbstractChunkEmbedder` defines the interface for embedding systems that operate on document chunks. It transforms a list of `Chunk` objects with empty vectors into a list of `Chunk` objects with their `vector` fields populated, ready for indexing into a vector store.

## Design Notes

### Interaction Patterns

The `AbstractChunkEmbedder` supports a single, focused interaction pattern:

1. **Embedding Pattern**:
   - Accept an ordered list of `Chunk` objects with empty vectors
   - Populate each `Chunk.vector` with a dense float embedding
   - Return chunks in the same order with identity fields preserved

### Implementation Requirements

* **Ordering Consistency**: Returned chunks must preserve the original input order — index *i* in the output must correspond to index *i* in the input
* **Identity Preservation**: `document_id`, `chunk_index`, `text`, and `metadata` must be passed through unchanged
* **Length Invariant**: The returned list must always have the same length as the input list
* **Empty Input**: An empty input list must return an empty list without error or side effects
* **Non-empty Vectors**: Every returned `Chunk.vector` must be a non-empty float sequence

### Pydantic Integration

`AbstractChunkEmbedder` inherits from `BaseModel` rather than plain `ABC`. This enforces that all implementations are Pydantic models, enabling:

- Declarative field definitions with validation
- `PrivateAttr` for non-serializable runtime objects (e.g. HTTP clients, loaded models)
- `model_post_init` as the standard hook for post-construction initialization

## Docstring abstract chunk embedder

::: database_builder_libs.models.abstract_chunk_embedder
    handler: python