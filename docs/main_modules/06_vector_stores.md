# Abstract vector store

## Overview

The AbstractVectorStore defines the interface for semantic retrieval storage systems. It manages document chunks with their embeddings and enables nearest-neighbor similarity search across various vector database backends (FAISS, Qdrant, Pinecone, etc.).

## Design Notes

### Interaction Patterns

The AbstractVectorStore supports three main interaction patterns:

1. **Indexing Pattern**:
    - Store chunks with embeddings
    - Overwrite existing chunks with same identity
    - Validate embedding dimensionality

2. **Semantic Search Pattern**:
    - Accept query vector
    - Find k-nearest neighbors by similarity
    - Return chunks ordered by relevance (without embeddings)

3. **Document Management Pattern**:
    - Retrieve all chunks for a document
    - Delete all vectors for GDPR compliance
    - Maintain document-level consistency

### Implementation Requirements

* **Embedding Consistency**: All vectors in an index must use the same model and dimensionality
* **Distance Metric**: Implementations must specify and validate the distance metric (cosine, euclidean, etc.)
* **Batch Operations**: Support efficient batch insertion for large documents
* **Memory Safety**: Never return stored embeddings in search results to prevent memory issues

## Docstring abstract chunk

::: database_builder_libs.models.chunk
    handler: python

## Docstring abstract store
::: database_builder_libs.models.abstract_vector_store
    handler: python
