# Transformer-based chunk embedder

## Overview

The `TransformersChunkEmbedder` embeds document chunks locally using any HuggingFace Transformers model. It runs inference directly on the host machine via `transformers` and `torch`, making it suitable for air-gapped environments or deployments where sending data to an external API is not acceptable.

Vectors are produced by mean-pooling the last hidden state over the token dimension, weighted by the attention mask.

### Configuration

Configuration happens through Pydantic fields passed at construction:

```python
embedder = TransformersChunkEmbedder(
    model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda",      # Optional, defaults to "cuda" if available else "cpu"
    max_length=512,     # Optional, default 512 tokens
    batch_size=32,      # Optional, default 32 chunks per forward pass
)
```

## Usage and Practical Examples

### Basic Embedding

```python
from database_builder_libs.models.chunk import Chunk
from database_builder_libs.utility.embed_chunk.transformer_based import TransformersChunkEmbedder

embedder = TransformersChunkEmbedder(
    model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
)

chunks = [
    Chunk(document_id="doc-1", chunk_index=0, text="Hello world", vector=[], metadata={}),
    Chunk(document_id="doc-1", chunk_index=1, text="Second chunk", vector=[], metadata={}),
]

embedded = embedder.embed(chunks)
# Each chunk now has its vector field populated
print(embedded[0].vector)  # [0.023, -0.117, ...]
```

### Using a Local Model

```python
embedder = TransformersChunkEmbedder(
    model_name_or_path="/models/bge-m3",
    device="cuda",
)
```

### Using with CPU Only

```python
embedder = TransformersChunkEmbedder(
    model_name_or_path="BAAI/bge-small-en-v1.5",
    device="cpu",
    batch_size=8,  # Reduce batch size on CPU to avoid memory pressure
)
```

### Empty Input

```python
# Empty input is handled gracefully without running any inference
result = embedder.embed([])
assert result == []
```

## Behavior and Edge Cases

### Batching

The embedder processes chunks in batches of `batch_size`, running one forward pass per batch. Reduce `batch_size` if running into GPU out-of-memory errors on large documents:

```python
# 5 chunks with batch_size=2 produces 3 forward passes
embedder = TransformersChunkEmbedder(
    model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda",
    batch_size=2,
)
embedded = embedder.embed(chunks)  # 3 forward passes
assert len(embedded) == 5
```

### Truncation

Sequences longer than `max_length` tokens are silently truncated by the tokenizer. Increase `max_length` for models that support longer contexts:

```python
embedder = TransformersChunkEmbedder(
    model_name_or_path="jinaai/jina-embeddings-v2-base-en",
    max_length=8192,
)
```

### Device Selection

The `device` field defaults to `"cuda"` if a GPU is available, otherwise `"cpu"`. Override explicitly for multi-GPU setups or to force CPU inference:

```python
# Force a specific GPU
embedder = TransformersChunkEmbedder(
    model_name_or_path="BAAI/bge-m3",
    device="cuda:1",
)
```

### Mean Pooling

The embedding for each chunk is computed by mean-pooling the last hidden state across the token dimension, weighted by the attention mask. Padding tokens are excluded from the average:

```python
# Longer and shorter chunks in the same batch are handled correctly —
# padding tokens do not contribute to the embedding
embedded = embedder.embed([short_chunk, long_chunk])
```

## Docstring

::: database_builder_libs.utility.embed_chunk.transformer_based
    handler: python