# OpenAI-compatible chunk embedder

## Overview

The `OpenAICompatibleChunkEmbedder` embeds document chunks using any OpenAI-compatible `/v1/embeddings` endpoint. It works with OpenAI directly as well as self-hosted servers such as Ollama, vLLM, and LiteLLM, making it suitable for both cloud and air-gapped deployments.

### Configuration

Configuration happens through Pydantic fields passed at construction:

```python
embedder = OpenAICompatibleChunkEmbedder(
    base_url="http://localhost:11434/v1",
    api_key="ollama",           # Any non-empty string for servers without auth
    model="qwen3-embedding:8b", # Default
    timeout=60.0,               # Optional, default 60 seconds
)
```

## Usage and Practical Examples

### Basic Embedding

```python
from database_builder_libs.models.chunk import Chunk
from database_builder_libs.utility.embed_chunk.openai_compatible import OpenAICompatibleChunkEmbedder

embedder = OpenAICompatibleChunkEmbedder(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

chunks = [
    Chunk(document_id="doc-1", chunk_index=0, text="Hello world", vector=[], metadata={}),
    Chunk(document_id="doc-1", chunk_index=1, text="Second chunk", vector=[], metadata={}),
]

embedded = embedder.embed(chunks)
# Each chunk now has its vector field populated
print(embedded[0].vector)  # [0.023, -0.117, ...]
```

### Using with OpenAI

```python
embedder = OpenAICompatibleChunkEmbedder(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model="text-embedding-3-small",
)
```

### Using with vLLM

```python
embedder = OpenAICompatibleChunkEmbedder(
    base_url="http://my-vllm-server:8000/v1",
    api_key="token",
    model="BAAI/bge-m3",
    timeout=120.0,  # Increase for large batches on slower hardware
)
```

### Empty Input

```python
# Empty input is handled gracefully without calling the API
result = embedder.embed([])
assert result == []
```

## Behavior and Edge Cases

### Response Ordering

The embedder sorts the API response by the `index` field before reconstructing chunks. This guards against servers that return embeddings out of order:

```python
# Safe regardless of server response ordering
embedded = embedder.embed(chunks)
assert embedded[0].chunk_index == chunks[0].chunk_index
```

### Batch Size Mismatch

If the server returns a different number of vectors than chunks submitted, a `RuntimeError` is raised before any chunks are reconstructed:

```python
# Raises RuntimeError: Embedding batch size mismatch: got 2 vectors for 3 chunks.
embedded = embedder.embed(three_chunks)
```

### Timeout Configuration

```python
# Disable timeout entirely for very large batches
embedder = OpenAICompatibleChunkEmbedder(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=None,
)
```

## Docstring

::: database_builder_libs.utility.embed_chunk.openai_compatible.OpenAICompatibleChunkEmbedder
    handler: python