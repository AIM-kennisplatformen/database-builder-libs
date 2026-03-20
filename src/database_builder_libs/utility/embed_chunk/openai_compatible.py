from __future__ import annotations

from openai import OpenAI
from pydantic import PrivateAttr

from database_builder_libs.models.chunk import Chunk

from database_builder_libs.models.abstract_chunk_embedder import AbstractChunkEmbedder


class OpenAICompatibleChunkEmbedder(AbstractChunkEmbedder):
    """
    Chunk embedder backed by any OpenAI-compatible ``/v1/embeddings`` endpoint.

    Works with OpenAI and any OpenAI-compatible server (e.g. Ollama, vLLM,
    LiteLLM) to embed chunks in a single batched request.

    Attributes
    ----------
    base_url:
        Base URL of the embeddings server, e.g. ``"http://localhost:11434/v1"``.
    api_key:
        API key passed to the underlying ``openai.OpenAI`` client.  Use any
        non-empty string for servers that do not enforce authentication.
    model:
        Model identifier forwarded to the ``/v1/embeddings`` endpoint.
    timeout:
        Per-request timeout in seconds.  ``None`` disables the timeout.
    """

    base_url: str
    api_key: str
    model: str = "qwen3-embedding:8b"
    timeout: float | None = 60.0

    _client: OpenAI = PrivateAttr()

    def model_post_init(self, __context: object) -> None:
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []
        res = self._client.embeddings.create(
            model=self.model,
            input=[c.text for c in chunks],
        )
        data = list(res.data)
        try:
            data.sort(key=lambda d: d.index)
        except Exception:
            pass
        if len(data) != len(chunks):
            raise RuntimeError(
                f"Embedding batch size mismatch: "
                f"got {len(data)} vectors for {len(chunks)} chunks."
            )
        return [
            Chunk(
                document_id=c.document_id,
                chunk_index=c.chunk_index,
                text=c.text,
                vector=list(d.embedding),
                metadata=c.metadata,
            )
            for c, d in zip(chunks, data)
        ]