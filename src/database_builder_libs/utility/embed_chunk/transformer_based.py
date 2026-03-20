from __future__ import annotations

import torch
from typing import cast
from pydantic import PrivateAttr
from transformers import AutoModel, AutoTokenizer, BatchEncoding
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from database_builder_libs.models.chunk import Chunk
from database_builder_libs.models.abstract_chunk_embedder import AbstractChunkEmbedder


class TransformersChunkEmbedder(AbstractChunkEmbedder):
    """
    Chunk embedder backed by any HuggingFace Transformers model.

    Runs inference locally using ``transformers`` and ``torch``.  The vector
    for each chunk is produced by mean-pooling the last hidden state over the
    token dimension, respecting the attention mask.

    Attributes
    ----------
    model_name_or_path:
        HuggingFace model identifier or local path, e.g.
        ``"sentence-transformers/all-MiniLM-L6-v2"`` or ``"/models/bge-m3"``.
    device:
        Torch device to run inference on.  Defaults to ``"cuda"`` if available,
        otherwise ``"cpu"``.
    max_length:
        Maximum token length passed to the tokenizer.  Sequences longer than
        this are truncated.
    batch_size:
        Number of chunks to encode in a single forward pass.  Reduce if
        running into GPU memory errors.
    """

    model_name_or_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 512
    batch_size: int = 32

    _tokenizer: PreTrainedTokenizerBase = PrivateAttr()
    _model: PreTrainedModel = PrivateAttr()

    def model_post_init(self, __context: object) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self._model = AutoModel.from_pretrained(self.model_name_or_path)
        self._model.to(self.device)
        self._model.eval()

    def _mean_pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return torch.sum(last_hidden_state * mask, dim=1) / torch.clamp(
            mask.sum(dim=1), min=1e-9
        )

    def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []

        embedded: list[Chunk] = []

        for batch_start in range(0, len(chunks), self.batch_size):
            batch = chunks[batch_start : batch_start + self.batch_size]
            encoded: BatchEncoding = self._tokenizer(
                [c.text for c in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            attention_mask = cast(torch.Tensor, encoded["attention_mask"])

            with torch.no_grad():
                output = self._model(**encoded)

            vectors = self._mean_pool(
                output.last_hidden_state,
                attention_mask,
            ).cpu().tolist()

            embedded.extend(
                Chunk(
                    document_id=c.document_id,
                    chunk_index=c.chunk_index,
                    text=c.text,
                    vector=v,
                    metadata=c.metadata,
                )
                for c, v in zip(batch, vectors)
            )

        return embedded