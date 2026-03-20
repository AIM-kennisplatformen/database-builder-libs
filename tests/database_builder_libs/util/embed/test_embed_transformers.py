from unittest.mock import MagicMock, patch

import pytest
import torch

from database_builder_libs.models.chunk import Chunk
from database_builder_libs.utility.embed_chunk.transformer_based import (
    TransformersChunkEmbedder,
)

_PATCH_TOKENIZER = "database_builder_libs.utility.embed_chunk.transformer_based.AutoTokenizer"
_PATCH_MODEL = "database_builder_libs.utility.embed_chunk.transformer_based.AutoModel"
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(document_id: str = "doc-1", chunk_index: int = 0, text: str = "hello") -> Chunk:
    return Chunk(document_id=document_id, chunk_index=chunk_index, text=text, vector=[], metadata={})


def _stub_embedder(embedder: TransformersChunkEmbedder, vectors: list[list[float]]) -> None:
    """Wire embedder._model so forward() returns hidden states that mean-pool to *vectors*."""
    tensor = torch.tensor(vectors).unsqueeze(1)  # (batch, seq_len=1, hidden)
    fake_output = MagicMock()
    fake_output.last_hidden_state = tensor
    embedder._model.return_value = fake_output

    fake_encoded = {"attention_mask": torch.ones(len(vectors), 1, dtype=torch.long)}
    embedder._tokenizer.return_value = MagicMock(
        **fake_encoded,
        to=lambda device: fake_encoded,
    )


@pytest.fixture
def mock_auto():
    """Patch AutoTokenizer and AutoModel, exposing the AutoModel mock for inspection."""
    with (
        patch(_PATCH_TOKENIZER),
        patch(_PATCH_MODEL) as mock_model,
    ):
        yield mock_model


@pytest.fixture
def fresh_embedder(mock_auto):
    """A freshly constructed TransformersChunkEmbedder with real model_post_init."""
    return TransformersChunkEmbedder(model_name_or_path=_MODEL_NAME, device="cpu")


@pytest.fixture
def embedder(mock_auto):
    """A TransformersChunkEmbedder with tokenizer and model replaced by plain MagicMocks."""
    instance = TransformersChunkEmbedder(model_name_or_path=_MODEL_NAME, device="cpu")
    instance._tokenizer = MagicMock()
    instance._model = MagicMock()
    return instance


def test_default_batch_size_is_set(fresh_embedder):
    """The default batch size of 32 is applied when none is provided."""
    assert fresh_embedder.batch_size == 32


def test_default_max_length_is_set(fresh_embedder):
    """The default max token length of 512 is applied when none is provided."""
    assert fresh_embedder.max_length == 512


def test_model_set_to_eval_on_init(mock_auto):
    """The underlying model is set to eval mode during initialisation."""
    TransformersChunkEmbedder(model_name_or_path=_MODEL_NAME, device="cpu")
    mock_auto.from_pretrained.return_value.eval.assert_called_once()



def test_embed_empty_list_returns_empty(embedder):
    """An empty input list returns an empty list without calling the model."""
    result = embedder.embed([])

    assert result == []
    embedder._model.assert_not_called()


def test_embed_returns_chunks(embedder):
    """A successful embed call returns Chunk instances."""
    chunks = [make_chunk(text="hello")]
    _stub_embedder(embedder, [[0.1, 0.2, 0.3]])

    result = embedder.embed(chunks)

    assert len(result) == 1
    assert isinstance(result[0], Chunk)


def test_embed_preserves_ordering(embedder):
    """Returned chunks preserve the original input ordering."""
    chunks = [make_chunk(chunk_index=i, text=f"text {i}") for i in range(3)]
    _stub_embedder(embedder, [[float(i)] * 4 for i in range(3)])

    result = embedder.embed(chunks)

    for i, chunk in enumerate(result):
        assert chunk.chunk_index == i


def test_embed_preserves_identity_fields(embedder):
    """document_id, chunk_index, text, and metadata are preserved on returned chunks."""
    chunk = Chunk(
        document_id="doc-99",
        chunk_index=7,
        text="preserve me",
        vector=[],
        metadata={"key": "value"},
    )
    _stub_embedder(embedder, [[0.5, 0.6]])

    result = embedder.embed([chunk])

    assert result[0].document_id == "doc-99"
    assert result[0].chunk_index == 7
    assert result[0].text == "preserve me"
    assert result[0].metadata == {"key": "value"}


def test_embed_populates_vectors(embedder):
    """Each returned chunk has its vector field populated from the model output."""
    chunks = [make_chunk(text="embed me")]
    _stub_embedder(embedder, [[0.1, 0.2, 0.3]])

    result = embedder.embed(chunks)

    assert len(result[0].vector) == 3


def test_embed_passes_correct_texts_to_tokenizer(embedder):
    """The texts extracted from chunks are forwarded to the tokenizer."""
    chunks = [make_chunk(text="alpha"), make_chunk(chunk_index=1, text="beta")]
    _stub_embedder(embedder, [[0.1], [0.2]])

    embedder.embed(chunks)

    call_args = embedder._tokenizer.call_args
    assert "alpha" in call_args[0][0]
    assert "beta" in call_args[0][0]


def test_embed_passes_correct_max_length_to_tokenizer(embedder):
    """The configured max_length is forwarded to the tokenizer."""
    chunks = [make_chunk()]
    _stub_embedder(embedder, [[0.1]])

    embedder.embed(chunks)

    assert embedder._tokenizer.call_args.kwargs["max_length"] == 512


def test_embed_processes_all_chunks_across_batches(embedder):
    """All chunks are embedded correctly when the input exceeds batch_size."""
    embedder.batch_size = 2
    chunks = [make_chunk(chunk_index=i, text=f"chunk {i}") for i in range(5)]
    _stub_embedder(embedder, [[float(i)] * 3 for i in range(5)])

    result = embedder.embed(chunks)

    assert len(result) == 5


def test_embed_calls_model_once_per_batch(embedder):
    """The model forward pass is called once per batch, not once per chunk."""
    embedder.batch_size = 2
    chunks = [make_chunk(chunk_index=i, text=f"chunk {i}") for i in range(4)]
    _stub_embedder(embedder, [[0.1] * 2 for _ in range(4)])

    embedder.embed(chunks)

    assert embedder._model.call_count == 2


def test_embed_single_batch_calls_model_once(embedder):
    """When the input fits in a single batch the model is called exactly once."""
    chunks = [make_chunk(chunk_index=i) for i in range(3)]
    _stub_embedder(embedder, [[0.1] for _ in range(3)])

    embedder.embed(chunks)

    assert embedder._model.call_count == 1