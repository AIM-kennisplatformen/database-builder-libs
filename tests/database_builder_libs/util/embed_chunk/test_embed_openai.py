from unittest.mock import MagicMock, patch

import pytest

from database_builder_libs.models.chunk import Chunk
from database_builder_libs.utility.embed_chunk.openai_compatible import (
    OpenAICompatibleChunkEmbedder,
)

_PATCH_OPENAI = "database_builder_libs.utility.embed_chunk.openai_compatible.OpenAI"
_BASE_URL = "http://localhost:11434/v1"
_MODEL = "qwen3-embedding:8b"


def make_chunk(document_id: str = "doc-1", chunk_index: int = 0, text: str = "hello") -> Chunk:
    return Chunk(document_id=document_id, chunk_index=chunk_index, text=text, vector=[], metadata={})


def make_embedding_data(index: int, vector: list[float]) -> MagicMock:
    item = MagicMock()
    item.index = index
    item.embedding = vector
    return item


def _stub_embedder(embedder: OpenAICompatibleChunkEmbedder, vectors: list[list[float]]) -> None:
    """Wire embedder._client so embeddings.create() returns *vectors* in order."""
    fake_response = MagicMock()
    fake_response.data = [make_embedding_data(i, v) for i, v in enumerate(vectors)]
    embedder._client.embeddings.create.return_value = fake_response


@pytest.fixture
def mock_openai():
    with patch(_PATCH_OPENAI):
        yield


@pytest.fixture
def fresh_embedder(mock_openai):
    """A freshly constructed OpenAICompatibleChunkEmbedder with real model_post_init."""
    return OpenAICompatibleChunkEmbedder(base_url=_BASE_URL, api_key="key", model=_MODEL)


@pytest.fixture
def embedder(mock_openai):
    """An OpenAICompatibleChunkEmbedder with the client replaced by a plain MagicMock."""
    instance = OpenAICompatibleChunkEmbedder(base_url=_BASE_URL, api_key="test", model=_MODEL)
    instance._client = MagicMock()
    return instance


def test_default_model_is_set(fresh_embedder):
    """The default model identifier is applied when none is provided."""
    assert fresh_embedder.model == _MODEL


def test_custom_model_is_preserved(mock_openai):
    """A custom model identifier passed at construction is preserved on the instance."""
    instance = OpenAICompatibleChunkEmbedder(
        base_url=_BASE_URL, api_key="key", model="text-embedding-3-small"
    )
    assert instance.model == "text-embedding-3-small"


def test_default_timeout_is_set(fresh_embedder):
    """The default timeout of 60.0 seconds is applied when none is provided."""
    assert fresh_embedder.timeout == 60.0


def test_embed_empty_list_returns_empty(embedder):
    """An empty input list returns an empty list without calling the API."""
    result = embedder.embed([])

    assert result == []
    embedder._client.embeddings.create.assert_not_called()


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


def test_embed_populates_vectors(embedder):
    """Each returned chunk has its vector field populated with the API response."""
    chunks = [make_chunk(text="embed me")]
    expected_vector = [0.1, 0.2, 0.3]
    _stub_embedder(embedder, [expected_vector])

    result = embedder.embed(chunks)

    assert result[0].vector == expected_vector


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


def test_embed_passes_correct_texts_to_api(embedder):
    """The texts extracted from chunks are forwarded to the embeddings API."""
    chunks = [make_chunk(text="alpha"), make_chunk(chunk_index=1, text="beta")]
    _stub_embedder(embedder, [[0.1], [0.2]])

    embedder.embed(chunks)

    assert embedder._client.embeddings.create.call_args.kwargs["input"] == ["alpha", "beta"]


def test_embed_passes_correct_model_to_api(embedder):
    """The configured model identifier is forwarded to the embeddings API."""
    chunks = [make_chunk()]
    _stub_embedder(embedder, [[0.1]])

    embedder.embed(chunks)

    assert embedder._client.embeddings.create.call_args.kwargs["model"] == _MODEL


def test_embed_batch_of_multiple_chunks(embedder):
    """All chunks in a batch are embedded and returned correctly."""
    chunks = [make_chunk(chunk_index=i, text=f"chunk {i}") for i in range(5)]
    _stub_embedder(embedder, [[float(i)] * 3 for i in range(5)])

    assert len(embedder.embed(chunks)) == 5


def test_embed_sorts_response_by_index(embedder):
    """Out-of-order API responses are sorted by index before chunk reconstruction."""
    chunks = [make_chunk(chunk_index=i, text=f"text {i}") for i in range(3)]

    fake_response = MagicMock()
    fake_response.data = [
        make_embedding_data(2, [0.2]),
        make_embedding_data(0, [0.0]),
        make_embedding_data(1, [0.1]),
    ]
    embedder._client.embeddings.create.return_value = fake_response

    result = embedder.embed(chunks)

    assert result[0].vector == [0.0]
    assert result[1].vector == [0.1]
    assert result[2].vector == [0.2]


def test_embed_raises_on_batch_size_mismatch(embedder):
    """A RuntimeError is raised when the API returns fewer vectors than input chunks."""
    chunks = [make_chunk(chunk_index=i) for i in range(3)]
    _stub_embedder(embedder, [[0.1], [0.2]])

    with pytest.raises(RuntimeError, match="mismatch"):
        embedder.embed(chunks)