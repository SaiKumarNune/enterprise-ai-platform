import math

import pytest

from app.services.embeddings import EMBEDDING_DIMENSIONS, EmbeddingError, validate_embeddings
from tests.fake_embeddings import HashEmbedder


def test_hash_embedder_returns_normalized_384d_vectors() -> None:
    embedder = HashEmbedder()
    vectors = embedder.embed_texts(["alpha", "beta"])

    assert embedder.batch_sizes == [2]
    assert len(vectors) == 2
    for vector in vectors:
        assert len(vector) == EMBEDDING_DIMENSIONS
        assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0, rel_tol=1e-6)


def test_validate_embeddings_rejects_wrong_dimension() -> None:
    with pytest.raises(EmbeddingError):
        validate_embeddings([[0.1, 0.2]], EMBEDDING_DIMENSIONS)


def test_validate_embeddings_rejects_non_finite_values() -> None:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[0] = float("nan")
    with pytest.raises(EmbeddingError):
        validate_embeddings([vector], EMBEDDING_DIMENSIONS)
