from __future__ import annotations

import math
from typing import Protocol

from app.core.config import settings

EMBEDDING_DIMENSIONS = 384


class EmbeddingError(Exception):
    """Embedding generation produced invalid vectors."""


class EmbeddingBackend(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one normalized embedding per input string."""


class SentenceTransformerEmbedder:
    """Lazy local encoder. The model may download on first real use."""

    def __init__(self, model_name: str, dimensions: int) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self._model = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load().encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        packed = [vector.tolist() for vector in vectors]
        return validate_embeddings(packed, self.dimensions)

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model


_embedder: SentenceTransformerEmbedder | None = None


def get_embedder() -> EmbeddingBackend:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformerEmbedder(
            model_name=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    return _embedder


def validate_embeddings(
    vectors: list[list[float]],
    dimensions: int,
) -> list[list[float]]:
    validated: list[list[float]] = []
    for vector in vectors:
        if len(vector) != dimensions:
            raise EmbeddingError("Embedding dimension mismatch")
        if any(not math.isfinite(value) for value in vector):
            raise EmbeddingError("Embedding contains non-finite values")
        validated.append(_l2_normalize(vector))
    return validated


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise EmbeddingError("Embedding vector has zero magnitude")
    return [value / norm for value in vector]
