from __future__ import annotations

import math
import zlib

from app.services.embeddings import EMBEDDING_DIMENSIONS, validate_embeddings


class HashEmbedder:
    """Deterministic 384-d fake embedder. Does not load ML models."""

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        self.dimensions = dimensions
        self.batch_sizes: list[int] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        vectors = [_token_vector(text, self.dimensions) for text in texts]
        return validate_embeddings(vectors, self.dimensions)


class FailingEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding backend unavailable")


def _token_vector(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    tokens = text.lower().split() or [text]
    for token in tokens:
        index = zlib.crc32(token.encode("utf-8")) % dimensions
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]
