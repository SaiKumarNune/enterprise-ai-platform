from __future__ import annotations

import math
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.chunk import DocumentChunk
from app.services.authorization import Principal
from app.services.retrieval import SEMANTIC, retrieve_chunks


def search_similar_chunks(
    db: Session,
    query_embedding: list[float],
    *,
    top_k: int,
    principal: Principal,
    document_id: UUID | None = None,
) -> list[tuple[DocumentChunk, float]]:
    """Semantic pgvector retrieval. Prefer retrieve_chunks for hybrid/lexical modes."""
    hits = retrieve_chunks(
        db,
        query="",
        principal=principal,
        top_k=top_k,
        mode=SEMANTIC,
        query_embedding=query_embedding,
        document_id=document_id,
    )
    return [(hit.chunk, hit.similarity) for hit in hits]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
