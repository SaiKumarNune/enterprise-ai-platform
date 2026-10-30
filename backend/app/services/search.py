from __future__ import annotations

import math
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chunk import DocumentChunk


def search_similar_chunks(
    db: Session,
    query_embedding: list[float],
    *,
    top_k: int,
    document_id: UUID | None = None,
) -> list[tuple[DocumentChunk, float]]:
    in_memory = getattr(db, "search_chunks", None)
    if callable(in_memory):
        return in_memory(query_embedding, top_k, document_id)

    distance = DocumentChunk.embedding.cosine_distance(query_embedding)
    similarity = (1 - distance).label("similarity")
    statement = (
        select(DocumentChunk, similarity)
        .where(DocumentChunk.embedding.is_not(None))
        .order_by(distance)
        .limit(top_k)
    )
    if document_id is not None:
        statement = statement.where(DocumentChunk.document_id == document_id)
    return [(row[0], float(row[1])) for row in db.execute(statement).all()]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
