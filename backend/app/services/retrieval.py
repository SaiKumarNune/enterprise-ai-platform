"""Authorized hybrid retrieval: lexical FTS, pgvector, and RRF fusion.

PostgreSQL ranking uses ts_rank_cd, which is not BM25. True BM25 would need a
separate extension or in-process scorer that cannot enforce ACL in SQL.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.services.authorization import (
    Principal,
    authorized_document_clause,
    authorized_document_ids,
    require_document,
)
from app.core.metrics import observe_retrieval
from app.services.rrf import DEFAULT_RRF_K, reciprocal_rank_fusion

SEMANTIC = "semantic"
LEXICAL = "lexical"
HYBRID = "hybrid"
RETRIEVAL_MODES = (SEMANTIC, LEXICAL, HYBRID)
RetrievalMode = Literal["semantic", "lexical", "hybrid"]

# Approximate PostgreSQL `english` stop words for in-memory tests only.
ENGLISH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "not",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?", re.IGNORECASE)


@dataclass(frozen=True)
class RetrievalHit:
    chunk: DocumentChunk
    score: float
    similarity: float
    mode: str


def normalize_retrieval_mode(value: str | None) -> str:
    mode = (value or settings.retrieval_mode).strip().lower()
    if mode not in RETRIEVAL_MODES:
        raise ValueError("retrieval_mode must be semantic, lexical, or hybrid")
    return mode


def lexical_tokens(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall(text.lower())
        if token not in ENGLISH_STOPWORDS
    ]


def retrieve_chunks(
    db: Session,
    *,
    query: str,
    principal: Principal,
    top_k: int,
    mode: str | None = None,
    query_embedding: list[float] | None = None,
    document_id: UUID | None = None,
    candidate_pool: int | None = None,
    rrf_k: int | None = None,
) -> list[RetrievalHit]:
    if document_id is not None:
        require_document(db, principal, document_id, "read")
    resolved = normalize_retrieval_mode(mode)
    pool = max(top_k, candidate_pool or settings.retrieval_candidate_pool)
    k = rrf_k or settings.rrf_k or DEFAULT_RRF_K
    started = time.perf_counter()
    outcome = "error"
    try:
        hits = _retrieve_chunks(
            db,
            query=query,
            principal=principal,
            top_k=top_k,
            resolved=resolved,
            query_embedding=query_embedding,
            document_id=document_id,
            pool=pool,
            k=k,
        )
        outcome = "empty" if not hits else "success"
        return hits
    finally:
        observe_retrieval(resolved, outcome, time.perf_counter() - started)


def _retrieve_chunks(
    db: Session,
    *,
    query: str,
    principal: Principal,
    top_k: int,
    resolved: str,
    query_embedding: list[float] | None,
    document_id: UUID | None,
    pool: int,
    k: int,
) -> list[RetrievalHit]:
    if resolved == SEMANTIC:
        rows = _semantic_rows(
            db,
            query_embedding=_require_embedding(query_embedding),
            top_k=top_k,
            principal=principal,
            document_id=document_id,
        )
        return [
            RetrievalHit(chunk=chunk, score=score, similarity=score, mode=SEMANTIC)
            for chunk, score in rows
        ]
    if resolved == LEXICAL:
        rows = _lexical_rows(db, query=query, top_k=top_k, principal=principal, document_id=document_id)
        return [
            RetrievalHit(chunk=chunk, score=score, similarity=_cosine_or_zero(chunk, query_embedding), mode=LEXICAL)
            for chunk, score in rows
        ]
    semantic_rows = []
    if query_embedding is not None:
        semantic_rows = _semantic_rows(
            db,
            query_embedding=query_embedding,
            top_k=pool,
            principal=principal,
            document_id=document_id,
        )
    lexical_rows = _lexical_rows(db, query=query, top_k=pool, principal=principal, document_id=document_id)
    fused = reciprocal_rank_fusion(
        [
            [chunk.id for chunk, _score in semantic_rows],
            [chunk.id for chunk, _score in lexical_rows],
        ],
        k=k,
    )
    by_id: dict[UUID, DocumentChunk] = {}
    cosine_by_id: dict[UUID, float] = {}
    for chunk, score in semantic_rows:
        by_id[chunk.id] = chunk
        cosine_by_id[chunk.id] = score
    for chunk, _score in lexical_rows:
        by_id.setdefault(chunk.id, chunk)
        cosine_by_id.setdefault(chunk.id, _cosine_or_zero(chunk, query_embedding))
    hits: list[RetrievalHit] = []
    for chunk_id, rrf_score in fused[:top_k]:
        chunk = by_id[chunk_id]
        hits.append(
            RetrievalHit(
                chunk=chunk,
                score=rrf_score,
                similarity=cosine_by_id.get(chunk_id, 0.0),
                mode=HYBRID,
            )
        )
    return hits


def _require_embedding(query_embedding: list[float] | None) -> list[float]:
    if query_embedding is None:
        raise ValueError("semantic retrieval requires a query embedding")
    return query_embedding


def _cosine_or_zero(chunk: DocumentChunk, query_embedding: list[float] | None) -> float:
    embedding = getattr(chunk, "embedding", None)
    if query_embedding is None or embedding is None:
        return 0.0
    return _cosine_similarity(query_embedding, list(embedding))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _allowed_chunks(
    db: Session,
    principal: Principal,
    document_id: UUID | None,
    *,
    require_embedding: bool,
) -> list[DocumentChunk]:
    allowed = authorized_document_ids(db, principal, "read")
    if document_id is not None:
        allowed = allowed & {document_id}
    rows = []
    for item in getattr(db, "chunks", []):
        if item.document_id not in allowed:
            continue
        if require_embedding and getattr(item, "embedding", None) is None:
            continue
        rows.append(item)
    return rows


def _semantic_rows(
    db: Session,
    *,
    query_embedding: list[float],
    top_k: int,
    principal: Principal,
    document_id: UUID | None,
) -> list[tuple[DocumentChunk, float]]:
    in_memory = getattr(db, "search_chunks", None)
    if callable(in_memory):
        allowed = authorized_document_ids(db, principal, "read")
        if document_id is not None:
            allowed = allowed & {document_id}
        return in_memory(query_embedding, top_k, document_id, allowed_document_ids=allowed)

    distance = DocumentChunk.embedding.cosine_distance(query_embedding)
    similarity = (1 - distance).label("similarity")
    statement = (
        select(DocumentChunk, similarity)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.embedding.is_not(None))
        .where(authorized_document_clause(principal, "read"))
        .order_by(distance.asc(), DocumentChunk.id.asc())
        .limit(top_k)
    )
    if document_id is not None:
        statement = statement.where(DocumentChunk.document_id == document_id)
    return [(row[0], float(row[1])) for row in db.execute(statement).all()]


def _lexical_rows(
    db: Session,
    *,
    query: str,
    top_k: int,
    principal: Principal,
    document_id: UUID | None,
) -> list[tuple[DocumentChunk, float]]:
    if hasattr(db, "chunks"):
        tokens = lexical_tokens(query)
        if not tokens:
            return []
        scored: list[tuple[DocumentChunk, float]] = []
        for chunk in _allowed_chunks(db, principal, document_id, require_embedding=False):
            score = _memory_lexical_score(chunk.content, tokens)
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: (-pair[1], str(pair[0].id)))
        return scored[:top_k]

    tsquery = func.plainto_tsquery("english", query)
    rank = func.ts_rank_cd(DocumentChunk.search_vector, tsquery)
    statement = (
        select(DocumentChunk, rank.label("lexical_rank"))
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(authorized_document_clause(principal, "read"))
        .where(func.numnode(tsquery) > 0)
        .where(DocumentChunk.search_vector.op("@@")(tsquery))
        .order_by(rank.desc(), DocumentChunk.id.asc())
        .limit(top_k)
    )
    if document_id is not None:
        statement = statement.where(DocumentChunk.document_id == document_id)
    return [(row[0], float(row[1])) for row in db.execute(statement).all()]


def _memory_lexical_score(content: str, query_tokens: list[str]) -> float:
    counts = Counter(lexical_tokens(content))
    score = 0.0
    matched = 0
    for token in query_tokens:
        freq = counts.get(token, 0)
        if freq:
            matched += 1
            score += freq / (freq + 1.0)
    if matched == 0:
        return 0.0
    return score + matched
