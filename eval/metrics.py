"""Deterministic IR and citation metrics for evaluation runners."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from uuid import UUID


def precision_at_k(retrieved: Sequence[UUID], gold: Sequence[UUID]) -> float | None:
    if not retrieved:
        return 0.0 if gold else None
    return len(set(retrieved) & set(gold)) / len(retrieved)


def recall_at_k(retrieved: Sequence[UUID], gold: Sequence[UUID]) -> float | None:
    if not gold:
        return None
    return len(set(retrieved) & set(gold)) / len(gold)


def mrr(retrieved: Sequence[UUID], gold: Sequence[UUID]) -> float | None:
    if not gold:
        return None
    wanted = set(gold)
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in wanted:
            return 1.0 / rank
    return 0.0


def mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 4)


def citation_precision(
    cited_chunk_ids: Sequence[UUID],
    gold_chunk_ids: Sequence[UUID],
) -> float | None:
    if not cited_chunk_ids:
        return None if not gold_chunk_ids else 0.0
    if not gold_chunk_ids:
        return 0.0
    return len(set(cited_chunk_ids) & set(gold_chunk_ids)) / len(cited_chunk_ids)


def claim_coverage(text: str, claims: Sequence[str]) -> float | None:
    if not claims:
        return None
    haystack = _normalize(text)
    hits = sum(1 for claim in claims if _normalize(claim) in haystack)
    return hits / len(claims)


def _normalize(value: str) -> str:
    return " ".join(value.lower().split())
