"""Reciprocal Rank Fusion over ranked identifier lists."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[UUID]],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[tuple[UUID, float]]:
    """Fuse rank positions: score = sum 1/(k + rank) with 1-based ranks.

    Duplicate IDs inside one list are ignored after the first occurrence.
    Ties break by UUID string so the order is deterministic.
    """
    if k < 1:
        raise ValueError("RRF k must be >= 1")
    scores: dict[UUID, float] = {}
    for ranked in ranked_lists:
        seen: set[UUID] = set()
        for rank, chunk_id in enumerate(ranked, start=1):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))
