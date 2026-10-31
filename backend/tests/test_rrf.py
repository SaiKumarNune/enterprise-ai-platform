from uuid import UUID

import pytest

from app.services.rrf import reciprocal_rank_fusion


def test_rrf_adds_reciprocal_ranks() -> None:
    first = UUID("11111111-1111-4111-8111-111111111111")
    second = UUID("22222222-2222-4222-8222-222222222222")
    fused = reciprocal_rank_fusion([[first, second], [second, first]], k=60)
    scores = {chunk_id: score for chunk_id, score in fused}
    assert scores[first] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[second] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0][0] == first


def test_rrf_deduplicates_within_a_list() -> None:
    chunk = UUID("11111111-1111-4111-8111-111111111111")
    fused = reciprocal_rank_fusion([[chunk, chunk]], k=60)
    assert len(fused) == 1
    assert fused[0][0] == chunk
    assert fused[0][1] == pytest.approx(1 / 61)


def test_rrf_tie_breaks_by_uuid_string() -> None:
    later = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
    earlier = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    fused = reciprocal_rank_fusion([[later], [earlier]], k=60)
    assert [chunk_id for chunk_id, _score in fused] == [earlier, later]


def test_rrf_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be"):
        reciprocal_rank_fusion([[]], k=0)
