from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.run_retrieval_eval import DATASET, run_eval


def test_synthetic_retrieval_eval_compares_three_modes() -> None:
    report = run_eval(DATASET, top_k=3)
    assert report["mode"] == "offline_mock_hash_embedder"
    for mode in ("semantic", "lexical", "hybrid"):
        stats = report["aggregates"][mode]
        assert 0.0 <= stats["precision_at_k"] <= 1.0
        assert 0.0 <= stats["recall_at_k"] <= 1.0
        assert 0.0 <= stats["mrr"] <= 1.0
        assert stats["mean_latency_ms"] >= 0
    identifier = next(
        item for item in report["cases"]["lexical"] if item["case_id"] == "rare-identifier"
    )
    assert identifier["recall_at_k"] == 1.0
    limitations = report["limitations"]
    blob = limitations if isinstance(limitations, str) else " ".join(limitations)
    assert "HashEmbedder" in blob
    assert "not BM25" in blob
