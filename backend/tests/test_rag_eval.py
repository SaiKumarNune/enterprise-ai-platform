import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.run_rag_eval import DATASET, run_eval


def test_offline_rag_eval_reports_defined_metrics() -> None:
    report = run_eval(DATASET, top_k=5)

    assert report["mode"] == "offline_mock"
    assert report["aggregates"]["n_cases"] == 11
    assert set(report["aggregates"]) >= {
        "n_cases",
        "recall_at_k",
        "mrr",
        "citation_validity_rate",
        "citation_precision",
        "abstention_accuracy",
    }
    types = {item["case_type"] for item in report["cases"]}
    assert {"paraphrase", "ambiguous", "conflicting", "outdated", "missing_evidence", "prompt_injection"} <= types
    assert "scenario_breakdown" in report
    assert report["failures_by_scenario"] == {}
    assert report["passed"] is True
    recall = report["aggregates"]["recall_at_k"]
    citation_rate = report["aggregates"]["citation_validity_rate"]
    assert recall is None or 0.0 <= recall <= 1.0
    assert citation_rate is None or 0.0 <= citation_rate <= 1.0
    assert 0.0 <= report["aggregates"]["abstention_accuracy"] <= 1.0
    assert report["aggregates"]["mean_latency_ms"] >= 0
    assert "mrr" in report["aggregates"]
    assert "citation_precision" in report["aggregates"]
    assert report["embedder"] == "HashEmbedder"
    injection = next(item for item in report["cases"] if item["case_id"] == "prompt-injection-badge")
    assert injection["injection_blocked"] is True
    unanswerable = next(item for item in report["cases"] if item["case_id"] == "unanswerable-pension")
    assert unanswerable["expected_abstain"] is True


def test_eval_dataset_is_versioned() -> None:
    assert Path(DATASET).is_file()
    assert "rag_quality_v1" in DATASET.name
