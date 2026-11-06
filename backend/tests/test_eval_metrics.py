from uuid import uuid4

from eval.classify import classify_rag_failure, failures_by_scenario, scenario_breakdown
from eval.metrics import claim_coverage, citation_precision, mean, mrr, precision_at_k, recall_at_k
from eval.reporting import compare_baseline, markdown_report


def test_ir_metrics() -> None:
    gold_a, gold_b, other = uuid4(), uuid4(), uuid4()
    retrieved = [other, gold_a]
    gold = [gold_a, gold_b]
    assert recall_at_k(retrieved, gold) == 0.5
    assert precision_at_k(retrieved, gold) == 0.5
    assert mrr(retrieved, gold) == 0.5
    assert recall_at_k([], []) is None
    assert mrr([other], gold) == 0.0


def test_citation_and_claim_metrics() -> None:
    gold = [uuid4()]
    cited = [gold[0], uuid4()]
    assert citation_precision(cited, gold) == 0.5
    assert claim_coverage("Employees receive 15 vacation days each year", ["15 vacation days"]) == 1.0
    assert claim_coverage("unrelated", ["15 vacation days"]) == 0.0
    assert mean([1.0, None, 0.0]) == 0.5


def test_baseline_ignores_llm_judge() -> None:
    current = {"aggregates": {"recall_at_k": 0.5, "llm_judge_faithfulness": 0.1}}
    baseline = {"aggregates": {"recall_at_k": 0.9, "llm_judge_faithfulness": 0.9}}
    failures = compare_baseline(current, baseline, max_drop={"recall_at_k": 0.1, "llm_judge_faithfulness": 0.0})
    assert any("recall_at_k" in item for item in failures)
    assert not any("llm_judge" in item for item in failures)


def test_failure_layers_distinguish_retrieval_from_generation() -> None:
    assert (
        classify_rag_failure(
            passed=False,
            gold_chunk_ids=["a"],
            recall_at_k=0.0,
            abstain_match=False,
            injection_blocked=None,
            citation_valid=None,
        )
        == "retrieval"
    )
    assert (
        classify_rag_failure(
            passed=False,
            gold_chunk_ids=["a"],
            recall_at_k=1.0,
            abstain_match=False,
            injection_blocked=None,
            citation_valid=True,
        )
        == "generation"
    )
    cases = [
        {"case_id": "ok", "type": "answerable", "passed": True},
        {"case_id": "miss", "type": "paraphrase", "passed": False, "failure_layer": "retrieval"},
    ]
    assert scenario_breakdown(cases)["paraphrase"]["failed"] == 1
    assert failures_by_scenario(cases)["paraphrase"][0]["case_id"] == "miss"
    rendered = markdown_report(
        {
            "title": "t",
            "mode": "offline_mock",
            "passed": False,
            "scenario_breakdown": scenario_breakdown(cases),
            "failures_by_scenario": failures_by_scenario(cases),
            "cases": cases,
        }
    )
    assert "Failures by scenario" in rendered
    assert "paraphrase" in rendered
