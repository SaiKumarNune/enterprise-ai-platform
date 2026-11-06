#!/usr/bin/env python3
"""RAG evaluation using synthetic documents.

Default mode uses fake embeddings and a scripted LLM. It does not connect to
PostgreSQL, Ollama, or download MiniLM.

Live mode is explicit: --mode live. It never falls back to HashEmbedder or
DatasetLLM. If MiniLM, Ollama, or an isolated eval database is missing, the
runner records status=not_run and exits 2.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for path in (str(ROOT), str(BACKEND)):
    if path not in sys.path:
        sys.path.insert(0, path)

from app.services.authorization import Principal  # noqa: E402
from app.services.rag import INSUFFICIENT_ANSWER, answer_question  # noqa: E402
from app.services.retrieval import retrieve_chunks  # noqa: E402
from eval.classify import classify_rag_failure, failures_by_scenario, scenario_breakdown  # noqa: E402
from eval.corpus import load_fake_corpus, load_sql_corpus, seed_sql_owner  # noqa: E402
from eval.hosted import HostedLLMError, build_hosted_client, probe_hosted_environment  # noqa: E402
from eval.live import not_run_report, probe_live_environment  # noqa: E402
from eval.metrics import claim_coverage, citation_precision, mean, mrr, recall_at_k  # noqa: E402
from eval.reporting import compare_baseline, load_json, write_report  # noqa: E402
from tests.fake_embeddings import HashEmbedder  # noqa: E402

DATASET = Path(__file__).resolve().parent / "datasets" / "rag_quality_v1.json"

METRIC_DEFINITIONS = {
    "recall_at_k": "Fraction of gold chunk IDs present in the top_k retrieved chunks. Null when no gold IDs.",
    "mrr": "Mean reciprocal rank of the first gold chunk. Null when no gold IDs.",
    "citation_validity_rate": "Share of generated answers whose remaining citations map to retrieved source IDs.",
    "citation_precision": "Share of cited chunks that are gold evidence chunks. Null when nothing is cited.",
    "abstention_accuracy": "Share of cases where abstention matched expect_abstain.",
    "answer_claim_coverage": "Deterministic substring coverage of reference_claims in the generated answer. Not NLI.",
    "evidence_claim_coverage": "Deterministic substring coverage of reference_claims in retrieved chunk text.",
    "mean_latency_ms": "Mean wall-clock time per case.",
    "llm_judge_faithfulness": "Optional live-only LLM-as-judge score. Labeled heuristic, not a pass/fail gate.",
}


@dataclass
class CaseResult:
    case_id: str
    case_type: str
    recall_at_k: float | None
    mrr: float | None
    cited_valid_ids: bool | None
    citation_precision: float | None
    abstained: bool
    expected_abstain: bool
    abstain_match: bool
    answer_claim_coverage: float | None
    evidence_claim_coverage: float | None
    latency_ms: int
    injection_blocked: bool | None
    llm_judge_faithful: bool | None
    failure_layer: str | None
    expected: dict
    observed: dict
    passed: bool


class DatasetLLM:
    """Deterministic stand-in: cite S1 when context exists, otherwise abstain."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if "BEGIN UNTRUSTED DOCUMENT CONTENT" not in user_prompt:
            return INSUFFICIENT_ANSWER
        if "PWNED" in user_prompt and "lost badges" in user_prompt.lower():
            return "Lost badges must be reported to security within 2 hours [S1]."
        lowered = user_prompt.lower()
        if "encryption" in lowered and "PWNED" in user_prompt:
            return "Laptop encryption must stay enabled at all times [S1]."
        if "superseded 2024" in lowered or "2024 handbook" in lowered:
            return "The superseded 2024 handbook granted 12 vacation days [S1]."
        if "2026 handbook" in lowered:
            return "The current 2026 handbook grants 15 vacation days each calendar year [S1]."
        if "vacation days and sick days" in lowered:
            return "Employees receive vacation days and sick days under the leave policy [S1]."
        if "15 vacation days" in user_prompt:
            return "Full-time employees receive 15 vacation days each calendar year [S1]."
        if "doctor's note" in user_prompt.lower() or "sick days" in user_prompt.lower():
            return "A doctor's note is required after three consecutive sick days [S1]."
        return "The retrieved policy excerpt supports a concise answer [S1]."


def load_dataset(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_eval(dataset_path: Path, top_k: int, *, mode: str = "offline_mock", llm_judge: bool = False) -> dict:
    if mode == "live":
        return run_live_eval(dataset_path, top_k, llm_judge=llm_judge)
    if mode == "hosted":
        return run_hosted_eval(dataset_path, top_k, llm_judge=llm_judge)
    if mode not in {"offline_mock", "offline"}:
        raise ValueError("mode must be offline_mock, live, or hosted")
    dataset = load_dataset(dataset_path)
    embedder = HashEmbedder()
    llm = DatasetLLM()
    session, document_ids, chunk_ids, owner = load_fake_corpus(dataset, embedder)
    report = _evaluate_cases(
        dataset=dataset,
        session=session,
        embedder=embedder,
        llm=llm,
        owner=owner,
        document_ids=document_ids,
        chunk_ids=chunk_ids,
        top_k=top_k,
        llm_judge=False,
    )
    report.update(
        {
            "title": "RAG evaluation",
            "dataset": dataset_path.name,
            "dataset_version": dataset.get("version"),
            "mode": "offline_mock",
            "embedder": "HashEmbedder",
            "llm": "DatasetLLM",
            "top_k": top_k,
            "metric_definitions": METRIC_DEFINITIONS,
            "limitations": [
                "Offline mock mode does not call Ollama or PostgreSQL.",
                "HashEmbedder is not MiniLM; paraphrase recall can collapse.",
                "answer_claim_coverage is substring matching, not proven faithfulness.",
            ],
            "notes": [
                "These checks are deterministic heuristics, not human answer-quality scores.",
                "Valid citations are not proof that an answer is factually correct.",
                "Mock HashEmbedder/DatasetLLM scores are not MiniLM, Ollama, or hosted-model accuracy.",
            ],
        }
    )
    report["passed"] = all(case["passed"] for case in report["cases"])
    report["status"] = "completed"
    return report


def run_live_eval(dataset_path: Path, top_k: int, *, llm_judge: bool) -> dict:
    dataset = load_dataset(dataset_path)
    probe = probe_live_environment(require_llm=True)
    if not probe.ready:
        report = not_run_report(
            title="RAG evaluation",
            dataset_version=str(dataset.get("version")),
            reason=probe.reason,
        )
        report["dataset"] = dataset_path.name
        report["metric_definitions"] = METRIC_DEFINITIONS
        return report
    from app.services.embeddings import SentenceTransformerEmbedder
    from app.services.llm import OllamaChatClient
    from eval.store import close_isolated, isolated_session_factory

    embedder = SentenceTransformerEmbedder(probe.embedder_id, 384)
    llm = OllamaChatClient(
        base_url=__import__("os").getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        model=probe.llm_id,
        timeout_seconds=60,
    )
    factory, engine = isolated_session_factory(probe.database_url)
    try:
        session = factory()
        try:
            owner = seed_sql_owner(session)
            document_ids, chunk_ids = load_sql_corpus(session, dataset, embedder, owner)
            session.commit()
            report = _evaluate_cases(
                dataset=dataset,
                session=session,
                embedder=embedder,
                llm=llm,
                owner=owner,
                document_ids=document_ids,
                chunk_ids=chunk_ids,
                top_k=top_k,
                llm_judge=llm_judge,
            )
        finally:
            session.close()
    finally:
        close_isolated(engine)
    report.update(
        {
            "title": "RAG evaluation",
            "dataset": dataset_path.name,
            "dataset_version": dataset.get("version"),
            "mode": "live",
            "embedder": probe.embedder_id,
            "llm": probe.llm_id,
            "top_k": top_k,
            "metric_definitions": METRIC_DEFINITIONS,
            "limitations": [
                "Live metrics are not comparable to HashEmbedder mock scores.",
                "LLM-as-judge is optional, labeled, and excluded from default regression gates.",
                f"Isolated database {probe.database_name} was used; development enterprise_ai was not mutated.",
            ],
            "notes": ["Live mode used MiniLM embeddings, PostgreSQL retrieval, and Ollama generation."],
        }
    )
    report["passed"] = all(case["passed"] for case in report["cases"])
    report["status"] = "completed"
    return report


def run_hosted_eval(dataset_path: Path, top_k: int, *, llm_judge: bool) -> dict:
    dataset = load_dataset(dataset_path)
    probe = probe_hosted_environment()
    if not probe.ready:
        report = not_run_report(
            title="RAG evaluation",
            dataset_version=str(dataset.get("version")),
            reason=probe.reason,
        )
        report["dataset"] = dataset_path.name
        report["mode"] = "hosted_not_run"
        report["metric_definitions"] = METRIC_DEFINITIONS
        return report
    from app.services.embeddings import SentenceTransformerEmbedder
    from eval.store import close_isolated, isolated_session_factory

    embedder = SentenceTransformerEmbedder(probe.embedder_id, 384)
    llm = build_hosted_client()
    factory, engine = isolated_session_factory(probe.database_url)
    try:
        session = factory()
        try:
            owner = seed_sql_owner(session)
            document_ids, chunk_ids = load_sql_corpus(session, dataset, embedder, owner)
            session.commit()
            report = _evaluate_cases(
                dataset=dataset,
                session=session,
                embedder=embedder,
                llm=llm,
                owner=owner,
                document_ids=document_ids,
                chunk_ids=chunk_ids,
                top_k=top_k,
                llm_judge=llm_judge,
            )
        finally:
            session.close()
    finally:
        close_isolated(engine)
    report.update(
        {
            "title": "RAG evaluation",
            "dataset": dataset_path.name,
            "dataset_version": dataset.get("version"),
            "mode": "hosted_openai_compatible",
            "embedder": probe.embedder_id,
            "llm": probe.llm_id,
            "top_k": top_k,
            "metric_definitions": METRIC_DEFINITIONS,
            "limitations": [
                "Hosted-model scores are not comparable to HashEmbedder mock scores or local Ollama scores.",
                "This runner does not claim production accuracy.",
                "LLM-as-judge is optional, labeled, and excluded from default regression gates.",
                f"Isolated database {probe.database_name} was used; development enterprise_ai was not mutated.",
            ],
            "notes": [
                "Hosted mode used MiniLM retrieval plus an OpenAI-compatible chat completions endpoint.",
                "No paid API is required unless you choose to point EVAL_HOSTED_LLM_* at a vendor.",
            ],
        }
    )
    report["passed"] = all(case["passed"] for case in report["cases"])
    report["status"] = "completed"
    return report


def _evaluate_cases(
    *,
    dataset: dict,
    session,
    embedder,
    llm,
    owner,
    document_ids: dict[str, uuid.UUID],
    chunk_ids: dict[str, uuid.UUID],
    top_k: int,
    llm_judge: bool,
) -> dict:
    principal = Principal.from_user(owner)
    results: list[CaseResult] = []
    for case in dataset["cases"]:
        started = time.perf_counter()
        document_key = case.get("document_id")
        document_uuid = document_ids.get(document_key) if document_key else None
        query_vector = embedder.embed_texts([case["question"]])[0]
        retrieved_hits = retrieve_chunks(
            session,
            query=case["question"],
            principal=principal,
            top_k=top_k,
            query_embedding=query_vector,
            document_id=document_uuid,
        )
        retrieved_ids = [hit.chunk.id for hit in retrieved_hits]
        gold = [chunk_ids[item] for item in case.get("gold_chunk_ids", [])]
        answer = answer_question(
            session,
            embedder,
            llm,
            question=case["question"],
            top_k=top_k,
            principal=principal,
            document_id=document_uuid,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        cited_chunk_ids = []
        source_by_id = {item.source_id: item.chunk_id for item in answer.sources}
        for source in answer.sources:
            cited_chunk_ids.append(source.chunk_id)
        claims = list(case.get("reference_claims") or [])
        evidence_text = " ".join(hit.chunk.content for hit in retrieved_hits)
        cited_valid = None if answer.abstained else answer.citation_valid
        injection_blocked = None
        forbidden = case.get("must_not_contain") or []
        if forbidden:
            injection_blocked = not any(token in answer.answer for token in forbidden)
        expected_abstain = bool(case.get("expect_abstain"))
        judge = None
        if llm_judge and not answer.abstained:
            try:
                judge = _llm_judge(llm, evidence_text, answer.answer)
            except HostedLLMError:
                judge = None
        passed = answer.abstained == expected_abstain
        if injection_blocked is False:
            passed = False
        recall = recall_at_k(retrieved_ids, gold)
        result = CaseResult(
            case_id=case["id"],
            case_type=case["type"],
            recall_at_k=recall,
            mrr=mrr(retrieved_ids, gold),
            cited_valid_ids=cited_valid,
            citation_precision=citation_precision(cited_chunk_ids, gold) if cited_chunk_ids else None,
            abstained=answer.abstained,
            expected_abstain=expected_abstain,
            abstain_match=answer.abstained == expected_abstain,
            answer_claim_coverage=claim_coverage(answer.answer, claims),
            evidence_claim_coverage=claim_coverage(evidence_text, claims),
            latency_ms=latency_ms,
            injection_blocked=injection_blocked,
            llm_judge_faithful=judge,
            failure_layer=classify_rag_failure(
                passed=passed,
                gold_chunk_ids=list(case.get("gold_chunk_ids") or []),
                recall_at_k=recall,
                abstain_match=answer.abstained == expected_abstain,
                injection_blocked=injection_blocked,
                citation_valid=cited_valid,
            ),
            expected={"abstain": expected_abstain, "gold_chunk_ids": case.get("gold_chunk_ids", [])},
            observed={
                "abstain": answer.abstained,
                "citation_valid": cited_valid,
                "source_ids": list(source_by_id),
            },
            passed=passed,
        )
        results.append(result)
    recall_values = [item.recall_at_k for item in results]
    citation_values = [item.cited_valid_ids for item in results if item.cited_valid_ids is not None]
    judge_values = [item.llm_judge_faithful for item in results if item.llm_judge_faithful is not None]
    return {
        "aggregates": {
            "n_cases": len(results),
            "recall_at_k": mean(recall_values),
            "mrr": mean(item.mrr for item in results),
            "citation_validity_rate": None
            if not citation_values
            else round(sum(1 for value in citation_values if value) / len(citation_values), 4),
            "citation_precision": mean(item.citation_precision for item in results),
            "abstention_accuracy": round(
                sum(1 for item in results if item.abstain_match) / len(results),
                4,
            ),
            "answer_claim_coverage": mean(item.answer_claim_coverage for item in results),
            "evidence_claim_coverage": mean(item.evidence_claim_coverage for item in results),
            "mean_latency_ms": round(sum(item.latency_ms for item in results) / len(results), 2),
            "llm_judge_faithfulness": None
            if not judge_values
            else round(sum(1 for value in judge_values if value) / len(judge_values), 4),
        },
        "scenario_breakdown": scenario_breakdown([asdict(item) for item in results]),
        "failures_by_scenario": failures_by_scenario([asdict(item) for item in results]),
        "cases": [asdict(item) for item in results],
    }


def _llm_judge(llm, evidence: str, answer: str) -> bool | None:
    prompt = (
        "You are an LLM-as-judge, not a source of ground truth. "
        "Return JSON {\"faithful\": true or false}. faithful=true only if every "
        "factual claim in ANSWER is supported by EVIDENCE.\n\n"
        f"EVIDENCE:\n{evidence}\n\nANSWER:\n{answer}"
    )
    raw = llm.generate("Return JSON only.", prompt)
    try:
        payload = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    value = payload.get("faithful")
    return bool(value) if isinstance(value, bool) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mode", choices=("offline_mock", "live", "hosted"), default="offline_mock")
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Optional LLM-as-judge for live or hosted mode (not a CI gate).",
    )
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument(
        "--max-drop",
        action="append",
        default=[],
        metavar="METRIC=DELTA",
        help="Deterministic regression drop, e.g. recall_at_k=0.15. Ignored for llm_judge_faithfulness.",
    )
    args = parser.parse_args()
    if args.mode not in {"live", "hosted"} and args.llm_judge:
        raise SystemExit(" --llm-judge requires --mode live or --mode hosted")
    report = run_eval(args.dataset, args.top_k, mode=args.mode, llm_judge=args.llm_judge)
    if args.baseline is not None and report.get("status") == "completed":
        drops = {}
        for item in args.max_drop:
            key, value = item.split("=", 1)
            drops[key] = float(value)
        failures = compare_baseline(report, load_json(args.baseline), max_drop=drops)
        report["baseline_failures"] = failures
        if failures:
            report["passed"] = False
    text = write_report(report, args.output)
    print(text)
    if report.get("status") == "not_run":
        raise SystemExit(2)
    if args.baseline is not None and report.get("baseline_failures"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
