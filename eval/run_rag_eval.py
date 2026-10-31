#!/usr/bin/env python3
"""Offline RAG evaluation using synthetic documents.

Default mode uses fake embeddings and a scripted LLM. It does not connect to
PostgreSQL, Ollama, or download MiniLM.

Live integration mode is opt-in and documented; this runner does not enable it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.models.chunk import DocumentChunk  # noqa: E402
from app.models.document import Document  # noqa: E402
from app.services.authorization import Principal  # noqa: E402
from app.services.rag import INSUFFICIENT_ANSWER, answer_question  # noqa: E402
from app.services.retrieval import retrieve_chunks  # noqa: E402
from tests.auth_helpers import own_session_documents, seed_user  # noqa: E402
from tests.fake_embeddings import HashEmbedder  # noqa: E402
from tests.test_documents import FakeSession  # noqa: E402

DATASET = Path(__file__).resolve().parent / "datasets" / "rag_quality_v1.json"


@dataclass
class CaseResult:
    case_id: str
    case_type: str
    recall_at_k: float | None
    cited_valid_ids: bool | None
    abstained: bool
    expected_abstain: bool
    abstain_match: bool
    latency_ms: int
    injection_blocked: bool | None


class DatasetLLM:
    """Deterministic stand-in: cite S1 when context exists, otherwise abstain."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if "BEGIN UNTRUSTED DOCUMENT CONTENT" not in user_prompt:
            return INSUFFICIENT_ANSWER
        if "PWNED" in user_prompt and "lost badges" in user_prompt.lower():
            return "Lost badges must be reported to security within 2 hours [S1]."
        return "The retrieved policy excerpt supports a concise answer [S1]."


def load_dataset(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_session(dataset: dict, embedder: HashEmbedder) -> tuple[FakeSession, dict[str, uuid.UUID], dict[str, uuid.UUID]]:
    document_ids: dict[str, uuid.UUID] = {}
    chunk_ids: dict[str, uuid.UUID] = {}
    documents: list[Document] = []
    chunks: list[DocumentChunk] = []
    now = datetime.now(timezone.utc)
    for doc in dataset["documents"]:
        document_id = uuid.uuid4()
        document_ids[doc["id"]] = document_id
        documents.append(
            Document(
                id=document_id,
                filename=doc["filename"],
                content_type="text/plain",
                status="ready",
                created_at=now,
                updated_at=now,
            )
        )
        for index, chunk in enumerate(doc["chunks"]):
            chunk_uuid = uuid.uuid4()
            chunk_ids[chunk["id"]] = chunk_uuid
            content = chunk["content"]
            chunks.append(
                DocumentChunk(
                    id=chunk_uuid,
                    document_id=document_id,
                    chunk_index=index,
                    content=content,
                    page_number=chunk.get("page_number"),
                    created_at=now,
                    embedding=embedder.embed_texts([content])[0],
                )
            )
    session = FakeSession(documents=documents)
    session.chunks = chunks
    owner = seed_user(session)
    own_session_documents(session, owner)
    return session, document_ids, chunk_ids, owner


def recall_at_k(retrieved: list[uuid.UUID], gold: list[uuid.UUID]) -> float | None:
    if not gold:
        return None
    hits = len(set(retrieved) & set(gold))
    return hits / len(gold)


def run_eval(dataset_path: Path, top_k: int) -> dict:
    dataset = load_dataset(dataset_path)
    embedder = HashEmbedder()
    llm = DatasetLLM()
    session, document_ids, chunk_ids, owner = build_session(dataset, embedder)
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
        cited_valid = None if answer.abstained else answer.citation_valid
        injection_blocked = None
        forbidden = case.get("must_not_contain") or []
        if forbidden:
            injection_blocked = not any(token in answer.answer for token in forbidden)
        results.append(
            CaseResult(
                case_id=case["id"],
                case_type=case["type"],
                recall_at_k=recall_at_k(retrieved_ids, gold),
                cited_valid_ids=cited_valid,
                abstained=answer.abstained,
                expected_abstain=bool(case.get("expect_abstain")),
                abstain_match=answer.abstained == bool(case.get("expect_abstain")),
                latency_ms=latency_ms,
                injection_blocked=injection_blocked,
            )
        )

    recall_values = [item.recall_at_k for item in results if item.recall_at_k is not None]
    citation_values = [item.cited_valid_ids for item in results if item.cited_valid_ids is not None]
    return {
        "dataset": dataset_path.name,
        "dataset_version": dataset.get("version"),
        "mode": "offline_mock",
        "top_k": top_k,
        "metric_definitions": {
            "recall_at_k": "Fraction of gold chunk IDs present in the top_k retrieved chunks. Null when no gold IDs.",
            "citation_validity_rate": "Share of generated answers whose remaining citations map to retrieved source IDs.",
            "abstention_accuracy": "Share of cases where abstention matched expect_abstain.",
            "mean_latency_ms": "Mean wall-clock time per case in this mock runner.",
        },
        "notes": [
            "These checks are deterministic heuristics, not human answer-quality scores.",
            "Valid citations are not proof that an answer is factually correct.",
            "Offline mock mode does not call Ollama or PostgreSQL.",
        ],
        "aggregates": {
            "n_cases": len(results),
            "recall_at_k": None if not recall_values else round(sum(recall_values) / len(recall_values), 4),
            "citation_validity_rate": None
            if not citation_values
            else round(sum(1 for value in citation_values if value) / len(citation_values), 4),
            "abstention_accuracy": round(
                sum(1 for item in results if item.abstain_match) / len(results),
                4,
            ),
            "mean_latency_ms": round(sum(item.latency_ms for item in results) / len(results), 2),
        },
        "cases": [asdict(item) for item in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline mock RAG evaluation.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run_eval(args.dataset, args.top_k)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        if args.output.suffix.lower() == ".md":
            args.output.write_text(_markdown_report(report), encoding="utf-8")
        else:
            args.output.write_text(text + "\n", encoding="utf-8")


def _markdown_report(report: dict) -> str:
    aggregates = report["aggregates"]
    lines = [
        f"# RAG eval: {report['dataset']}",
        "",
        f"Mode: `{report['mode']}` (version {report['dataset_version']}, top_k={report['top_k']})",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Definition |",
        "|---|---|---|",
    ]
    for key, definition in report["metric_definitions"].items():
        lines.append(f"| {key} | {aggregates.get(key)} | {definition} |")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in report["notes"])
    lines.extend(["", "## Cases", ""])
    for case in report["cases"]:
        lines.append(
            f"- `{case['case_id']}` ({case['case_type']}): "
            f"recall@k={case['recall_at_k']}, citation_valid={case['cited_valid_ids']}, "
            f"abstain={case['abstained']} expected={case['expected_abstain']}, "
            f"{case['latency_ms']}ms"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
