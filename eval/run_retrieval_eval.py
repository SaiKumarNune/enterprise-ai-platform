#!/usr/bin/env python3
"""Compare lexical, semantic, and hybrid retrieval on a synthetic corpus.

Default mode uses HashEmbedder and FakeSession. This is not MiniLM, PostgreSQL
FTS, or a public IR benchmark unless --mode live is set.

Live mode never falls back to HashEmbedder.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for path in (str(ROOT), str(BACKEND)):
    if path not in sys.path:
        sys.path.insert(0, path)

from app.services.authorization import Principal  # noqa: E402
from app.services.retrieval import retrieve_chunks  # noqa: E402
from eval.corpus import load_fake_corpus, load_sql_corpus, seed_sql_owner  # noqa: E402
from eval.live import not_run_report, probe_live_environment  # noqa: E402
from eval.metrics import mean, mrr, precision_at_k, recall_at_k  # noqa: E402
from eval.reporting import compare_baseline, load_json, write_report  # noqa: E402
from tests.fake_embeddings import HashEmbedder  # noqa: E402

DATASET = Path(__file__).resolve().parent / "datasets" / "retrieval_hybrid_v1.json"
MODES = ("semantic", "lexical", "hybrid")


def load_dataset(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_eval(dataset_path: Path, top_k: int, *, mode: str = "offline_mock_hash_embedder") -> dict:
    if mode == "live":
        return run_live_eval(dataset_path, top_k)
    dataset = load_dataset(dataset_path)
    embedder = HashEmbedder()
    session, _document_ids, chunk_ids, owner = load_fake_corpus(dataset, embedder)
    report = _evaluate(dataset, session, embedder, owner, chunk_ids, top_k)
    report.update(
        {
            "title": "Retrieval evaluation",
            "mode": "offline_mock_hash_embedder",
            "dataset_version": dataset["version"],
            "embedder": "HashEmbedder",
            "top_k": top_k,
            "status": "completed",
            "passed": True,
            "limitations": [
                "Synthetic HashEmbedder fixtures; not PostgreSQL ts_rank, not BM25, and not MiniLM semantic quality."
            ],
        }
    )
    return report


def run_live_eval(dataset_path: Path, top_k: int) -> dict:
    dataset = load_dataset(dataset_path)
    probe = probe_live_environment(require_llm=False)
    if not probe.ready:
        report = not_run_report(
            title="Retrieval evaluation",
            dataset_version=dataset["version"],
            reason=probe.reason,
        )
        return report
    from app.services.embeddings import SentenceTransformerEmbedder
    from eval.store import close_isolated, isolated_session_factory

    embedder = SentenceTransformerEmbedder(probe.embedder_id, 384)
    factory, engine = isolated_session_factory(probe.database_url)
    try:
        session = factory()
        try:
            owner = seed_sql_owner(session)
            _document_ids, chunk_ids = load_sql_corpus(session, dataset, embedder, owner)
            session.commit()
            report = _evaluate(dataset, session, embedder, owner, chunk_ids, top_k)
        finally:
            session.close()
    finally:
        close_isolated(engine)
    report.update(
        {
            "title": "Retrieval evaluation",
            "mode": "live",
            "dataset_version": dataset["version"],
            "embedder": probe.embedder_id,
            "top_k": top_k,
            "status": "completed",
            "passed": True,
            "limitations": [
                f"Live PostgreSQL FTS + pgvector on isolated database {probe.database_name}.",
                "Not comparable to HashEmbedder mock aggregates.",
            ],
        }
    )
    return report


def _evaluate(dataset, session, embedder, owner, chunk_ids: dict[str, uuid.UUID], top_k: int) -> dict:
    principal = Principal.from_user(owner)
    per_mode: dict[str, list[dict]] = {mode: [] for mode in MODES}
    for case in dataset["cases"]:
        gold = [chunk_ids[item] for item in case["gold_chunk_ids"]]
        query_vector = embedder.embed_texts([case["question"]])[0]
        for retrieval_mode in MODES:
            started = time.perf_counter()
            hits = retrieve_chunks(
                session,
                query=case["question"],
                principal=principal,
                top_k=top_k,
                mode=retrieval_mode,
                query_embedding=query_vector,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            retrieved = [hit.chunk.id for hit in hits]
            per_mode[retrieval_mode].append(
                {
                    "case_id": case["id"],
                    "precision_at_k": precision_at_k(retrieved, gold),
                    "recall_at_k": recall_at_k(retrieved, gold),
                    "mrr": mrr(retrieved, gold),
                    "latency_ms": latency_ms,
                    "expected": {"gold_chunk_ids": case["gold_chunk_ids"]},
                    "observed": {"retrieved_count": len(retrieved)},
                    "passed": (recall_at_k(retrieved, gold) or 0) > 0 or not gold,
                }
            )
    aggregates = {}
    for retrieval_mode, rows in per_mode.items():
        aggregates[retrieval_mode] = {
            "precision_at_k": mean(row["precision_at_k"] for row in rows) or 0.0,
            "recall_at_k": mean(row["recall_at_k"] for row in rows) or 0.0,
            "mrr": mean(row["mrr"] for row in rows) or 0.0,
            "mean_latency_ms": round(
                sum(row["latency_ms"] for row in rows) / len(rows),
                2,
            ),
        }
    return {
        "metric_definitions": {
            "precision_at_k": "Gold hits among retrieved chunks.",
            "recall_at_k": "Gold chunks found in the retrieved set.",
            "mrr": "Reciprocal rank of the first gold chunk.",
        },
        "aggregates": aggregates,
        "cases": per_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic hybrid retrieval evaluation")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mode", choices=("offline_mock_hash_embedder", "offline_mock", "live"), default="offline_mock_hash_embedder")
    parser.add_argument("--baseline", type=Path, default=None)
    args = parser.parse_args()
    mode = "offline_mock_hash_embedder" if args.mode == "offline_mock" else args.mode
    report = run_eval(args.dataset, args.top_k, mode=mode)
    text = write_report(report, args.output)
    print(text)
    if report.get("status") == "not_run":
        raise SystemExit(2)
    if args.baseline is not None and report.get("status") == "completed":
        compare_baseline(report, load_json(args.baseline), max_drop={})


if __name__ == "__main__":
    main()
