#!/usr/bin/env python3
"""Compare lexical, semantic, and hybrid retrieval on a synthetic corpus.

Uses HashEmbedder and FakeSession. This is not MiniLM, PostgreSQL FTS, or a
public IR benchmark. Metrics describe fixture behavior only.
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
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.models.chunk import DocumentChunk  # noqa: E402
from app.models.document import Document  # noqa: E402
from app.services.authorization import Principal  # noqa: E402
from app.services.retrieval import retrieve_chunks  # noqa: E402
from tests.auth_helpers import own_session_documents, seed_user  # noqa: E402
from tests.fake_embeddings import HashEmbedder  # noqa: E402
from tests.test_documents import FakeSession  # noqa: E402

DATASET = Path(__file__).resolve().parent / "datasets" / "retrieval_hybrid_v1.json"


def load_dataset(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_session(dataset: dict, embedder: HashEmbedder) -> tuple[FakeSession, dict, dict, object]:
    documents: list[Document] = []
    chunks: list[DocumentChunk] = []
    document_ids: dict[str, uuid.UUID] = {}
    chunk_ids: dict[str, uuid.UUID] = {}
    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    for item in dataset["documents"]:
        document = Document(
            id=uuid.uuid4(),
            filename=item["filename"],
            content_type="text/plain",
            status="ready",
            created_at=now,
            updated_at=now,
        )
        document_ids[item["id"]] = document.id
        documents.append(document)
        for index, chunk in enumerate(item["chunks"]):
            row = DocumentChunk(
                id=uuid.uuid4(),
                document_id=document.id,
                chunk_index=index,
                content=chunk["content"],
                page_number=1,
                created_at=now,
                embedding=embedder.embed_texts([chunk["content"]])[0],
            )
            chunk_ids[chunk["id"]] = row.id
            chunks.append(row)
    session = FakeSession(documents=documents)
    session.chunks = chunks
    owner = seed_user(session)
    own_session_documents(session, owner)
    return session, document_ids, chunk_ids, owner


def precision_at_k(retrieved: list[uuid.UUID], gold: list[uuid.UUID]) -> float | None:
    if not retrieved:
        return 0.0 if gold else None
    hits = len(set(retrieved) & set(gold))
    return hits / len(retrieved)


def recall_at_k(retrieved: list[uuid.UUID], gold: list[uuid.UUID]) -> float | None:
    if not gold:
        return None
    return len(set(retrieved) & set(gold)) / len(gold)


def mrr(retrieved: list[uuid.UUID], gold: list[uuid.UUID]) -> float | None:
    if not gold:
        return None
    wanted = set(gold)
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in wanted:
            return 1.0 / rank
    return 0.0


def run_eval(dataset_path: Path, top_k: int) -> dict:
    dataset = load_dataset(dataset_path)
    embedder = HashEmbedder()
    session, _document_ids, chunk_ids, owner = build_session(dataset, embedder)
    principal = Principal.from_user(owner)
    modes = ("semantic", "lexical", "hybrid")
    per_mode: dict[str, list[dict]] = {mode: [] for mode in modes}
    for case in dataset["cases"]:
        gold = [chunk_ids[item] for item in case["gold_chunk_ids"]]
        query_vector = embedder.embed_texts([case["question"]])[0]
        for mode in modes:
            started = time.perf_counter()
            hits = retrieve_chunks(
                session,
                query=case["question"],
                principal=principal,
                top_k=top_k,
                mode=mode,
                query_embedding=query_vector,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            retrieved = [hit.chunk.id for hit in hits]
            per_mode[mode].append(
                {
                    "case_id": case["id"],
                    "precision_at_k": precision_at_k(retrieved, gold),
                    "recall_at_k": recall_at_k(retrieved, gold),
                    "mrr": mrr(retrieved, gold),
                    "latency_ms": latency_ms,
                }
            )
    aggregates = {}
    for mode, rows in per_mode.items():
        def _mean(key: str) -> float:
            values = [row[key] for row in rows if row[key] is not None]
            return sum(values) / len(values) if values else 0.0

        aggregates[mode] = {
            "precision_at_k": round(_mean("precision_at_k"), 4),
            "recall_at_k": round(_mean("recall_at_k"), 4),
            "mrr": round(_mean("mrr"), 4),
            "mean_latency_ms": round(_mean("latency_ms"), 2),
        }
    return {
        "mode": "offline_mock_hash_embedder",
        "dataset_version": dataset["version"],
        "top_k": top_k,
        "limitations": (
            "Synthetic HashEmbedder fixtures; not PostgreSQL ts_rank, not BM25, "
            "and not MiniLM semantic quality."
        ),
        "aggregates": aggregates,
        "cases": per_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic hybrid retrieval evaluation")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    report = run_eval(args.dataset, args.top_k)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
