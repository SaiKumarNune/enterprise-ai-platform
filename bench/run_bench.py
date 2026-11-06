#!/usr/bin/env python3
"""Bounded local performance benchmarks.

Modes:
  mock      In-memory FakeSession + HashEmbedder + ScriptedLLM (always safe)
  database  Isolated *_test PostgreSQL + HashEmbedder (no Ollama)
  live      Isolated *_test PostgreSQL + MiniLM; Ollama RAG is a separate scenario
  auto      mock, then database/live only when those services are configured

Never uses or resets the development database named enterprise_ai.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from itertools import cycle
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for path in (str(ROOT), str(BACKEND)):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("JWT_SECRET", "bench-only-jwt-secret-key-not-for-prod-use")
os.environ.setdefault("METRICS_ENABLED", "false")

from fastapi.testclient import TestClient  # noqa: E402

from app.core.database import get_db  # noqa: E402
from app.core.security import create_access_token  # noqa: E402
from app.main import app  # noqa: E402
from app.models.document import Document  # noqa: E402
from app.services.authorization import Principal  # noqa: E402
from app.services.embeddings import get_embedder  # noqa: E402
from app.services.llm import get_llm  # noqa: E402
from app.services.processing_jobs import claim_next_job, enqueue_processing_job, run_claimed_job  # noqa: E402
from app.services.retrieval import retrieve_chunks  # noqa: E402
from app.services.storage import LocalStorage, get_storage  # noqa: E402
from bench.corpus import DOCUMENTS, QUERIES  # noqa: E402
from bench.harness import run_pool, time_call  # noqa: E402
from eval.corpus import load_fake_corpus, load_sql_corpus, seed_sql_owner  # noqa: E402
from eval.live import evaluation_database_url, not_run_report, probe_live_environment  # noqa: E402
from eval.reporting import write_report  # noqa: E402
from eval.store import close_isolated, isolated_session_factory  # noqa: E402
from tests.auth_helpers import bearer_headers  # noqa: E402
from tests.fake_embeddings import HashEmbedder  # noqa: E402
from tests.fake_llm import ScriptedLLM  # noqa: E402
from tests.postgres_isolation import require_isolated_url  # noqa: E402
from tests.test_documents import FakeSession, _override_db  # noqa: E402

DATASET = {
    "documents": [
        {
            "id": item["id"],
            "filename": item["filename"],
            "chunks": [
                {"id": f"{item['id']}-{index}", "content": chunk, "page_number": 1}
                for index, chunk in enumerate(item["chunks"])
            ],
        }
        for item in DOCUMENTS
    ]
}

MOCK_ANSWER = "Full-time employees receive 15 vacation days each calendar year [S1]."


def benchmark_database_url() -> str:
    return os.getenv("BENCH_DATABASE_URL", "").strip() or evaluation_database_url()


def _scenario(name: str, layer: str, summary, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "id": name,
        "layer": layer,
        "status": "completed",
        **summary.as_dict(),
    }
    if extra:
        payload.update(extra)
    return payload


def _not_run_scenario(name: str, layer: str, reason: str) -> dict[str, Any]:
    return {"id": name, "layer": layer, "status": "not_run", "reason": reason}


def _search(client: TestClient, headers: dict[str, str], query: str, mode: str) -> None:
    response = client.post(
        "/search",
        json={"query": query, "top_k": 5, "retrieval_mode": mode},
        headers=headers,
    )
    if response.status_code != 200:
        raise RuntimeError(f"search HTTP {response.status_code}")


def _rag(client: TestClient, headers: dict[str, str], question: str) -> None:
    response = client.post(
        "/rag/ask",
        json={"question": question, "top_k": 5},
        headers=headers,
    )
    if response.status_code != 200:
        raise RuntimeError(f"rag HTTP {response.status_code}")


def _run_search_http(client: TestClient, headers: dict[str, str], *, n: int, concurrency: int, mode: str):
    queries = cycle(QUERIES)

    def work() -> float:
        return time_call(lambda: _search(client, headers, next(queries), mode))

    return run_pool(work, n=n, concurrency=concurrency)


def _run_rag_http(client: TestClient, headers: dict[str, str], *, n: int, concurrency: int):
    queries = cycle(QUERIES)

    def work() -> float:
        return time_call(lambda: _rag(client, headers, next(queries)))

    return run_pool(work, n=n, concurrency=concurrency)


def _run_retrieval_sql(session_factory, principal: Principal, embedder, *, n: int, concurrency: int, mode: str):
    queries = cycle(QUERIES)

    def work() -> float:
        query = next(queries)
        db = session_factory()
        try:
            embedding = None
            if mode in {"semantic", "hybrid"}:
                embedding = embedder.embed_texts([query])[0]
            started = time.perf_counter()
            retrieve_chunks(
                db,
                query=query,
                principal=principal,
                top_k=5,
                mode=mode,
                query_embedding=embedding,
            )
            return (time.perf_counter() - started) * 1000.0
        finally:
            db.close()

    return run_pool(work, n=n, concurrency=concurrency)


def _run_jobs(session, storage, embedder, documents: list[Document], principal, *, n: int) -> dict[str, Any]:
    jobs = []
    enqueue_started = time.perf_counter()
    for document in documents[:n]:
        jobs.append(enqueue_processing_job(session, document, principal))
    enqueue_s = time.perf_counter() - enqueue_started
    drain_started = time.perf_counter()
    succeeded = 0
    failed = 0
    latencies: list[float] = []
    while True:
        claimed = claim_next_job(session, "bench-worker")
        if claimed is None:
            break
        started = time.perf_counter()
        finished = run_claimed_job(session, claimed, storage, embedder, worker_id="bench-worker")
        latencies.append((time.perf_counter() - started) * 1000.0)
        if finished.status == "succeeded":
            succeeded += 1
        else:
            failed += 1
    drain_s = max(time.perf_counter() - drain_started, 1e-9)
    from bench.metrics import summarize

    timing = summarize(latencies, errors=failed, elapsed_s=drain_s, n=max(len(latencies), 1))
    payload = timing.as_dict()
    payload.update(
        {
            "enqueue_s": round(enqueue_s, 4),
            "jobs_enqueued": len(jobs),
            "jobs_succeeded": succeeded,
            "jobs_failed": failed,
            "job_throughput_per_s": round(succeeded / drain_s, 3),
            "queue_drain_s": round(drain_s, 4),
        }
    )
    return payload


def run_mock(*, requests: int, concurrency: int, jobs: int) -> dict[str, Any]:
    embedder = HashEmbedder()
    session, _docs, _chunks, owner = load_fake_corpus(DATASET, embedder)
    _override_db(session)
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_llm] = lambda: ScriptedLLM(MOCK_ANSWER)
    headers = bearer_headers(owner)
    scenarios: list[dict[str, Any]] = []
    try:
        with TestClient(app) as client:
            for mode in ("lexical", "semantic", "hybrid"):
                summary, rss = _run_search_http(
                    client, headers, n=requests, concurrency=concurrency, mode=mode
                )
                scenarios.append(
                    _scenario(
                        f"search_{mode}",
                        "http_mock",
                        summary,
                        extra={"retrieval_mode": mode, "rss_delta_bytes": rss},
                    )
                )
            rag_summary, rag_rss = _run_rag_http(client, headers, n=requests, concurrency=concurrency)
            scenarios.append(
                _scenario("rag_mocked_llm", "http_mock", rag_summary, extra={"rss_delta_bytes": rag_rss})
            )
        tmp = Path(tempfile.mkdtemp(prefix="bench-jobs-"))
        storage = LocalStorage(root=tmp, max_bytes=10 * 1024 * 1024)
        job_docs: list[Document] = []
        for index in range(jobs):
            spec = DOCUMENTS[index % len(DOCUMENTS)]
            name = f"{uuid.uuid4()}.txt"
            (tmp / name).write_text(spec["text"], encoding="utf-8")
            document = Document(
                id=uuid.uuid4(),
                filename=spec["filename"],
                content_type="text/plain",
                status="uploaded",
                storage_path=name,
                tenant_id=owner.tenant_id,
                owner_id=owner.id,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            session.add(document)
            job_docs.append(document)
        job_stats = _run_jobs(
            session, storage, embedder, job_docs, Principal.from_user(owner), n=jobs
        )
        scenarios.append(
            {
                "id": "processing_jobs",
                "layer": "worker_mock",
                "status": "completed",
                **job_stats,
            }
        )
        principal = Principal.from_user(owner)
        session_lock = threading.Lock()
        for mode in ("lexical", "semantic", "hybrid"):
            queries = cycle(QUERIES)

            def work(current_mode: str = mode) -> float:
                query = next(queries)
                embedding = None if current_mode == "lexical" else embedder.embed_texts([query])[0]
                started = time.perf_counter()
                with session_lock:
                    retrieve_chunks(
                        session,
                        query=query,
                        principal=principal,
                        top_k=5,
                        mode=current_mode,
                        query_embedding=embedding,
                    )
                return (time.perf_counter() - started) * 1000.0

            summary, rss = run_pool(work, n=requests, concurrency=min(concurrency, 2))
            scenarios.append(
                _scenario(
                    f"retrieval_{mode}",
                    "in_memory",
                    summary,
                    extra={"retrieval_mode": mode, "rss_delta_bytes": rss},
                )
            )
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_embedder, None)
        app.dependency_overrides.pop(get_llm, None)
    return {
        "title": "Local performance benchmark",
        "mode": "mock",
        "status": "completed",
        "embedder": "HashEmbedder",
        "llm": "ScriptedLLM",
        "scenarios": scenarios,
        "limitations": [
            "Mock mode does not measure PostgreSQL, MiniLM, or Ollama.",
            "HashEmbedder and ScriptedLLM timings are not production model latency.",
            "TestClient concurrency is in-process ASGI, not a separate uvicorn worker.",
        ],
        "notes": [
            "Layer labels: http_mock, worker_mock, in_memory.",
            "RSS delta is ru_maxrss change and is not a precise cgroup measurement.",
        ],
    }


def run_database(*, requests: int, concurrency: int, jobs: int) -> dict[str, Any]:
    url = benchmark_database_url()
    if not url:
        report = not_run_report(
            title="Local performance benchmark",
            dataset_version="bench-v1",
            reason="BENCH_DATABASE_URL / EVAL_DATABASE_URL / TEST_DATABASE_URL unset",
        )
        report["mode"] = "database_not_run"
        return report
    try:
        require_isolated_url(url)
    except RuntimeError as exc:
        report = not_run_report(
            title="Local performance benchmark",
            dataset_version="bench-v1",
            reason=str(exc),
        )
        report["mode"] = "database_not_run"
        return report
    embedder = HashEmbedder()
    factory, engine = isolated_session_factory(url)
    scenarios: list[dict[str, Any]] = []
    tmp = Path(tempfile.mkdtemp(prefix="bench-db-jobs-"))
    storage = LocalStorage(root=tmp, max_bytes=10 * 1024 * 1024)
    try:
        session = factory()
        owner = seed_sql_owner(session)
        load_sql_corpus(session, DATASET, embedder, owner)
        session.commit()
        principal = Principal.from_user(owner)

        def override_db():
            db = factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_db
        app.dependency_overrides[get_embedder] = lambda: embedder
        app.dependency_overrides[get_llm] = lambda: ScriptedLLM(MOCK_ANSWER)
        app.dependency_overrides[get_storage] = lambda: storage
        headers = {"Authorization": f"Bearer {create_access_token(owner.id)}"}
        with TestClient(app) as client:
            for mode in ("lexical", "semantic", "hybrid"):
                summary, rss = _run_search_http(
                    client, headers, n=requests, concurrency=concurrency, mode=mode
                )
                scenarios.append(
                    _scenario(
                        f"search_{mode}",
                        "http_database",
                        summary,
                        extra={"retrieval_mode": mode, "rss_delta_bytes": rss},
                    )
                )
            rag_summary, rag_rss = _run_rag_http(client, headers, n=min(requests, 20), concurrency=min(concurrency, 4))
            scenarios.append(
                _scenario(
                    "rag_mocked_llm_over_postgres",
                    "http_database",
                    rag_summary,
                    extra={"rss_delta_bytes": rag_rss, "llm": "ScriptedLLM"},
                )
            )
        for mode in ("lexical", "semantic", "hybrid"):
            summary, rss = _run_retrieval_sql(
                factory, principal, embedder, n=requests, concurrency=concurrency, mode=mode
            )
            scenarios.append(
                _scenario(
                    f"pg_retrieval_{mode}",
                    "sql_database",
                    summary,
                    extra={"retrieval_mode": mode, "rss_delta_bytes": rss},
                )
            )
        job_docs: list[Document] = []
        for index in range(jobs):
            spec = DOCUMENTS[index % len(DOCUMENTS)]
            name = f"{uuid.uuid4()}.txt"
            (tmp / name).write_text(spec["text"], encoding="utf-8")
            document = Document(
                id=uuid.uuid4(),
                filename=spec["filename"],
                content_type="text/plain",
                status="uploaded",
                storage_path=name,
                tenant_id=owner.tenant_id,
                owner_id=owner.id,
            )
            session.add(document)
            job_docs.append(document)
        session.commit()
        for document in job_docs:
            session.refresh(document)
        job_stats = _run_jobs(session, storage, embedder, job_docs, principal, n=jobs)
        scenarios.append(
            {
                "id": "processing_jobs",
                "layer": "worker_database",
                "status": "completed",
                **job_stats,
            }
        )
        session.close()
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_embedder, None)
        app.dependency_overrides.pop(get_llm, None)
        app.dependency_overrides.pop(get_storage, None)
        close_isolated(engine)
    return {
        "title": "Local performance benchmark",
        "mode": "database",
        "status": "completed",
        "embedder": "HashEmbedder",
        "llm": "ScriptedLLM",
        "database": "isolated_*_test",
        "scenarios": scenarios,
        "limitations": [
            "Database mode uses HashEmbedder, not MiniLM.",
            "RAG in this mode uses ScriptedLLM; it is not Ollama latency.",
            "isolated_session_factory recreates tables in the isolated *_test database only.",
        ],
        "notes": [
            "sql_database scenarios call retrieve_chunks on PostgreSQL.",
            "http_database scenarios hit FastAPI against the same isolated database.",
        ],
    }


def run_live(*, requests: int, concurrency: int, rag_requests: int) -> dict[str, Any]:
    retrieval_probe = probe_live_environment(require_llm=False)
    if not retrieval_probe.ready:
        report = not_run_report(
            title="Local performance benchmark",
            dataset_version="bench-v1",
            reason=retrieval_probe.reason,
        )
        report["mode"] = "live_not_run"
        return report
    from app.services.embeddings import SentenceTransformerEmbedder
    from app.core.config import settings

    embedder = SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_dimensions)
    factory, engine = isolated_session_factory(retrieval_probe.database_url)
    scenarios: list[dict[str, Any]] = []
    try:
        session = factory()
        owner = seed_sql_owner(session)
        load_sql_corpus(session, DATASET, embedder, owner)
        session.commit()
        principal = Principal.from_user(owner)
        for mode in ("lexical", "semantic", "hybrid"):
            summary, rss = _run_retrieval_sql(
                factory, principal, embedder, n=requests, concurrency=concurrency, mode=mode
            )
            scenarios.append(
                _scenario(
                    f"pg_retrieval_{mode}",
                    "sql_live_minilm",
                    summary,
                    extra={"retrieval_mode": mode, "rss_delta_bytes": rss},
                )
            )
        session.close()
    finally:
        close_isolated(engine)
    llm_probe = probe_live_environment(require_llm=True)
    if not llm_probe.ready:
        scenarios.append(_not_run_scenario("rag_ollama", "http_live_ollama", llm_probe.reason))
    else:
        from app.services.llm import OllamaChatClient

        llm = OllamaChatClient(settings.ollama_base_url, settings.ollama_model, settings.ollama_timeout_seconds)
        factory, engine = isolated_session_factory(llm_probe.database_url)
        try:
            session = factory()
            owner = seed_sql_owner(session)
            load_sql_corpus(session, DATASET, embedder, owner)
            session.commit()

            def override_db():
                db = factory()
                try:
                    yield db
                finally:
                    db.close()

            app.dependency_overrides[get_db] = override_db
            app.dependency_overrides[get_embedder] = lambda: embedder
            app.dependency_overrides[get_llm] = lambda: llm
            headers = {"Authorization": f"Bearer {create_access_token(owner.id)}"}
            with TestClient(app) as client:
                summary, rss = _run_rag_http(
                    client, headers, n=rag_requests, concurrency=min(concurrency, 2)
                )
            scenarios.append(
                _scenario(
                    "rag_ollama",
                    "http_live_ollama",
                    summary,
                    extra={"rss_delta_bytes": rss, "llm": settings.ollama_model},
                )
            )
            session.close()
        finally:
            app.dependency_overrides.pop(get_db, None)
            app.dependency_overrides.pop(get_embedder, None)
            app.dependency_overrides.pop(get_llm, None)
            close_isolated(engine)
    return {
        "title": "Local performance benchmark",
        "mode": "live",
        "status": "completed",
        "embedder": retrieval_probe.embedder_id,
        "llm": "Ollama" if llm_probe.ready else "not_run",
        "scenarios": scenarios,
        "limitations": [
            "Live MiniLM/Ollama numbers must not be mixed with HashEmbedder or ScriptedLLM rows.",
            "Small synthetic corpus; not a production-scale load test.",
        ],
        "notes": [
            "sql_live_minilm is PostgreSQL + MiniLM without Ollama.",
            "http_live_ollama is the only scenario that measures generation.",
        ],
    }


def run_auto(*, requests: int, concurrency: int, jobs: int, rag_requests: int) -> dict[str, Any]:
    reports = [run_mock(requests=requests, concurrency=concurrency, jobs=jobs)]
    db_report = run_database(requests=requests, concurrency=concurrency, jobs=jobs)
    reports.append(db_report)
    live_report = run_live(requests=min(requests, 20), concurrency=min(concurrency, 4), rag_requests=rag_requests)
    reports.append(live_report)
    return {
        "title": "Local performance benchmark",
        "mode": "auto",
        "status": "completed",
        "reports": reports,
        "limitations": [
            "auto concatenates mock, database, and live reports; compare only rows with the same layer label.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded local performance benchmarks.")
    parser.add_argument("--mode", choices=("mock", "database", "live", "auto"), default="auto")
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--live-rag-requests", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1 or args.jobs < 1:
        raise SystemExit("requests, concurrency, and jobs must be >= 1")
    if args.mode == "mock":
        report = run_mock(requests=args.requests, concurrency=args.concurrency, jobs=args.jobs)
    elif args.mode == "database":
        report = run_database(requests=args.requests, concurrency=args.concurrency, jobs=args.jobs)
    elif args.mode == "live":
        report = run_live(
            requests=args.requests, concurrency=args.concurrency, rag_requests=args.live_rag_requests
        )
    else:
        report = run_auto(
            requests=args.requests,
            concurrency=args.concurrency,
            jobs=args.jobs,
            rag_requests=args.live_rag_requests,
        )
    text = write_report(report, args.output)
    print(text)
    if report.get("status") == "not_run" or report.get("mode", "").endswith("not_run"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
