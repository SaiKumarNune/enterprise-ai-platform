from pathlib import Path

from bench.metrics import percentile, summarize
from bench.run_bench import run_database, run_live, run_mock


def test_percentile_and_error_rate() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert percentile(values, 50) == 3.0
    assert percentile([], 95) is None
    summary = summarize([10.0, 20.0], errors=1, elapsed_s=1.0, n=3)
    assert summary.error_rate == 1 / 3
    assert summary.rps == 3.0
    assert summary.success_rps == 2.0
    assert summary.p50_ms == 15.0


def test_mock_bench_is_bounded_and_labeled() -> None:
    report = run_mock(requests=6, concurrency=2, jobs=3)
    assert report["status"] == "completed"
    assert report["mode"] == "mock"
    assert report["embedder"] == "HashEmbedder"
    assert report["llm"] == "ScriptedLLM"
    ids = {item["id"] for item in report["scenarios"]}
    assert "search_hybrid" in ids
    assert "rag_mocked_llm" in ids
    assert "processing_jobs" in ids
    assert "retrieval_lexical" in ids
    rag = next(item for item in report["scenarios"] if item["id"] == "rag_mocked_llm")
    assert rag["layer"] == "http_mock"
    assert rag["n"] == 6
    assert rag["error_rate"] == 0
    jobs = next(item for item in report["scenarios"] if item["id"] == "processing_jobs")
    assert jobs["jobs_succeeded"] == 3
    assert jobs["job_throughput_per_s"] > 0


def test_database_mode_refuses_missing_and_development_urls(monkeypatch) -> None:
    monkeypatch.delenv("BENCH_DATABASE_URL", raising=False)
    monkeypatch.delenv("EVAL_DATABASE_URL", raising=False)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    missing = run_database(requests=2, concurrency=1, jobs=1)
    assert missing["status"] == "not_run"
    assert missing["mode"] == "database_not_run"

    monkeypatch.setenv(
        "BENCH_DATABASE_URL",
        "postgresql+psycopg://ci_user:ci_test_password_not_for_prod@127.0.0.1:5432/enterprise_ai",
    )
    refused = run_database(requests=2, concurrency=1, jobs=1)
    assert refused["status"] == "not_run"
    blob = " ".join(
        [
            str(refused.get("reason") or ""),
            " ".join(refused.get("limitations") or []),
            " ".join(refused.get("notes") or []),
        ]
    ).lower()
    assert "enterprise_ai" in blob or "development" in blob or "refusing" in blob or "_test" in blob


def test_live_mode_does_not_silently_use_mocks(monkeypatch) -> None:
    monkeypatch.delenv("BENCH_DATABASE_URL", raising=False)
    monkeypatch.delenv("EVAL_DATABASE_URL", raising=False)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    report = run_live(requests=2, concurrency=1, rag_requests=1)
    assert report["status"] == "not_run"
    assert report["mode"] == "live_not_run"
    assert report.get("embedder") not in {"HashEmbedder"}
    assert report.get("llm") not in {"ScriptedLLM"}


def test_write_report_roundtrip(tmp_path: Path) -> None:
    from eval.reporting import load_json, write_report

    report = run_mock(requests=4, concurrency=1, jobs=2)
    path = tmp_path / "bench.json"
    write_report(report, path)
    loaded = load_json(path)
    assert loaded["mode"] == "mock"
    assert path.with_suffix(".md").exists()
