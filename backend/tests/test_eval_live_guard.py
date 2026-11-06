from eval.live import probe_live_environment
from eval.run_rag_eval import DATASET, run_eval


def test_live_probe_rejects_development_database(monkeypatch) -> None:
    monkeypatch.setenv(
        "EVAL_DATABASE_URL",
        "postgresql+psycopg://ci_user:ci_test_password_not_for_prod@127.0.0.1:5432/enterprise_ai",
    )
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    probe = probe_live_environment(require_llm=False)
    assert probe.ready is False
    assert "development" in probe.reason.lower() or "enterprise_ai" in probe.reason.lower()


def test_rag_live_mode_does_not_fall_back_to_mocks(monkeypatch) -> None:
    monkeypatch.delenv("EVAL_DATABASE_URL", raising=False)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    report = run_eval(DATASET, 5, mode="live")
    assert report["status"] == "not_run"
    assert report["mode"] == "live_not_run"
    assert report.get("embedder") not in {"HashEmbedder"}
    assert report.get("llm") not in {"DatasetLLM"}
    blob = " ".join(report.get("notes") or []) + " ".join(report.get("limitations") or [])
    assert "fall back" in blob.lower() or "HashEmbedder" in blob
