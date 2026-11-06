"""Opt-in live evaluation probes. Never fall back to fake embeddings or scripted LLMs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import httpx
from sqlalchemy.engine import make_url

from tests.postgres_isolation import require_isolated_url

FORBIDDEN_LIVE_FALLBACK = "Live evaluation cannot use HashEmbedder or ScriptedLLM"


@dataclass(frozen=True)
class LiveProbe:
    ready: bool
    reason: str
    database_url: str = ""
    database_name: str = ""
    embedder_id: str = ""
    llm_id: str = ""


def evaluation_database_url() -> str:
    return os.getenv("EVAL_DATABASE_URL", "").strip() or os.getenv("TEST_DATABASE_URL", "").strip()


def probe_live_environment(*, require_llm: bool) -> LiveProbe:
    url = evaluation_database_url()
    if not url:
        return LiveProbe(
            False,
            "EVAL_DATABASE_URL or TEST_DATABASE_URL is unset; live mode will not use development PostgreSQL",
        )
    try:
        require_isolated_url(url)
    except RuntimeError as exc:
        return LiveProbe(False, str(exc))
    name = make_url(url).database or ""
    embedder_id = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    if not _minilm_is_cached(embedder_id):
        return LiveProbe(
            False,
            f"SentenceTransformer model {embedder_id} is not available locally (downloads are disabled)",
            database_url=url,
            database_name=name,
        )
    llm_id = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    if require_llm:
        ollama = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        if not _ollama_reachable(ollama, llm_id):
            return LiveProbe(
                False,
                f"Ollama is unavailable at {ollama} for model {llm_id}",
                database_url=url,
                database_name=name,
                embedder_id=embedder_id,
            )
    return LiveProbe(
        True,
        "ok",
        database_url=url,
        database_name=name,
        embedder_id=embedder_id,
        llm_id=llm_id if require_llm else "",
    )


def not_run_report(*, title: str, dataset_version: str, reason: str) -> dict:
    return {
        "title": title,
        "status": "not_run",
        "passed": None,
        "mode": "live_not_run",
        "dataset_version": dataset_version,
        "aggregates": {},
        "cases": [],
        "limitations": [reason, FORBIDDEN_LIVE_FALLBACK],
        "notes": ["Live mode did not fall back to offline mocks."],
    }


def _minilm_is_cached(model_name: str) -> bool:
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(model_name, local_files_only=True)
        return True
    except Exception:
        cache = Path.home() / ".cache" / "huggingface" / "hub"
        needle = model_name.replace("/", "--")
        if cache.is_dir() and any(needle in path.name for path in cache.iterdir()):
            return True
        return False


def _ollama_reachable(base_url: str, model: str) -> bool:
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=2.0)
        response.raise_for_status()
        names = [item.get("name", "") for item in response.json().get("models", [])]
        return any(model in name for name in names) if names else True
    except Exception:
        return False
