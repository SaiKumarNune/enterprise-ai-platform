"""Low-cardinality Prometheus metrics. Label values are never IDs or queries."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest

from app.core.config import settings

REGISTRY = CollectorRegistry()

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "HTTP requests",
    ["method", "route", "status"],
    registry=REGISTRY,
)
HTTP_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "route"],
    registry=REGISTRY,
)
RETRIEVAL_REQUESTS = Counter(
    "retrieval_requests_total",
    "Chunk retrieval attempts",
    ["mode", "outcome"],
    registry=REGISTRY,
)
RETRIEVAL_DURATION = Histogram(
    "retrieval_duration_seconds",
    "Chunk retrieval latency",
    ["mode"],
    registry=REGISTRY,
)
RAG_REQUESTS = Counter(
    "rag_requests_total",
    "RAG question attempts",
    ["outcome"],
    registry=REGISTRY,
)
RAG_DURATION = Histogram(
    "rag_request_duration_seconds",
    "RAG request latency",
    registry=REGISTRY,
)
LLM_REQUESTS = Counter(
    "llm_requests_total",
    "LLM generate attempts",
    ["outcome"],
    registry=REGISTRY,
)
LLM_DURATION = Histogram(
    "llm_request_duration_seconds",
    "LLM generate latency",
    registry=REGISTRY,
)
WORKFLOW_REQUESTS = Counter(
    "workflow_requests_total",
    "LangGraph workflow runs",
    ["outcome"],
    registry=REGISTRY,
)
WORKFLOW_DURATION = Histogram(
    "workflow_duration_seconds",
    "LangGraph workflow latency",
    registry=REGISTRY,
)

_SAFE_MODES = frozenset({"semantic", "lexical", "hybrid"})
_SAFE_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})
_SAFE_OUTCOMES = frozenset(
    {
        "success",
        "empty",
        "error",
        "grounded",
        "abstained",
        "ungrounded",
        "failed",
        "completed",
        "pending_approval",
        "needs_clarification",
        "rejected",
        "running",
        "other",
    }
)


def metrics_enabled() -> bool:
    raw = os.getenv("METRICS_ENABLED")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes"}
    return settings.metrics_enabled


def metrics_token() -> str:
    return os.getenv("METRICS_TOKEN", settings.metrics_token).strip()


def _safe(value: str, allowed: frozenset[str] | None = None, default: str = "other") -> str:
    cleaned = (value or default).strip() or default
    if allowed is not None and cleaned not in allowed:
        return default
    return cleaned[:64]


def observe_http(method: str, route: str, status: int, duration_seconds: float) -> None:
    if not metrics_enabled():
        return
    try:
        method_label = _safe(method.upper(), _SAFE_METHODS, "OTHER")
        route_label = route if route.startswith("/") else "unmatched"
        HTTP_REQUESTS.labels(method=method_label, route=route_label[:120], status=str(int(status))).inc()
        HTTP_DURATION.labels(method=method_label, route=route_label[:120]).observe(duration_seconds)
    except Exception:
        return


def observe_retrieval(mode: str, outcome: str, duration_seconds: float) -> None:
    if not metrics_enabled():
        return
    try:
        RETRIEVAL_REQUESTS.labels(
            mode=_safe(mode, _SAFE_MODES, "other"),
            outcome=_safe(outcome, _SAFE_OUTCOMES),
        ).inc()
        RETRIEVAL_DURATION.labels(mode=_safe(mode, _SAFE_MODES, "other")).observe(duration_seconds)
    except Exception:
        return


def observe_rag(outcome: str, duration_seconds: float) -> None:
    if not metrics_enabled():
        return
    try:
        RAG_REQUESTS.labels(outcome=_safe(outcome, _SAFE_OUTCOMES)).inc()
        RAG_DURATION.observe(duration_seconds)
    except Exception:
        return


def observe_llm(outcome: str, duration_seconds: float) -> None:
    if not metrics_enabled():
        return
    try:
        LLM_REQUESTS.labels(outcome=_safe(outcome, _SAFE_OUTCOMES)).inc()
        LLM_DURATION.observe(duration_seconds)
    except Exception:
        return


def observe_workflow(outcome: str, duration_seconds: float) -> None:
    if not metrics_enabled():
        return
    try:
        WORKFLOW_REQUESTS.labels(outcome=_safe(outcome, _SAFE_OUTCOMES)).inc()
        WORKFLOW_DURATION.observe(duration_seconds)
    except Exception:
        return


@contextmanager
def track_duration() -> Iterator[list[float]]:
    started = time.perf_counter()
    bucket = [0.0]
    try:
        yield bucket
    finally:
        bucket[0] = time.perf_counter() - started


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
