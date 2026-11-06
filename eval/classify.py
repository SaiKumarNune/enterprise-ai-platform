"""Classify eval failures by scenario and layer. Heuristic labels, not ground truth."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def classify_rag_failure(
    *,
    passed: bool,
    gold_chunk_ids: list,
    recall_at_k: float | None,
    abstain_match: bool,
    injection_blocked: bool | None,
    citation_valid: bool | None,
) -> str | None:
    if passed:
        return None
    if injection_blocked is False:
        return "generation"
    if gold_chunk_ids and recall_at_k == 0:
        return "retrieval"
    if citation_valid is False and abstain_match:
        return "generation"
    if not abstain_match:
        return "generation"
    return "unknown"


def classify_agent_failure(*, passed: bool, case: dict[str, Any], observed: dict[str, Any]) -> str | None:
    if passed:
        return None
    if observed.get("error") or case.get("expected_error") or case.get("cross_tenant"):
        return "authorization"
    if case.get("expect_execution") and not observed.get("execution"):
        return "authorization"
    if case.get("expect_approval_gate") and observed.get("execution"):
        return "authorization"
    if case.get("followup") == "reject" and observed.get("execution"):
        return "authorization"
    if case.get("expected_status") and observed.get("status") != case.get("expected_status"):
        return "generation"
    return "unknown"


def classify_mcp_failure(*, passed: bool, case: dict[str, Any]) -> str | None:
    if passed:
        return None
    if case.get("type") in {"security", "authorization"} or case.get("cross_tenant") or case.get("inactive"):
        return "authorization"
    if case.get("type") == "schema":
        return "generation"
    return "unknown"


def scenario_breakdown(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for case in cases:
        scenario = str(case.get("case_type") or case.get("type") or "unspecified")
        bucket = buckets.setdefault(
            scenario,
            {"n": 0, "passed": 0, "failed": 0, "failed_case_ids": [], "failure_layers": defaultdict(int)},
        )
        bucket["n"] += 1
        if case.get("passed"):
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1
            bucket["failed_case_ids"].append(case.get("case_id"))
            layer = case.get("failure_layer") or "unknown"
            bucket["failure_layers"][layer] += 1
    serialized = {}
    for key, bucket in buckets.items():
        serialized[key] = {
            "n": bucket["n"],
            "passed": bucket["passed"],
            "failed": bucket["failed"],
            "failed_case_ids": bucket["failed_case_ids"],
            "failure_layers": dict(bucket["failure_layers"]),
        }
    return serialized


def failures_by_scenario(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        if case.get("passed"):
            continue
        scenario = str(case.get("case_type") or case.get("type") or "unspecified")
        grouped[scenario].append(
            {
                "case_id": case.get("case_id"),
                "failure_layer": case.get("failure_layer") or "unknown",
            }
        )
    return dict(grouped)
