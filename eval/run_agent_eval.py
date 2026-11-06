#!/usr/bin/env python3
"""LangGraph agent orchestration evaluation.

Offline mode uses SequenceLLM and FakeSession. It never sends email or calls
external workflow systems. Approval still uses the existing approve/reject APIs
in-process. Live LLM mode is opt-in and does not fall back to SequenceLLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for path in (str(ROOT), str(BACKEND)):
    if path not in sys.path:
        sys.path.insert(0, path)

from app.models.chunk import DocumentChunk  # noqa: E402
from app.services.agent_workflow import (  # noqa: E402
    AgentNotFoundError,
    approve_agent_workflow,
    reject_agent_workflow,
    run_agent_workflow,
)
from app.services.authorization import Principal  # noqa: E402
from eval.classify import classify_agent_failure, failures_by_scenario, scenario_breakdown  # noqa: E402
from eval.live import not_run_report, probe_live_environment  # noqa: E402
from eval.reporting import write_report  # noqa: E402
from eval.ticketing_env import ticketing_enabled  # noqa: E402
from tests.auth_helpers import own_session_documents, seed_tenant, seed_user  # noqa: E402
from tests.fake_embeddings import HashEmbedder  # noqa: E402
from tests.fake_llm import SequenceLLM  # noqa: E402
from tests.test_agents import _chunk, _sample_document  # noqa: E402
from tests.test_documents import FakeSession  # noqa: E402

DATASET = Path(__file__).resolve().parent / "datasets" / "agent_orchestration_v1.json"
DEFAULT_POLICY = (
    "Full-time employees receive 15 vacation days each calendar year. Unused vacation does not carry over."
)
STRICT_POLICY = (
    "Employees may request up to 10 days of paid vacation annually. "
    "Vacation requests must be submitted at least 7 days before planned leave. "
    "Requests for 3 or more consecutive days require manager approval. "
    "Employees must specify the number of leave days and the reason. "
    "No leave request may be finalized without explicit manager approval."
)


def load_dataset(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_eval(dataset_path: Path, *, mode: str = "offline_mock") -> dict:
    dataset = load_dataset(dataset_path)
    if mode == "live":
        probe = probe_live_environment(require_llm=True)
        reason = (
            probe.reason
            if not probe.ready
            else (
                "Live Ollama agent eval is disabled by default because planner JSON is "
                "nondeterministic; offline SequenceLLM measures routing and approval gates."
            )
        )
        return not_run_report(
            title="Agent orchestration evaluation",
            dataset_version=str(dataset.get("version")),
            reason=reason,
        )
    cases = []
    for case in dataset["cases"]:
        cases.append(_run_case(case, live_llm=None))
    passed = all(item["passed"] for item in cases)
    return {
        "title": "Agent orchestration evaluation",
        "dataset": dataset_path.name,
        "dataset_version": dataset.get("version"),
        "mode": "offline_mock",
        "status": "completed",
        "passed": passed,
        "embedder": "HashEmbedder",
        "llm": "SequenceLLM",
        "metric_definitions": {
            "routing_accuracy": "Share of cases whose agent_trace includes expected agents.",
            "status_accuracy": "Share of cases whose final status matched expected_status.",
            "approval_gate_compliance": "Draft cases must not execute until followup=approve; rejects must not execute.",
            "issue_code_recall": "Share of expected validator issue codes observed.",
        },
        "aggregates": {
            "n_cases": len(cases),
            "routing_accuracy": _rate(cases, "routing_ok"),
            "status_accuracy": _rate(cases, "status_ok"),
            "approval_gate_compliance": _rate(cases, "approval_ok"),
            "issue_code_recall": _rate(cases, "issues_ok"),
        },
        "scenario_breakdown": scenario_breakdown(cases),
        "failures_by_scenario": failures_by_scenario(cases),
        "cases": cases,
        "limitations": [
            "Offline SequenceLLM scripts the planner/action text; this measures graph routing and gates, not Ollama planning quality.",
            "Execution remains simulated; no external HR or ticketing vendor action is taken.",
            "Ticket cases use the in-process mock adapter. Scores are not live planner quality.",
        ],
        "notes": ["Traces are structured agent/outcome/summary records, not private chain-of-thought."],
    }


def _run_case(case: dict, live_llm) -> dict:
    with ticketing_enabled(bool(case.get("ticketing_enabled"))):
        return _run_case_body(case, live_llm)


def _run_case_body(case: dict, live_llm) -> dict:
    session, document = _policy_session(case.get("policy") or "default")
    employee = seed_user(session, email=f"emp-{uuid4().hex[:8]}@example.test")
    manager = seed_user(session, role="manager", email=f"mgr-{uuid4().hex[:8]}@example.test", tenant_id=employee.tenant_id)
    own_session_documents(session, employee)
    embedder = HashEmbedder()
    llm = live_llm or SequenceLLM(list(case.get("llm_responses") or []))
    principal = Principal.from_user(employee)
    document_id = document.id
    if case.get("cross_tenant"):
        outsider = seed_user(session, tenant_id=seed_tenant(session, name="OtherEval").id)
        principal = Principal.from_user(outsider)
    expected_error = case.get("expected_error")
    observed: dict = {}
    try:
        workflow = run_agent_workflow(
            session,
            embedder,
            llm,
            request=case["request"],
            top_k=5,
            document_id=document_id,
            principal=principal,
        )
        followup = case.get("followup")
        if followup == "approve":
            workflow = approve_agent_workflow(
                session,
                embedder,
                llm,
                workflow.id,
                actor_id=manager.id,
                principal=Principal.from_user(manager),
            )
        elif followup == "reject":
            workflow = reject_agent_workflow(
                session,
                embedder,
                llm,
                workflow.id,
                actor_id=manager.id,
                principal=Principal.from_user(manager),
            )
        observed = {
            "status": workflow.status,
            "intent": workflow.intent,
            "agents": [step.get("agent") for step in (workflow.agent_trace or [])],
            "draft": workflow.draft is not None,
            "action_type": (workflow.draft or {}).get("action_type"),
            "execution": workflow.execution_result,
            "approval_decision": workflow.approval_decision,
            "issue_codes": [
                issue.get("code")
                for step in (workflow.agent_trace or [])
                for issue in step.get("issues") or []
            ],
            "error": None,
        }
    except AgentNotFoundError:
        observed = {
            "status": None,
            "intent": None,
            "agents": [],
            "draft": False,
            "action_type": None,
            "execution": None,
            "approval_decision": None,
            "issue_codes": [],
            "error": "document_not_found",
        }

    routing_ok = _routing_ok(case, observed)
    status_ok = _status_ok(case, observed)
    approval_ok = _approval_ok(case, observed)
    issues_ok = _issues_ok(case, observed)
    action_ok = _action_ok(case, observed)
    passed = routing_ok and status_ok and approval_ok and issues_ok and action_ok
    if expected_error:
        passed = observed.get("error") == expected_error
        routing_ok = status_ok = approval_ok = issues_ok = action_ok = passed
    return {
        "case_id": case["id"],
        "type": case["type"],
        "case_type": case["type"],
        "expected": {
            "status": case.get("expected_status"),
            "intent": case.get("expected_intent"),
            "error": expected_error,
            "expect_execution": bool(case.get("expect_execution")),
            "action_type": case.get("expected_action_type"),
            "ticket_created": case.get("expect_ticket_created"),
        },
        "observed": observed,
        "routing_ok": routing_ok,
        "status_ok": status_ok,
        "approval_ok": approval_ok,
        "issues_ok": issues_ok,
        "action_ok": action_ok,
        "failure_layer": classify_agent_failure(passed=passed, case=case, observed=observed),
        "passed": passed,
    }


def _policy_session(policy: str) -> tuple[FakeSession, object]:
    content = STRICT_POLICY if policy == "strict" else DEFAULT_POLICY
    document = _sample_document("leave_policy.txt")
    session = FakeSession(documents=[document])
    session.chunks = [_chunk(document.id, content)]
    return session, document


def _routing_ok(case: dict, observed: dict) -> bool:
    expected_agents = case.get("expected_agents") or []
    agents = observed.get("agents") or []
    return all(agent in agents for agent in expected_agents)


def _status_ok(case: dict, observed: dict) -> bool:
    expected = case.get("expected_status")
    if not expected:
        return True
    return observed.get("status") == expected and (
        case.get("expected_intent") in {None, observed.get("intent")}
    )


def _approval_ok(case: dict, observed: dict) -> bool:
    executed = bool(observed.get("execution"))
    created = bool((observed.get("execution") or {}).get("created"))
    if case.get("expect_ticket_created") is True:
        return executed and created
    if case.get("expect_ticket_created") is False and case.get("followup") == "approve":
        return executed and not created
    if case.get("expect_execution"):
        return executed and bool((observed.get("execution") or {}).get("simulated", True))
    if case.get("expect_approval_gate"):
        return observed.get("status") == "pending_approval" and not executed
    if case.get("followup") == "reject":
        return observed.get("approval_decision") == "rejected" and not executed
    return not executed


def _action_ok(case: dict, observed: dict) -> bool:
    expected = case.get("expected_action_type")
    if not expected:
        return True
    return observed.get("action_type") == expected


def _issues_ok(case: dict, observed: dict) -> bool:
    expected = set(case.get("expected_issue_codes") or [])
    if not expected:
        return True
    return expected.issubset(set(observed.get("issue_codes") or []))


def _rate(cases: list[dict], key: str) -> float:
    if not cases:
        return 0.0
    return round(sum(1 for item in cases if item.get(key)) / len(cases), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LangGraph agent orchestration.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mode", choices=("offline_mock", "live"), default="offline_mock")
    args = parser.parse_args()
    if args.mode == "live":
        probe = probe_live_environment(require_llm=True)
        report = not_run_report(
            title="Agent orchestration evaluation",
            dataset_version=load_dataset(args.dataset).get("version"),
            reason=(
                probe.reason
                if not probe.ready
                else "Live Ollama agent eval is not enabled by default; offline SequenceLLM measures routing gates. Pass remains mock unless a dedicated live harness is added."
            ),
        )
        text = write_report(report, args.output)
        print(text)
        raise SystemExit(2)
    report = run_eval(args.dataset, mode=args.mode)
    text = write_report(report, args.output)
    print(text)
    if not report.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
