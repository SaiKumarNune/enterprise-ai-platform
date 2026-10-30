from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.authorization import Principal
from app.services.agents.action import cited_sources, execute_simulated_action, prepare_leave_draft
from app.services.agents.planner import run_planner
from app.services.agents.research import run_research, synthesize_informational_answer
from app.services.agents.trace import append_trace
from app.services.agents.validator import validate_workflow
from app.services.embeddings import EmbeddingBackend
from app.services.llm import LLMBackend
from app.services.rag import INSUFFICIENT_ANSWER

CLARIFICATION_ANSWER = (
    "I need a supporting policy excerpt or a clearer request before preparing an action."
)

# Kept for tests and documentation that import graph prompt constants.
DRAFT_PROMPT = (
    "Create a JSON draft for a leave_request. Return JSON only with keys: "
    "action_type, days, reason, start_date, end_date, employee_name, source_ids."
)


class AgentState(TypedDict, total=False):
    workflow_id: str
    request: str
    document_id: str | None
    top_k: int
    intent: str
    tool: str
    action_type: str | None
    planner_fallback: bool
    fallback_reason: str | None
    task_plan: dict[str, Any]
    excerpts: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    selected_count: int
    retrieval_count: int
    answer: str | None
    draft: dict[str, Any] | None
    status: str
    approval_decision: str | None
    execution_result: dict[str, Any] | None
    authorized: bool
    resume_mode: str | None
    step_count: int
    grounded: bool | None
    citation_valid: bool | None
    abstained: bool | None
    error: str | None
    validation: dict[str, Any]
    agent_trace: list[dict[str, Any]]


def compile_agent_graph(
    db: Session,
    embedder: EmbeddingBackend,
    llm: LLMBackend,
    principal: Principal,
):
    def planner_agent(state: AgentState) -> dict[str, Any]:
        if _over_limit(state):
            return _failed(state, "step_limit")
        plan = run_planner(llm, state["request"])
        return {
            **plan,
            "status": "running",
            "authorized": False,
            "step_count": state.get("step_count", 0) + 1,
            "agent_trace": append_trace(
                state,
                agent="planner",
                outcome="fallback" if plan.get("planner_fallback") else "planned",
                summary=f"Intent {plan['intent']}; execution not authorized.",
            ),
        }

    def research_agent(state: AgentState) -> dict[str, Any]:
        if _over_limit(state):
            return _failed(state, "step_limit")
        retrieved = run_research(
            db,
            embedder,
            question=state["request"],
            top_k=int(state.get("top_k") or 5),
            document_id=state.get("document_id"),
            principal=principal,
        )
        update: dict[str, Any] = {
            **retrieved,
            "step_count": state.get("step_count", 0) + 1,
        }
        excerpts = retrieved["excerpts"]
        intent = state.get("intent") or "informational"
        grounding_issues: list[dict[str, str]] = []
        if not excerpts:
            if intent == "draft_action":
                update.update(_clarification())
                outcome = "needs_clarification"
                summary = "No policy evidence retrieved for the leave request."
            else:
                update.update(_insufficient())
                outcome = "abstained"
                summary = "No retrieved evidence for an informational answer."
        elif intent == "clarification":
            update.update(_clarification())
            outcome = "needs_clarification"
            summary = "Planner requested clarification."
        elif intent == "draft_action":
            outcome = "retrieved"
            summary = f"Retrieved {retrieved['selected_count']} evidence chunk(s)."
        else:
            synthesized = synthesize_informational_answer(
                llm,
                question=state["request"],
                excerpts=excerpts,
                sources=retrieved["sources"],
            )
            grounding_issues = list(synthesized.pop("grounding_issues", []) or [])
            update.update(synthesized)
            outcome = "abstained" if synthesized.get("abstained") else "answered"
            summary = f"Retrieved {retrieved['selected_count']} source(s) and produced an informational answer."
        update["agent_trace"] = append_trace(
            state,
            agent="research",
            outcome=outcome,
            summary=summary,
            issues=grounding_issues,
        )
        return update

    def action_agent(state: AgentState) -> dict[str, Any]:
        if _over_limit(state):
            return _failed(state, "step_limit")
        if state.get("resume_mode") == "approve":
            if state.get("execution_result"):
                return {
                    "status": "completed",
                    "step_count": state.get("step_count", 0) + 1,
                    "agent_trace": append_trace(
                        state,
                        agent="action",
                        outcome="already_executed",
                        summary="Simulated execution was already recorded.",
                    ),
                }
            result = execute_simulated_action(
                draft=state.get("draft") or {},
                authorized=bool(state.get("authorized")),
            )
            return {
                "execution_result": result,
                "approval_decision": "approved",
                "status": "completed",
                "step_count": state.get("step_count", 0) + 1,
                "agent_trace": append_trace(
                    state,
                    agent="action",
                    outcome="executed",
                    summary="Recorded simulated leave_request execution.",
                ),
            }
        draft = prepare_leave_draft(
            llm,
            request=state["request"],
            excerpts=state.get("excerpts") or [],
            sources=state.get("sources") or [],
            action_type=state.get("action_type"),
        )
        if draft is None:
            update = _clarification()
            update.update(
                {
                    "step_count": state.get("step_count", 0) + 1,
                    "agent_trace": append_trace(
                        state,
                        agent="action",
                        outcome="needs_clarification",
                        summary="Required leave fields were missing; no draft was created.",
                    ),
                }
            )
            return update
        cited = cited_sources(state.get("sources") or [], draft)
        return {
            "draft": draft,
            "answer": None,
            "status": "running",
            "sources": cited,
            "grounded": bool(draft.get("source_ids")) and not draft.get("had_invalid_citations"),
            "citation_valid": bool(draft.get("source_ids"))
            and not draft.get("had_invalid_citations"),
            "abstained": False,
            "step_count": state.get("step_count", 0) + 1,
            "agent_trace": append_trace(
                state,
                agent="action",
                outcome="drafted",
                summary="Prepared a leave_request draft; execution is not authorized.",
            ),
        }

    def validator_agent(state: AgentState) -> dict[str, Any]:
        if _over_limit(state):
            return _failed(state, "step_limit")
        result = validate_workflow(dict(state))
        issues = list((result.get("validation") or {}).get("issues") or [])
        passed = bool((result.get("validation") or {}).get("passed"))
        draft = dict(state.get("draft") or {}) if state.get("draft") else None
        if draft is not None:
            draft.pop("had_invalid_citations", None)
        update: dict[str, Any] = {
            **result,
            "draft": draft,
            "step_count": state.get("step_count", 0) + 1,
        }
        intent = state.get("intent")
        if intent == "draft_action" and not passed:
            messages = [
                item["message"]
                for item in issues
                if item.get("severity") == "error"
            ]
            update.update(_clarification())
            update["draft"] = None
            if messages:
                update["answer"] = CLARIFICATION_ANSWER + " " + " ".join(messages)
            outcome = "failed"
            summary = "Validation failed; the workflow will not wait for approval."
        elif intent == "draft_action" and draft and state.get("status") == "running":
            update["status"] = "running"
            outcome = "passed"
            summary = "Validation passed with retrieved-ID checks only; not a factual guarantee."
        else:
            outcome = "passed" if passed else "warnings"
            summary = "Validation finished. Citation ID match is not semantic verification."
        update["agent_trace"] = append_trace(
            state,
            agent="validator",
            outcome=outcome,
            summary=summary,
            issues=issues,
        )
        return update

    def approval_gate(state: AgentState) -> dict[str, Any]:
        if not state.get("draft"):
            return {
                **_clarification(),
                "step_count": state.get("step_count", 0) + 1,
                "agent_trace": append_trace(
                    state,
                    agent="action",
                    outcome="needs_clarification",
                    summary="Approval gate had no draft.",
                ),
            }
        return {
            "status": "pending_approval",
            "approval_decision": None,
            "execution_result": None,
            "authorized": False,
            "step_count": state.get("step_count", 0) + 1,
            "agent_trace": append_trace(
                state,
                agent="action",
                outcome="pending_approval",
                summary="Waiting for an explicit API approval before simulated execution.",
            ),
        }

    def reject_gate(state: AgentState) -> dict[str, Any]:
        return {
            "approval_decision": "rejected",
            "execution_result": None,
            "status": "rejected",
            "authorized": False,
            "step_count": state.get("step_count", 0) + 1,
            "agent_trace": append_trace(
                state,
                agent="action",
                outcome="rejected",
                summary="Human rejected the pending draft. No simulated execution ran.",
            ),
        }

    def finalize(state: AgentState) -> dict[str, Any]:
        status = state.get("status") or "completed"
        return {
            "status": status,
            "step_count": state.get("step_count", 0) + 1,
        }

    builder = StateGraph(AgentState)
    builder.add_node("planner_agent", planner_agent)
    builder.add_node("research_agent", research_agent)
    builder.add_node("action_agent", action_agent)
    builder.add_node("validator_agent", validator_agent)
    builder.add_node("approval_gate", approval_gate)
    builder.add_node("reject_gate", reject_gate)
    builder.add_node("finalize", finalize)
    builder.add_conditional_edges(
        START,
        _route_start,
        {
            "planner_agent": "planner_agent",
            "action_agent": "action_agent",
            "reject_gate": "reject_gate",
        },
    )
    builder.add_edge("planner_agent", "research_agent")
    builder.add_conditional_edges(
        "research_agent",
        _route_after_research,
        {
            "action_agent": "action_agent",
            "validator_agent": "validator_agent",
        },
    )
    builder.add_conditional_edges(
        "action_agent",
        _route_after_action,
        {
            "validator_agent": "validator_agent",
            "finalize": "finalize",
        },
    )
    builder.add_conditional_edges(
        "validator_agent",
        _route_after_validator,
        {
            "approval_gate": "approval_gate",
            "finalize": "finalize",
        },
    )
    builder.add_edge("approval_gate", END)
    builder.add_edge("reject_gate", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


def _route_start(state: AgentState) -> str:
    if state.get("resume_mode") == "approve":
        return "action_agent"
    if state.get("resume_mode") == "reject":
        return "reject_gate"
    return "planner_agent"


def _route_after_action(state: AgentState) -> str:
    if state.get("resume_mode") == "approve":
        return "finalize"
    return "validator_agent"


def _route_after_research(state: AgentState) -> str:
    if (
        state.get("intent") == "draft_action"
        and state.get("status") == "running"
        and state.get("excerpts")
    ):
        return "action_agent"
    return "validator_agent"


def _route_after_validator(state: AgentState) -> str:
    if state.get("status") == "running" and state.get("draft"):
        passed = bool((state.get("validation") or {}).get("passed", True))
        if passed:
            return "approval_gate"
    return "finalize"


def _over_limit(state: AgentState) -> bool:
    return int(state.get("step_count") or 0) >= settings.agent_max_steps


def _failed(state: AgentState, reason: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "error": reason,
        "draft": None,
        "authorized": False,
        "step_count": state.get("step_count", 0) + 1,
        "agent_trace": append_trace(
            state,
            agent="planner",
            outcome="failed",
            summary=f"Stopped: {reason}.",
        ),
    }


def _clarification() -> dict[str, Any]:
    return {
        "answer": CLARIFICATION_ANSWER,
        "draft": None,
        "status": "needs_clarification",
        "grounded": False,
        "citation_valid": True,
        "abstained": True,
    }


def _insufficient() -> dict[str, Any]:
    return {
        "answer": INSUFFICIENT_ANSWER,
        "draft": None,
        "status": "completed",
        "grounded": False,
        "citation_valid": True,
        "abstained": True,
    }
