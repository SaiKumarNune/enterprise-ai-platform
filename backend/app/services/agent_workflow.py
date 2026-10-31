from __future__ import annotations

import time
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.metrics import observe_workflow
from app.models.workflow import AgentWorkflow
from app.services.agent_graph import CLARIFICATION_ANSWER, compile_agent_graph
from app.services.audit import record_audit
from app.services.authorization import DocumentAccessDenied, Principal, require_document
from app.services.users import get_user_by_id
from app.services.agents.trace import append_trace
from app.services.agents.validator import validate_workflow
from app.services.embeddings import EmbeddingBackend
from app.services.llm import LLMBackend


class AgentNotFoundError(Exception):
    """Workflow ID does not exist."""


class AgentConflictError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def run_agent_workflow(
    db: Session,
    embedder: EmbeddingBackend,
    llm: LLMBackend,
    *,
    request: str,
    top_k: int,
    document_id: UUID | None,
    principal: Principal,
    owner_id: UUID | None = None,
) -> AgentWorkflow:
    if document_id is not None:
        try:
            require_document(db, principal, document_id, "read")
        except DocumentAccessDenied:
            raise AgentNotFoundError("document") from None
    now = datetime.now(timezone.utc)
    workflow = AgentWorkflow(
        id=uuid4(),
        request=request,
        owner_id=owner_id or principal.user_id,
        tenant_id=principal.tenant_id,
        document_id=document_id,
        top_k=top_k,
        status="running",
        retrieval_count=0,
        created_at=now,
        updated_at=now,
    )
    db.add(workflow)
    db.commit()
    graph = compile_agent_graph(db, embedder, llm, principal)
    started = time.perf_counter()
    outcome = "error"
    initial = {
        "workflow_id": str(workflow.id),
        "request": request,
        "document_id": str(document_id) if document_id else None,
        "auth_user_id": str(principal.user_id),
        "auth_tenant_id": str(principal.tenant_id),
        "top_k": top_k,
        "step_count": 0,
        "authorized": False,
        "status": "running",
        "sources": [],
        "excerpts": [],
        "retrieval_count": 0,
        "agent_trace": [],
    }
    try:
        state = graph.invoke(initial, config={"recursion_limit": settings.agent_max_steps})
    except Exception:
        workflow.status = "failed"
        workflow.updated_at = datetime.now(timezone.utc)
        db.commit()
        raise
    else:
        _apply_state(workflow, state)
        db.commit()
        db.refresh(workflow)
        outcome = workflow.status or "success"
        return workflow
    finally:
        observe_workflow(outcome, time.perf_counter() - started)


def get_agent_workflow(db: Session, workflow_id: UUID) -> AgentWorkflow:
    workflow = db.get(AgentWorkflow, workflow_id)
    if workflow is None:
        raise AgentNotFoundError("workflow")
    return workflow


def approve_agent_workflow(
    db: Session,
    embedder: EmbeddingBackend,
    llm: LLMBackend,
    workflow_id: UUID,
    *,
    actor_id: UUID | None = None,
    principal: Principal | None = None,
) -> AgentWorkflow:
    actor = principal or _principal_from_actor(db, actor_id)
    workflow = _claim_pending(db, workflow_id, decision="approved")
    state = _state_from_workflow(workflow)
    check = validate_workflow(state)
    if not check["validation"]["passed"]:
        now = datetime.now(timezone.utc)
        issues = list(check["validation"]["issues"])
        messages = [
            item["message"] for item in issues if item.get("severity") == "error"
        ]
        workflow.status = "needs_clarification"
        workflow.approval_decision = None
        workflow.execution_result = None
        workflow.abstained = True
        workflow.answer = CLARIFICATION_ANSWER + ((" " + " ".join(messages)) if messages else "")
        state.update(check)
        state["status"] = "needs_clarification"
        state["authorized"] = False
        state["agent_trace"] = append_trace(
            state,
            agent="validator",
            outcome="failed",
            summary="Re-validation blocked approval of a stale or invalid pending draft.",
            issues=issues,
        )
        _apply_state(workflow, state)
        workflow.updated_at = now
        db.commit()
        db.refresh(workflow)
        return workflow
    state["resume_mode"] = "approve"
    state["authorized"] = True
    graph = compile_agent_graph(db, embedder, llm, actor)
    try:
        next_state = graph.invoke(state, config={"recursion_limit": settings.agent_max_steps})
    except Exception:
        db.rollback()
        raise
    now = datetime.now(timezone.utc)
    _apply_state(workflow, next_state)
    workflow.approval_decision = "approved"
    workflow.decided_by_user_id = actor_id
    workflow.decided_at = workflow.decided_at or now
    if workflow.execution_result:
        workflow.executed_at = workflow.executed_at or now
    workflow.updated_at = now
    record_audit(
        db,
        action="workflow.approve",
        outcome="success",
        principal=actor,
        resource_type="workflow",
        resource_id=workflow.id,
    )
    db.commit()
    db.refresh(workflow)
    return workflow


def reject_agent_workflow(
    db: Session,
    embedder: EmbeddingBackend,
    llm: LLMBackend,
    workflow_id: UUID,
    *,
    actor_id: UUID | None = None,
    principal: Principal | None = None,
) -> AgentWorkflow:
    actor = principal or _principal_from_actor(db, actor_id)
    workflow = _claim_pending(db, workflow_id, decision="rejected")
    state = _state_from_workflow(workflow)
    state["resume_mode"] = "reject"
    state["authorized"] = False
    graph = compile_agent_graph(db, embedder, llm, actor)
    try:
        next_state = graph.invoke(state, config={"recursion_limit": settings.agent_max_steps})
    except Exception:
        db.rollback()
        raise
    now = datetime.now(timezone.utc)
    _apply_state(workflow, next_state)
    workflow.status = "rejected"
    workflow.approval_decision = "rejected"
    workflow.decided_by_user_id = actor_id
    workflow.execution_result = None
    workflow.decided_at = now
    workflow.updated_at = now
    record_audit(
        db,
        action="workflow.reject",
        outcome="success",
        principal=actor,
        resource_type="workflow",
        resource_id=workflow.id,
    )
    db.commit()
    db.refresh(workflow)
    return workflow


def _principal_from_actor(db: Session, actor_id: UUID | None) -> Principal:
    if actor_id is None:
        raise AgentConflictError("A trusted principal is required")
    user = get_user_by_id(db, actor_id)
    if user is None or not user.is_active:
        raise AgentConflictError("Actor is not an active user")
    return Principal.from_user(user)


def _claim_pending(db: Session, workflow_id: UUID, *, decision: str) -> AgentWorkflow:
    workflow = db.get(AgentWorkflow, workflow_id, with_for_update=True)
    if workflow is None:
        db.rollback()
        raise AgentNotFoundError("workflow")
    if workflow.status != "pending_approval":
        db.rollback()
        if workflow.intent == "informational" or workflow.draft is None:
            raise AgentConflictError("This workflow is not waiting for action approval")
        raise AgentConflictError("This workflow has already been decided")
    if workflow.approval_decision is not None or workflow.execution_result is not None:
        db.rollback()
        raise AgentConflictError("This workflow has already been decided")
    workflow.approval_decision = decision
    workflow.updated_at = datetime.now(timezone.utc)
    return workflow


def _state_from_workflow(workflow: AgentWorkflow) -> dict:
    if workflow.graph_state:
        state = dict(workflow.graph_state)
    else:
        state = {
            "request": workflow.request,
            "document_id": str(workflow.document_id) if workflow.document_id else None,
            "top_k": workflow.top_k,
            "intent": workflow.intent,
            "draft": workflow.draft,
            "sources": workflow.sources or [],
            "answer": workflow.answer,
            "status": workflow.status,
            "retrieval_count": workflow.retrieval_count,
        }
    state["workflow_id"] = str(workflow.id)
    state["authorized"] = False
    return state


def _apply_state(workflow: AgentWorkflow, state: dict) -> None:
    workflow.status = state.get("status") or workflow.status
    workflow.intent = state.get("intent")
    workflow.answer = state.get("answer")
    workflow.draft = state.get("draft")
    workflow.sources = state.get("sources") or []
    workflow.retrieval_count = int(state.get("retrieval_count") or 0)
    workflow.grounded = state.get("grounded")
    workflow.citation_valid = state.get("citation_valid")
    workflow.abstained = state.get("abstained")
    workflow.approval_decision = state.get("approval_decision")
    workflow.execution_result = state.get("execution_result")
    workflow.agent_trace = list(state.get("agent_trace") or [])
    persistable = {
        key: value
        for key, value in state.items()
        if key not in {"selected"}
    }
    workflow.graph_state = persistable
    workflow.updated_at = datetime.now(timezone.utc)
