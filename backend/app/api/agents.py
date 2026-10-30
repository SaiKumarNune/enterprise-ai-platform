from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_roles
from app.core.config import settings
from app.core.database import get_db
from app.models.user import ADMIN, MANAGER, User
from app.models.workflow import AgentWorkflow
from app.schemas.agent import (
    AgentDecisionRequest,
    AgentRunRequest,
    AgentTraceStep,
    AgentWorkflowResponse,
)
from app.schemas.rag import RagSourceResponse
from app.services.access import can_read_workflow
from app.services.agent_workflow import (
    AgentConflictError,
    AgentNotFoundError,
    approve_agent_workflow,
    get_agent_workflow,
    reject_agent_workflow,
    run_agent_workflow,
)
from app.services.embeddings import EmbeddingBackend, EmbeddingError, get_embedder
from app.services.llm import (
    LLMBackend,
    LLMError,
    LLMTimeoutError,
    LLMUnavailableError,
    get_llm,
)

router = APIRouter()
CurrentUser = Annotated[User, Depends(get_current_user)]
Approver = Annotated[User, Depends(require_roles(MANAGER, ADMIN))]


@router.post("/agents/run", response_model=AgentWorkflowResponse)
def run_agent(
    payload: AgentRunRequest,
    user: CurrentUser,
    db: Session = Depends(get_db),
    embedder: EmbeddingBackend = Depends(get_embedder),
    llm: LLMBackend = Depends(get_llm),
) -> AgentWorkflowResponse:
    try:
        workflow = run_agent_workflow(
            db,
            embedder,
            llm,
            request=payload.request,
            top_k=payload.top_k,
            document_id=payload.document_id,
            owner_id=user.id,
        )
    except AgentNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to embed request",
        ) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The local language model timed out",
        ) from exc
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The local language model is unavailable",
        ) from exc
    except LLMError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The local language model failed",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent workflow failed",
        ) from exc
    return _to_response(workflow)


@router.get("/agents/{workflow_id}", response_model=AgentWorkflowResponse)
def inspect_agent(
    workflow_id: UUID,
    user: CurrentUser,
    db: Session = Depends(get_db),
) -> AgentWorkflowResponse:
    try:
        workflow = get_agent_workflow(db, workflow_id)
    except AgentNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workflow not found")
    if not can_read_workflow(user, workflow):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workflow not found")
    return _to_response(workflow)


@router.post("/agents/{workflow_id}/approve", response_model=AgentWorkflowResponse)
def approve_agent(
    workflow_id: UUID,
    user: Approver,
    payload: AgentDecisionRequest | None = None,
    db: Session = Depends(get_db),
    embedder: EmbeddingBackend = Depends(get_embedder),
    llm: LLMBackend = Depends(get_llm),
) -> AgentWorkflowResponse:
    _ = payload
    return _decide(db, embedder, llm, workflow_id, approve=True, actor_id=user.id)


@router.post("/agents/{workflow_id}/reject", response_model=AgentWorkflowResponse)
def reject_agent(
    workflow_id: UUID,
    user: Approver,
    payload: AgentDecisionRequest | None = None,
    db: Session = Depends(get_db),
    embedder: EmbeddingBackend = Depends(get_embedder),
    llm: LLMBackend = Depends(get_llm),
) -> AgentWorkflowResponse:
    _ = payload
    return _decide(db, embedder, llm, workflow_id, approve=False, actor_id=user.id)


def _decide(
    db: Session,
    embedder: EmbeddingBackend,
    llm: LLMBackend,
    workflow_id: UUID,
    *,
    approve: bool,
    actor_id: UUID,
) -> AgentWorkflowResponse:
    try:
        if approve:
            workflow = approve_agent_workflow(db, embedder, llm, workflow_id, actor_id=actor_id)
        else:
            workflow = reject_agent_workflow(db, embedder, llm, workflow_id, actor_id=actor_id)
    except AgentNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workflow not found")
    except AgentConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=exc.message) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The local language model timed out",
        ) from exc
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The local language model is unavailable",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent decision failed",
        ) from exc
    return _to_response(workflow)


def _to_response(workflow: AgentWorkflow) -> AgentWorkflowResponse:
    sources = []
    for item in workflow.sources or []:
        sources.append(
            RagSourceResponse(
                source_id=item["source_id"],
                document_id=item["document_id"],
                chunk_id=item["chunk_id"],
                chunk_index=item["chunk_index"],
                page_number=item.get("page_number"),
                similarity=item["similarity"],
            )
        )
    return AgentWorkflowResponse(
        workflow_id=workflow.id,
        request=workflow.request,
        owner_id=workflow.owner_id,
        document_id=workflow.document_id,
        status=workflow.status,
        intent=workflow.intent,
        answer=workflow.answer,
        draft=workflow.draft,
        sources=sources,
        retrieval_count=workflow.retrieval_count,
        grounded=workflow.grounded,
        citation_valid=workflow.citation_valid,
        abstained=workflow.abstained,
        approval_decision=workflow.approval_decision,
        decided_by_user_id=workflow.decided_by_user_id,
        execution_result=workflow.execution_result,
        model=settings.ollama_model,
        created_at=workflow.created_at,
        updated_at=workflow.updated_at,
        decided_at=workflow.decided_at,
        executed_at=workflow.executed_at,
        agent_trace=[AgentTraceStep.model_validate(item) for item in workflow.agent_trace or []],
    )
