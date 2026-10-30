from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.rag import RagSourceResponse


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    request: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: UUID | None = None

    @field_validator("request")
    @classmethod
    def reject_blank_request(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class AgentDecisionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    comment: str | None = Field(default=None, max_length=1000)


class AgentWorkflowResponse(BaseModel):
    workflow_id: UUID
    request: str
    document_id: UUID | None
    status: str
    intent: str | None
    answer: str | None
    draft: dict[str, Any] | None
    sources: list[RagSourceResponse]
    retrieval_count: int
    grounded: bool | None
    citation_valid: bool | None
    abstained: bool | None
    approval_decision: str | None
    execution_result: dict[str, Any] | None
    model: str
    created_at: datetime
    updated_at: datetime
    decided_at: datetime | None
    executed_at: datetime | None
