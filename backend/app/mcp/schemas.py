"""Typed MCP tool input and output schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class KnowledgeSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: str | None = None

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class KnowledgeSearchExcerpt(BaseModel):
    source_id: str
    content: str


class KnowledgeSearchSource(BaseModel):
    source_id: str
    document_id: str
    chunk_id: str
    chunk_index: int
    page_number: int | None = None
    similarity: float


class KnowledgeSearchOutput(BaseModel):
    excerpts: list[KnowledgeSearchExcerpt]
    sources: list[KnowledgeSearchSource]
    retrieval_count: int
    selected_count: int


class PrepareApprovalInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    request: str = Field(min_length=1, max_length=2000)
    raw_draft: str = Field(min_length=1, max_length=8000)
    valid_source_ids: list[str] = Field(default_factory=list)
    action_type: str | None = "leave_request"


class PrepareApprovalOutput(BaseModel):
    draft: dict[str, Any] | None = None


class PrepareTicketInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    request: str = Field(min_length=1, max_length=2000)
    raw_draft: str = Field(min_length=1, max_length=8000)
    valid_source_ids: list[str] = Field(default_factory=list)


class PrepareTicketOutput(BaseModel):
    draft: dict[str, Any] | None = None


class CreateTicketInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=1, max_length=4000)
    priority: str = Field(default="medium", max_length=20)

    @field_validator("title", "description")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return " ".join(value.split())

    @field_validator("priority")
    @classmethod
    def allowlisted_priority(cls, value: str) -> str:
        priority = value.strip().lower()
        if priority not in {"low", "medium", "high", "urgent"}:
            raise ValueError("priority is not allowed")
        return priority


class CreateTicketOutput(BaseModel):
    simulated: bool
    created: bool
    action_type: str
    ticket_id: str
    tenant_id: str
    requester_user_id: str
    title: str
    priority: str
    status: str
    idempotency_key: str
    message: str
