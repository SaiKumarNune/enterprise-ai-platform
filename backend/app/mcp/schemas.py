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
