from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

RetrievalMode = Literal["semantic", "lexical", "hybrid"]


class SearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: UUID | None = None
    retrieval_mode: RetrievalMode | None = None
    include_diagnostics: bool = False

    @field_validator("query")
    @classmethod
    def reject_blank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class SearchHit(BaseModel):
    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    page_number: int | None
    similarity: float
    retrieval_mode: RetrievalMode | None = None
