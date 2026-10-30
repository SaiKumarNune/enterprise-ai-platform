from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RagAskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: UUID | None = None

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class RagSourceResponse(BaseModel):
    source_id: str
    document_id: UUID
    chunk_id: UUID
    chunk_index: int
    page_number: int | None
    similarity: float


class RagAskResponse(BaseModel):
    answer: str
    sources: list[RagSourceResponse]
    retrieval_count: int
    grounded: bool
    model: str
    citation_valid: bool = True
    abstained: bool = False
