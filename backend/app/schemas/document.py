from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

DocumentStatus = Literal["uploaded", "processing", "ready", "failed"]


def _reject_blank(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("must not be blank")
    return stripped


class DocumentCreate(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(min_length=1, max_length=100)

    @field_validator("filename", "content_type")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        return _reject_blank(value)


class DocumentUpdate(BaseModel):
    filename: str | None = Field(default=None, min_length=1, max_length=255)
    content_type: str | None = Field(default=None, min_length=1, max_length=100)
    status: DocumentStatus | None = None

    @field_validator("filename", "content_type")
    @classmethod
    def reject_blank_values(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _reject_blank(value)


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    content_type: str
    status: str
    created_at: datetime
    updated_at: datetime
    storage_path: str | None = None
