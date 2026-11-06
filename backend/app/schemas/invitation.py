from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.security import is_valid_email, normalize_email


class CreateInvitationRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    email: str = Field(min_length=3, max_length=320)
    expires_in_hours: int = Field(default=72, ge=1, le=720)

    @field_validator("email")
    @classmethod
    def normalize(cls, value: str) -> str:
        email = normalize_email(value)
        if not is_valid_email(email):
            raise ValueError("must be a valid email address")
        return email


class CreatedInvitationResponse(BaseModel):
    id: UUID
    email: str
    tenant_id: UUID
    token: str
    expires_at: datetime
    created_at: datetime


class InvitationResponse(BaseModel):
    id: UUID
    email: str
    tenant_id: UUID
    expires_at: datetime
    consumed_at: datetime | None
    created_at: datetime
