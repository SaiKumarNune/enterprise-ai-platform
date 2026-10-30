from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.security import is_valid_email, normalize_email


class RegisterRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=10, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize(cls, value: str) -> str:
        email = normalize_email(value)
        if not is_valid_email(email):
            raise ValueError("must be a valid email address")
        return email

    @field_validator("password")
    @classmethod
    def reject_blank_password(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("must not be blank or padded with whitespace")
        return value


class LoginRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize(cls, value: str) -> str:
        return normalize_email(value)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    id: UUID
    email: str
    role: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
