from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

Privilege = Literal["read", "write", "manage"]


class PermissionGrantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: UUID
    privilege: Privilege


class PermissionGrantResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: UUID
    privilege: str
    granted_by_user_id: UUID | None = None
    created_at: datetime | None = None
