from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class QuarantineDocumentResponse(BaseModel):
    """Metadata-only view. Content, chunks, and storage paths are omitted."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    content_type: str
    status: str
    tenant_id: UUID
    created_at: datetime
    updated_at: datetime


class AssignOwnerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: UUID
