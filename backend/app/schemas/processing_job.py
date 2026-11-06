from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ProcessingJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    tenant_id: UUID
    requested_by: UUID
    status: str
    attempt_count: int
    max_attempts: int
    available_at: datetime
    last_error: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
