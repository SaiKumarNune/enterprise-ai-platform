from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from app.models.audit import AuditEvent
from app.services.authorization import Principal


def record_audit(
    db: Session,
    *,
    action: str,
    outcome: str,
    principal: Principal | None = None,
    resource_type: str | None = None,
    resource_id: UUID | None = None,
    details: dict | None = None,
) -> None:
    event = AuditEvent(
        id=uuid4(),
        actor_user_id=principal.user_id if principal else None,
        actor_tenant_id=principal.tenant_id if principal else None,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        outcome=outcome,
        details=details,
    )
    db.add(event)
