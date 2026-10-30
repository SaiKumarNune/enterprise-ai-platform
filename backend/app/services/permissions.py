from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.acl import MANAGE, PRIVILEGES, DocumentACL
from app.models.document import Document
from app.services.audit import record_audit
from app.services.authorization import Principal, has_privilege
from app.services.users import get_user_by_id


class PermissionError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def list_grants(db: Session, document_id: UUID) -> list[DocumentACL]:
    if hasattr(db, "acls"):
        return [item for item in db.acls if item.document_id == document_id]
    return list(
        db.scalars(select(DocumentACL).where(DocumentACL.document_id == document_id)).all()
    )


def grant_privilege(
    db: Session,
    principal: Principal,
    *,
    document: Document,
    target_user_id: UUID,
    privilege: str,
) -> DocumentACL:
    if privilege not in PRIVILEGES:
        raise PermissionError("invalid_privilege", "Privilege must be read, write, or manage")
    if not has_privilege(db, principal, document, MANAGE):
        raise PermissionError("forbidden", "Insufficient permissions")
    if target_user_id == principal.user_id:
        raise PermissionError("escalation", "Users cannot grant themselves elevated privileges")
    target = get_user_by_id(db, target_user_id)
    if target is None or not target.is_active or target.tenant_id != principal.tenant_id:
        raise PermissionError("not_found", "User not found")
    if target.tenant_id != document.tenant_id:
        raise PermissionError("not_found", "User not found")
    existing = _grant_for(db, document.id, target_user_id)
    if existing is None:
        existing = DocumentACL(
            id=uuid4(),
            document_id=document.id,
            user_id=target_user_id,
            privilege=privilege,
            granted_by_user_id=principal.user_id,
        )
        db.add(existing)
    else:
        existing.privilege = privilege
        existing.granted_by_user_id = principal.user_id
    record_audit(
        db,
        action="permission.grant",
        outcome="success",
        principal=principal,
        resource_type="document",
        resource_id=document.id,
        details={"target_user_id": str(target_user_id), "privilege": privilege},
    )
    db.commit()
    db.refresh(existing)
    return existing


def revoke_privilege(
    db: Session,
    principal: Principal,
    *,
    document: Document,
    target_user_id: UUID,
) -> None:
    if not has_privilege(db, principal, document, MANAGE):
        raise PermissionError("forbidden", "Insufficient permissions")
    existing = _grant_for(db, document.id, target_user_id)
    if existing is None:
        raise PermissionError("not_found", "Permission grant not found")
    db.delete(existing)
    record_audit(
        db,
        action="permission.revoke",
        outcome="success",
        principal=principal,
        resource_type="document",
        resource_id=document.id,
        details={"target_user_id": str(target_user_id)},
    )
    db.commit()


def _grant_for(db: Session, document_id: UUID, user_id: UUID) -> DocumentACL | None:
    if hasattr(db, "acls"):
        for item in db.acls:
            if item.document_id == document_id and item.user_id == user_id:
                return item
        return None
    return db.scalar(
        select(DocumentACL).where(
            DocumentACL.document_id == document_id,
            DocumentACL.user_id == user_id,
        )
    )
