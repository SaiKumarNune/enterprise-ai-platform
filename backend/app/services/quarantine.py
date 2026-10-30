from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.user import ADMIN
from app.services.audit import record_audit
from app.services.authorization import Principal, is_quarantined
from app.services.users import get_user_by_id


class QuarantineError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def list_quarantined_documents(db: Session, principal: Principal) -> list[Document]:
    if principal.role != ADMIN:
        raise QuarantineError("forbidden", "Insufficient permissions")
    if hasattr(db, "documents"):
        rows = [
            item
            for item in db.documents
            if item.tenant_id == principal.tenant_id and is_quarantined(item)
        ]
        rows.sort(key=lambda item: (item.created_at, item.id), reverse=True)
        return rows
    statement = (
        select(Document)
        .where(Document.tenant_id == principal.tenant_id, Document.owner_id.is_(None))
        .order_by(Document.created_at.desc(), Document.id.desc())
    )
    return list(db.scalars(statement).all())


def assign_quarantined_owner(
    db: Session,
    principal: Principal,
    *,
    document_id: UUID,
    owner_id: UUID,
) -> Document:
    if principal.role != ADMIN:
        raise QuarantineError("forbidden", "Insufficient permissions")
    document = db.get(Document, document_id)
    if document is None or document.tenant_id != principal.tenant_id:
        raise QuarantineError("not_found", "Document not found")
    if not is_quarantined(document):
        raise QuarantineError("conflict", "Document already has an owner")
    owner = get_user_by_id(db, owner_id)
    if owner is None or not owner.is_active or owner.tenant_id != principal.tenant_id:
        raise QuarantineError("not_found", "User not found")
    document.owner_id = owner.id
    record_audit(
        db,
        action="document.ownership.assign",
        outcome="success",
        principal=principal,
        resource_type="document",
        resource_id=document.id,
        details={"owner_id": str(owner.id)},
    )
    db.commit()
    db.refresh(document)
    return document
