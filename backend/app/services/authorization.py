from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import and_, exists, false, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.sql import ColumnElement

from app.models.acl import MANAGE, PRIVILEGE_RANK, READ, WRITE, DocumentACL
from app.models.document import Document
from app.models.user import ADMIN, APPROVER_ROLES, User
from app.models.workflow import AgentWorkflow


class DocumentAccessDenied(Exception):
    """Caller cannot access the document. Map to HTTP 404."""


@dataclass(frozen=True)
class Principal:
    user_id: UUID
    tenant_id: UUID
    role: str
    is_active: bool

    @classmethod
    def from_user(cls, user: User) -> Principal:
        if user.tenant_id is None:
            raise DocumentAccessDenied("user has no tenant")
        return cls(
            user_id=user.id,
            tenant_id=user.tenant_id,
            role=user.role,
            is_active=bool(user.is_active),
        )


def privileges_meeting(minimum: str) -> tuple[str, ...]:
    needed = PRIVILEGE_RANK[minimum]
    return tuple(name for name, rank in PRIVILEGE_RANK.items() if rank >= needed)


def acl_privilege(db: Session, document_id: UUID, user_id: UUID) -> str | None:
    if hasattr(db, "acls"):
        for item in db.acls:
            if item.document_id == document_id and item.user_id == user_id:
                return item.privilege
        return None
    return db.scalar(
        select(DocumentACL.privilege).where(
            DocumentACL.document_id == document_id,
            DocumentACL.user_id == user_id,
        )
    )


def is_quarantined(document: Document) -> bool:
    return document.owner_id is None


def effective_privilege(db: Session, principal: Principal, document: Document) -> str | None:
    if not principal.is_active:
        return None
    if document.tenant_id is None or document.tenant_id != principal.tenant_id:
        return None
    if is_quarantined(document):
        return None
    if document.owner_id == principal.user_id:
        return MANAGE
    if principal.role == ADMIN:
        return MANAGE
    granted = acl_privilege(db, document.id, principal.user_id)
    return granted


def has_privilege(db: Session, principal: Principal, document: Document, minimum: str) -> bool:
    actual = effective_privilege(db, principal, document)
    if actual is None:
        return False
    return PRIVILEGE_RANK[actual] >= PRIVILEGE_RANK[minimum]


def require_document(
    db: Session,
    principal: Principal,
    document_id: UUID,
    minimum: str = READ,
) -> Document:
    document = db.get(Document, document_id)
    if document is None or not has_privilege(db, principal, document, minimum):
        raise DocumentAccessDenied("document")
    return document


def authorized_document_ids(db: Session, principal: Principal, minimum: str = READ) -> set[UUID]:
    if hasattr(db, "documents"):
        return {
            item.id
            for item in db.documents
            if has_privilege(db, principal, item, minimum)
        }
    rows = db.scalars(select(Document.id).where(authorized_document_clause(principal, minimum)))
    return set(rows.all())


def authorized_document_clause(principal: Principal, minimum: str = READ) -> ColumnElement[bool]:
    allowed = privileges_meeting(minimum)
    shared = exists(
        select(DocumentACL.id).where(
            DocumentACL.document_id == Document.id,
            DocumentACL.user_id == principal.user_id,
            DocumentACL.privilege.in_(allowed),
        )
    )
    owner_or_share = or_(Document.owner_id == principal.user_id, shared)
    if principal.role == ADMIN:
        owner_or_share = or_(owner_or_share, Document.tenant_id == principal.tenant_id)
    if not principal.is_active:
        return false()
    return and_(
        Document.tenant_id == principal.tenant_id,
        Document.owner_id.is_not(None),
        owner_or_share,
    )


def can_read_workflow(user: User, workflow: AgentWorkflow) -> bool:
    if not user.is_active or user.tenant_id is None or workflow.tenant_id is None:
        return False
    if workflow.tenant_id != user.tenant_id:
        return False
    if user.role in APPROVER_ROLES:
        return True
    if workflow.owner_id is None:
        return False
    return workflow.owner_id == user.id


def can_decide_workflow(user: User, workflow: AgentWorkflow) -> bool:
    if not can_read_workflow(user, workflow):
        return False
    return user.role in APPROVER_ROLES
