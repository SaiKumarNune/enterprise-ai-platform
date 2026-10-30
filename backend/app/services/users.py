from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import hash_password, normalize_email
from app.models.user import EMPLOYEE, User
from app.services.legacy_backfill import GREENFIELD_TENANT_ID


class DuplicateUserError(Exception):
    """Email is already registered."""


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.scalar(select(User).where(User.email == normalize_email(email)))


def get_user_by_id(db: Session, user_id: UUID) -> User | None:
    return db.get(User, user_id)


def _registration_tenant_id() -> UUID:
    raw = settings.default_tenant_id.strip()
    if raw:
        return UUID(raw)
    return GREENFIELD_TENANT_ID


def create_employee(db: Session, *, email: str, password: str) -> User:
    user = User(
        id=uuid4(),
        email=normalize_email(email),
        tenant_id=_registration_tenant_id(),
        password_hash=hash_password(password),
        role=EMPLOYEE,
        is_active=True,
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise DuplicateUserError from exc
    db.refresh(user)
    return user


def upsert_admin(db: Session, *, email: str, password: str) -> User:
    """Development-only helper. Never expose over HTTP."""
    from app.models.user import ADMIN

    existing = get_user_by_email(db, email)
    if existing is None:
        user = User(
            id=uuid4(),
            email=normalize_email(email),
            tenant_id=_registration_tenant_id(),
            password_hash=hash_password(password),
            role=ADMIN,
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    existing.password_hash = hash_password(password)
    existing.role = ADMIN
    existing.is_active = True
    db.commit()
    db.refresh(existing)
    return existing
