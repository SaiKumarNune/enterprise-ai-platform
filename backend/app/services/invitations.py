from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.security import hash_secret, is_valid_email, normalize_email
from app.models.invitation import Invitation
from app.models.user import User


class InvitationError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def create_invitation(
    db: Session,
    *,
    actor: User,
    email: str,
    ttl_hours: int = 72,
) -> tuple[Invitation, str]:
    normalized = normalize_email(email)
    if not is_valid_email(normalized):
        raise InvitationError("invalid_email", "must be a valid email address")
    if ttl_hours < 1 or ttl_hours > 24 * 30:
        raise InvitationError("invalid_ttl", "expires_in_hours must be between 1 and 720")
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    invitation = Invitation(
        id=uuid4(),
        tenant_id=actor.tenant_id,
        email=normalized,
        token_hash=hash_secret(token),
        created_by=actor.id,
        expires_at=now + timedelta(hours=ttl_hours),
        created_at=now,
    )
    db.add(invitation)
    db.commit()
    db.refresh(invitation)
    return invitation, token


def list_invitations(db: Session, *, tenant_id: UUID) -> list[Invitation]:
    rows = db.scalars(
        select(Invitation)
        .where(Invitation.tenant_id == tenant_id)
        .order_by(Invitation.created_at.desc())
    ).all()
    return list(rows)


def find_open_invitation(db: Session, *, email: str) -> Invitation | None:
    now = datetime.now(timezone.utc)
    return db.scalar(
        select(Invitation)
        .where(Invitation.email == normalize_email(email))
        .where(Invitation.consumed_at.is_(None))
        .where(Invitation.expires_at > now)
        .order_by(Invitation.created_at.desc())
    )


def _claim_row(
    db: Session,
    invitation: Invitation,
    *,
    email: str,
    user_id: UUID | None,
    now: datetime,
) -> None:
    result = db.execute(
        update(Invitation)
        .where(
            Invitation.id == invitation.id,
            Invitation.consumed_at.is_(None),
            Invitation.expires_at > now,
            Invitation.email == normalize_email(email),
        )
        .values(consumed_at=now, consumed_by=user_id)
    )
    if result.rowcount != 1:
        raise InvitationError("consumed", "Invitation is required or invalid")
    invitation.consumed_at = now
    invitation.consumed_by = user_id


def consume_invitation(
    db: Session,
    *,
    email: str,
    token: str,
    user_id: UUID | None = None,
    commit: bool = True,
) -> Invitation:
    now = datetime.now(timezone.utc)
    try:
        invitation = db.scalar(
            select(Invitation)
            .where(Invitation.token_hash == hash_secret(token))
            .with_for_update()
        )
        if invitation is None:
            raise InvitationError("invalid", "Invitation is required or invalid")
        if invitation.consumed_at is not None:
            raise InvitationError("consumed", "Invitation is required or invalid")
        if _aware(invitation.expires_at) <= now:
            raise InvitationError("expired", "Invitation is required or invalid")
        if invitation.email != normalize_email(email):
            raise InvitationError("mismatch", "Invitation is required or invalid")
        _claim_row(db, invitation, email=email, user_id=user_id, now=now)
        if commit:
            db.commit()
            db.refresh(invitation)
        return invitation
    except Exception:
        if commit:
            db.rollback()
        raise


def claim_open_invitation(
    db: Session,
    *,
    email: str,
    user_id: UUID | None = None,
    commit: bool = True,
) -> Invitation:
    now = datetime.now(timezone.utc)
    try:
        invitation = db.scalar(
            select(Invitation)
            .where(Invitation.email == normalize_email(email))
            .where(Invitation.consumed_at.is_(None))
            .where(Invitation.expires_at > now)
            .order_by(Invitation.created_at.desc())
            .with_for_update()
        )
        if invitation is None:
            raise InvitationError("invalid", "Invitation is required or invalid")
        _claim_row(db, invitation, email=email, user_id=user_id, now=now)
        if commit:
            db.commit()
            db.refresh(invitation)
        return invitation
    except Exception:
        if commit:
            db.rollback()
        raise
