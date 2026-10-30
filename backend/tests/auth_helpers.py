from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.core.security import create_access_token, hash_password, normalize_email
from app.models.user import EMPLOYEE, User
from tests.test_documents import FakeSession

_counter = 0


def seed_user(
    session: FakeSession,
    *,
    role: str = EMPLOYEE,
    email: str | None = None,
    password: str = "correct-horse-battery",
    is_active: bool = True,
) -> User:
    global _counter
    _counter += 1
    now = datetime.now(timezone.utc)
    user = User(
        id=uuid.uuid4(),
        email=normalize_email(email or f"{role}{_counter}@example.com"),
        password_hash=hash_password(password),
        role=role,
        is_active=is_active,
        created_at=now,
        updated_at=now,
    )
    session.add(user)
    return user


def bearer_headers(user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user.id)}"}


def seed_agent_users(session: FakeSession) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    employee = seed_user(session, role="employee", email="employee@example.com")
    manager = seed_user(session, role="manager", email="manager@example.com")
    admin = seed_user(session, role="admin", email="admin@example.com")
    return bearer_headers(employee), bearer_headers(manager), bearer_headers(admin)
