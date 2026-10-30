from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.core.security import create_access_token, hash_password, normalize_email
from app.models.tenant import Tenant
from app.models.user import EMPLOYEE, User
from tests.test_documents import FakeSession

_counter = 0


def seed_tenant(session: FakeSession, *, name: str = "Acme") -> Tenant:
    tenant = Tenant(
        id=uuid.uuid4(),
        name=name,
        slug=f"{name.lower().replace(' ', '-')}-{uuid.uuid4().hex[:8]}",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    session.add(tenant)
    return tenant


def seed_user(
    session: FakeSession,
    *,
    role: str = EMPLOYEE,
    email: str | None = None,
    password: str = "correct-horse-battery",
    is_active: bool = True,
    tenant_id=None,
) -> User:
    global _counter
    _counter += 1
    now = datetime.now(timezone.utc)
    if tenant_id is None:
        tenant_id = seed_tenant(session).id
    user = User(
        id=uuid.uuid4(),
        email=normalize_email(email or f"{role}{_counter}@example.com"),
        tenant_id=tenant_id,
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


def own_session_documents(session: FakeSession, user: User) -> None:
    for document in session.documents:
        if getattr(document, "owner_id", None) is None:
            document.owner_id = user.id
        if getattr(document, "tenant_id", None) is None:
            document.tenant_id = user.tenant_id
    for workflow in session.workflows:
        if getattr(workflow, "tenant_id", None) is None:
            workflow.tenant_id = user.tenant_id


def seed_agent_users(session: FakeSession) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    tenant = seed_tenant(session, name="AgentCo")
    employee = seed_user(session, role="employee", email="employee@example.com", tenant_id=tenant.id)
    manager = seed_user(session, role="manager", email="manager@example.com", tenant_id=tenant.id)
    admin = seed_user(session, role="admin", email="admin@example.com", tenant_id=tenant.id)
    own_session_documents(session, employee)
    return bearer_headers(employee), bearer_headers(manager), bearer_headers(admin)


def auth_headers(session: FakeSession, *, role: str = EMPLOYEE) -> tuple[User, dict[str, str]]:
    user = seed_user(session, role=role)
    own_session_documents(session, user)
    return user, bearer_headers(user)
