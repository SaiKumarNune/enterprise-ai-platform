"""Optional PostgreSQL ACL checks.

These tests run only when TEST_DATABASE_URL points at an isolated database.
They never use the development DATABASE_URL, Docker volumes, or .env values
for the live enterprise_ai database. Unset TEST_DATABASE_URL skips them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.security import create_access_token, hash_password, normalize_email
from app.models.acl import DocumentACL
from app.models.audit import AuditEvent
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.tenant import Tenant
from app.models.user import ADMIN, EMPLOYEE, User
from app.models.workflow import AgentWorkflow
from tests.postgres_isolation import SKIP_ISOLATED_POSTGRES, postgres_client as postgres_client

pytestmark = SKIP_ISOLATED_POSTGRES


def _seed_user(session: Session, tenant_id, *, role: str, email: str) -> User:
    now = datetime.now(timezone.utc)
    user = User(
        id=uuid4(),
        email=normalize_email(email),
        tenant_id=tenant_id,
        password_hash=hash_password("pytest-only-acl-password"),
        role=role,
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def _headers(user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user.id)}"}


def test_postgres_admin_grant_and_revoke_uses_real_acl_rows(
    postgres_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    client, factory = postgres_client
    session = factory()
    try:
        now = datetime.now(timezone.utc)
        tenant = Tenant(
            id=uuid4(),
            name="ACL Test",
            slug=f"acl-test-{uuid4().hex[:8]}",
            created_at=now,
            updated_at=now,
        )
        session.add(tenant)
        session.commit()
        admin = _seed_user(session, tenant.id, role=ADMIN, email="acl-admin@example.test")
        employee = _seed_user(session, tenant.id, role=EMPLOYEE, email="acl-employee@example.test")
        document = Document(
            id=uuid4(),
            filename="policy.txt",
            content_type="text/plain",
            status="uploaded",
            tenant_id=tenant.id,
            owner_id=admin.id,
            created_at=now,
            updated_at=now,
        )
        session.add(document)
        session.commit()
        document_id = document.id
        employee_id = employee.id
        admin_headers = _headers(admin)
        employee_headers = _headers(employee)
    finally:
        session.close()

    denied = client.get(f"/documents/{document_id}", headers=employee_headers)
    granted = client.put(
        f"/documents/{document_id}/permissions",
        json={"user_id": str(employee_id), "privilege": "read"},
        headers=admin_headers,
    )
    allowed = client.get(f"/documents/{document_id}", headers=employee_headers)
    revoked = client.delete(
        f"/documents/{document_id}/permissions/{employee_id}",
        headers=admin_headers,
    )
    hidden = client.get(f"/documents/{document_id}", headers=employee_headers)

    assert denied.status_code == 404
    assert granted.status_code == 200
    assert allowed.status_code == 200
    assert revoked.status_code == 204
    assert hidden.status_code == 404

    verify = factory()
    try:
        grants = list(
            verify.scalars(select(DocumentACL).where(DocumentACL.document_id == document_id)).all()
        )
        audits = list(
            verify.scalars(
                select(AuditEvent).where(AuditEvent.resource_id == document_id)
            ).all()
        )
        verify.execute(text("SELECT 1"))
        _ = (DocumentChunk, AgentWorkflow)
    finally:
        verify.close()

    assert grants == []
    assert {event.action for event in audits} >= {"permission.grant", "permission.revoke"}
