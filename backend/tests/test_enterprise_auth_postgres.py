"""PostgreSQL invitation redemption and identity-mapping checks.

Runs only with TEST_DATABASE_URL on an isolated *_test database.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.core.security import hash_password, normalize_email
from app.models.invitation import Invitation
from app.models.tenant import Tenant
from app.models.user import ADMIN, User
from app.services.invitations import InvitationError, consume_invitation, create_invitation
from tests.postgres_isolation import SKIP_ISOLATED_POSTGRES, postgres_client as postgres_client

pytestmark = SKIP_ISOLATED_POSTGRES

PASSWORD = "correct-horse-battery"


def _seed_admin(session: Session) -> User:
    now = datetime.now(timezone.utc)
    tenant = Tenant(
        id=uuid4(),
        name="Auth Isolation",
        slug=f"auth-iso-{uuid4().hex[:8]}",
        created_at=now,
        updated_at=now,
    )
    session.add(tenant)
    session.commit()
    admin = User(
        id=uuid4(),
        email=normalize_email(f"admin-{uuid4().hex[:8]}@example.test"),
        tenant_id=tenant.id,
        password_hash=hash_password(PASSWORD),
        role=ADMIN,
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    session.add(admin)
    session.commit()
    session.refresh(admin)
    return admin


def test_postgres_concurrent_invitation_redemption(
    postgres_client: tuple[TestClient, sessionmaker[Session]],
) -> None:
    _client, factory = postgres_client
    setup = factory()
    try:
        admin = _seed_admin(setup)
        _invitation, token = create_invitation(setup, actor=admin, email="race@example.test", ttl_hours=1)
        invitation_id = _invitation.id
        email = "race@example.test"
    finally:
        setup.close()

    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    lock = threading.Lock()

    def redeem() -> None:
        session = factory()
        try:
            barrier.wait()
            try:
                consume_invitation(session, email=email, token=token, commit=True)
                with lock:
                    outcomes.append("won")
            except InvitationError:
                session.rollback()
                with lock:
                    outcomes.append("lost")
        finally:
            session.close()

    threads = [threading.Thread(target=redeem) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert outcomes.count("won") == 1
    assert outcomes.count("lost") == 1
    verify = factory()
    try:
        invitation = verify.get(Invitation, invitation_id)
        assert invitation is not None
        assert invitation.consumed_at is not None
    finally:
        verify.close()
