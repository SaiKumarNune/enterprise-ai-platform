import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.core.security import create_access_token
from app.main import app
from app.models.workflow import AgentWorkflow
from app.services.agent_tools import NOT_PROVIDED
from app.services.embeddings import get_embedder
from app.services.llm import get_llm
from tests.auth_helpers import bearer_headers, own_session_documents, seed_tenant, seed_user
from tests.fake_embeddings import HashEmbedder
from tests.fake_llm import SequenceLLM
from tests.test_agents import DRAFT, PLAN_DRAFT, STRICT_VACATION_POLICY, _session_with_policy
from tests.test_documents import FakeSession, _clear_overrides, _override_db

client = TestClient(app)

PASSWORD = "correct-horse-battery"


def _team(session, *roles: str):
    tenant = seed_tenant(session)
    users = []
    for role in roles:
        users.append(
            seed_user(
                session,
                role=role,
                email=f"{role}-{uuid.uuid4().hex[:8]}@example.com",
                tenant_id=tenant.id,
            )
        )
    own_session_documents(session, users[0])
    return tenant, users


def _override_llm() -> None:
    app.dependency_overrides[get_llm] = lambda: SequenceLLM([PLAN_DRAFT, DRAFT])
    app.dependency_overrides[get_embedder] = lambda: HashEmbedder()


def test_register_login_and_me() -> None:
    session = FakeSession()
    _override_db(session)
    try:
        created = client.post(
            "/auth/register",
            json={"email": "Alex@Example.com", "password": PASSWORD, "role": "admin"},
        )
        login = client.post("/auth/login", json={"email": "alex@example.com", "password": PASSWORD})
        me = client.get("/auth/me", headers={"Authorization": f"Bearer {login.json()['access_token']}"})
    finally:
        _clear_overrides()

    assert created.status_code == 201
    assert created.json()["email"] == "alex@example.com"
    assert created.json()["role"] == "employee"
    assert "password_hash" not in created.json()
    assert login.status_code == 200
    assert login.json()["token_type"] == "bearer"
    assert me.status_code == 200
    assert me.json()["id"] == created.json()["id"]
    assert me.json()["role"] == "employee"


def test_register_rejects_duplicate_email() -> None:
    session = FakeSession()
    _override_db(session)
    try:
        first = client.post("/auth/register", json={"email": "dup@example.com", "password": PASSWORD})
        second = client.post("/auth/register", json={"email": "DUP@example.com", "password": PASSWORD})
    finally:
        _clear_overrides()

    assert first.status_code == 201
    assert second.status_code == 409


def test_login_wrong_password_does_not_reveal_account() -> None:
    session = FakeSession()
    seed_user(session, email="hidden@example.com", password=PASSWORD)
    _override_db(session)
    try:
        missing = client.post("/auth/login", json={"email": "nobody@example.com", "password": PASSWORD})
        wrong = client.post("/auth/login", json={"email": "hidden@example.com", "password": "wrong-password-1"})
    finally:
        _clear_overrides()

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert missing.json()["detail"] == wrong.json()["detail"] == "Invalid email or password"


def test_me_rejects_missing_invalid_expired_and_disabled_tokens() -> None:
    session = FakeSession()
    active = seed_user(session, email="active@example.com")
    disabled = seed_user(session, email="disabled@example.com", is_active=False)
    _override_db(session)
    expired = create_access_token(active.id, expires_delta=timedelta(seconds=-5))
    try:
        missing = client.get("/auth/me")
        invalid = client.get("/auth/me", headers={"Authorization": "Bearer not-a-token"})
        expired_resp = client.get("/auth/me", headers={"Authorization": f"Bearer {expired}"})
        disabled_resp = client.get("/auth/me", headers=bearer_headers(disabled))
    finally:
        _clear_overrides()

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert expired_resp.status_code == 401
    assert disabled_resp.status_code == 401


def test_agent_run_requires_authentication() -> None:
    session = FakeSession()
    _override_db(session)
    try:
        response = client.post("/agents/run", json={"request": "How many vacation days?"})
    finally:
        _clear_overrides()

    assert response.status_code == 401


def test_employee_cannot_approve_or_reject() -> None:
    session, _document = _session_with_policy()
    _tenant, (employee,) = _team(session, "employee")
    _override_llm()
    _override_db(session)
    headers = bearer_headers(employee)
    try:
        started = client.post(
            "/agents/run",
            json={"request": "Prepare an approval request for three days off."},
            headers=headers,
        )
        workflow_id = started.json()["workflow_id"]
        approved = client.post(f"/agents/{workflow_id}/approve", json={}, headers=headers)
        rejected = client.post(f"/agents/{workflow_id}/reject", json={}, headers=headers)
    finally:
        _clear_overrides()

    assert started.status_code == 200
    assert approved.status_code == 403
    assert rejected.status_code == 403


def test_manager_approval_records_actor_and_duplicate_conflicts() -> None:
    session, _document = _session_with_policy()
    _tenant, (employee, manager) = _team(session, "employee", "manager")
    _override_llm()
    _override_db(session)
    try:
        started = client.post(
            "/agents/run",
            json={"request": "Prepare an approval request for three days off."},
            headers=bearer_headers(employee),
        )
        workflow_id = started.json()["workflow_id"]
        first = client.post(
            f"/agents/{workflow_id}/approve",
            json={},
            headers=bearer_headers(manager),
        )
        second = client.post(
            f"/agents/{workflow_id}/approve",
            json={},
            headers=bearer_headers(manager),
        )
    finally:
        _clear_overrides()

    assert started.json()["owner_id"] == str(employee.id)
    assert first.status_code == 200
    assert first.json()["decided_by_user_id"] == str(manager.id)
    assert first.json()["approval_decision"] == "approved"
    assert first.json()["decided_at"]
    assert second.status_code == 409


def test_admin_can_reject() -> None:
    session, _document = _session_with_policy()
    _tenant, (employee, admin) = _team(session, "employee", "admin")
    _override_llm()
    _override_db(session)
    try:
        started = client.post(
            "/agents/run",
            json={"request": "Prepare an approval request for three days off."},
            headers=bearer_headers(employee),
        )
        rejected = client.post(
            f"/agents/{started.json()['workflow_id']}/reject",
            json={},
            headers=bearer_headers(admin),
        )
    finally:
        _clear_overrides()

    assert rejected.status_code == 200
    assert rejected.json()["approval_decision"] == "rejected"
    assert rejected.json()["decided_by_user_id"] == str(admin.id)


def test_employee_cannot_read_others_or_historical_null_owner_workflows() -> None:
    session, _document = _session_with_policy()
    tenant, (owner, other, manager) = _team(session, "employee", "employee", "manager")
    now = datetime.now(timezone.utc)
    historical = AgentWorkflow(
        id=uuid.uuid4(),
        request="Historical leave draft",
        owner_id=None,
        tenant_id=tenant.id,
        status="pending_approval",
        intent="draft_action",
        created_at=now,
        updated_at=now,
    )
    session.add(historical)
    _override_llm()
    _override_db(session)
    try:
        started = client.post(
            "/agents/run",
            json={"request": "Prepare an approval request for three days off."},
            headers=bearer_headers(owner),
        )
        workflow_id = started.json()["workflow_id"]
        other_get = client.get(f"/agents/{workflow_id}", headers=bearer_headers(other))
        owner_get = client.get(f"/agents/{workflow_id}", headers=bearer_headers(owner))
        manager_get = client.get(f"/agents/{workflow_id}", headers=bearer_headers(manager))
        hist_employee = client.get(f"/agents/{historical.id}", headers=bearer_headers(other))
        hist_manager = client.get(f"/agents/{historical.id}", headers=bearer_headers(manager))
    finally:
        _clear_overrides()

    assert other_get.status_code == 404
    assert owner_get.status_code == 200
    assert manager_get.status_code == 200
    assert hist_employee.status_code == 404
    assert hist_manager.status_code == 200


def test_stale_invalid_approval_still_fails_for_manager() -> None:
    session, document = _session_with_policy(content=STRICT_VACATION_POLICY)
    _tenant, (manager,) = _team(session, "manager")
    now = datetime.now(timezone.utc)
    workflow = AgentWorkflow(
        id=uuid.uuid4(),
        request="Review the vacation policy and prepare an approval request for 25 days off.",
        owner_id=manager.id,
        tenant_id=manager.tenant_id,
        document_id=document.id,
        status="pending_approval",
        intent="draft_action",
        draft={"days": 25, "reason": NOT_PROVIDED, "start_date": NOT_PROVIDED, "source_ids": ["S1"]},
        sources=[
            {
                "source_id": "S1",
                "document_id": str(document.id),
                "chunk_id": str(session.chunks[0].id),
                "chunk_index": 0,
                "page_number": 1,
                "similarity": 0.9,
            }
        ],
        retrieval_count=1,
        created_at=now,
        updated_at=now,
        graph_state={
            "request": "Review the vacation policy and prepare an approval request for 25 days off.",
            "intent": "draft_action",
            "excerpts": [{"source_id": "S1", "content": STRICT_VACATION_POLICY}],
            "sources": [
                {
                    "source_id": "S1",
                    "document_id": str(document.id),
                    "chunk_id": str(session.chunks[0].id),
                    "chunk_index": 0,
                    "page_number": 1,
                    "similarity": 0.9,
                }
            ],
            "draft": {"days": 25, "reason": NOT_PROVIDED, "start_date": NOT_PROVIDED, "source_ids": ["S1"]},
            "status": "pending_approval",
        },
    )
    session.add(workflow)
    _override_db(session)
    try:
        response = client.post(
            f"/agents/{workflow.id}/approve",
            json={},
            headers=bearer_headers(manager),
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200
    assert response.json()["status"] == "needs_clarification"
    assert response.json()["execution_result"] is None
    assert response.json()["approval_decision"] is None


def test_login_fails_closed_when_jwt_secret_is_insecure(monkeypatch) -> None:
    from app.core import security as security_module
    from app.core.config import settings

    session = FakeSession()
    seed_user(session, email="jwt-fail@example.com", password=PASSWORD)
    _override_db(session)
    monkeypatch.setattr(security_module, "settings", replace(settings, jwt_secret="short"))
    try:
        response = client.post(
            "/auth/login",
            json={"email": "jwt-fail@example.com", "password": PASSWORD},
        )
    finally:
        _clear_overrides()

    assert response.status_code == 503
