"""HTTP authorization and document ACL tests.

These tests drive FastAPI with TestClient and an in-memory FakeSession.
They do not connect to PostgreSQL and do not prove live SQL JOIN behavior.
Test users, tenants, and documents are created in-process with pytest-only
credentials; production UUIDs and passwords are not used.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.models.chunk import DocumentChunk
from app.models.user import ADMIN, EMPLOYEE, MANAGER
from app.services.embeddings import get_embedder
from app.services.llm import get_llm
from tests.auth_helpers import bearer_headers, seed_tenant, seed_user
from tests.fake_embeddings import HashEmbedder
from tests.fake_llm import SequenceLLM
from tests.test_documents import FakeSession, _clear_overrides, _override_db, _sample_document

client = TestClient(app)

TEST_PASSWORD = "pytest-only-acl-password"


def _chunk(document_id, content: str) -> DocumentChunk:
    return DocumentChunk(
        id=uuid.uuid4(),
        document_id=document_id,
        chunk_index=0,
        content=content,
        page_number=1,
        created_at=datetime.now(timezone.utc),
        embedding=HashEmbedder().embed_texts([content])[0],
    )


def _owned_document(session: FakeSession, owner, filename: str = "admin-owned.txt"):
    document = _sample_document(filename)
    document.owner_id = owner.id
    document.tenant_id = owner.tenant_id
    session.add(document)
    return document


def test_unauthenticated_document_acl_requests_return_401() -> None:
    session = FakeSession()
    document_id = uuid.uuid4()
    user_id = uuid.uuid4()
    _override_db(session)
    try:
        responses = [
            client.get("/documents"),
            client.get(f"/documents/{document_id}"),
            client.patch(f"/documents/{document_id}", json={"filename": "renamed.txt"}),
            client.get(f"/documents/{document_id}/permissions"),
            client.put(
                f"/documents/{document_id}/permissions",
                json={"user_id": str(user_id), "privilege": "read"},
            ),
            client.delete(f"/documents/{document_id}/permissions/{user_id}"),
            client.delete(f"/documents/{document_id}"),
            client.post("/search", json={"query": "policy"}),
            client.post("/rag/ask", json={"question": "How many days?"}),
        ]
    finally:
        _clear_overrides()

    assert {item.status_code for item in responses} == {401}


def test_employee_cannot_read_admin_owned_document_without_acl() -> None:
    session = FakeSession()
    tenant = seed_tenant(session)
    admin = seed_user(session, role=ADMIN, tenant_id=tenant.id, password=TEST_PASSWORD)
    employee = seed_user(session, role=EMPLOYEE, tenant_id=tenant.id, password=TEST_PASSWORD)
    document = _owned_document(session, admin, "secret-policy.txt")
    _override_db(session)
    try:
        listed = client.get("/documents", headers=bearer_headers(employee))
        fetched = client.get(f"/documents/{document.id}", headers=bearer_headers(employee))
        chunks = client.get(f"/documents/{document.id}/chunks", headers=bearer_headers(employee))
    finally:
        _clear_overrides()

    assert listed.status_code == 200
    assert listed.json() == []
    assert fetched.status_code == 404
    assert fetched.json() == {"detail": "Document not found"}
    assert chunks.status_code == 404
    assert any(event.action == "document.access_denied" for event in session.audits)


def test_admin_grant_read_then_revoke_hides_document_immediately() -> None:
    session = FakeSession()
    tenant = seed_tenant(session)
    admin = seed_user(session, role=ADMIN, tenant_id=tenant.id, password=TEST_PASSWORD)
    employee = seed_user(session, role=EMPLOYEE, tenant_id=tenant.id, password=TEST_PASSWORD)
    document = _owned_document(session, admin, "shared-policy.txt")
    _override_db(session)
    try:
        denied = client.get(f"/documents/{document.id}", headers=bearer_headers(employee))
        granted = client.put(
            f"/documents/{document.id}/permissions",
            json={"user_id": str(employee.id), "privilege": "read"},
            headers=bearer_headers(admin),
        )
        allowed = client.get(f"/documents/{document.id}", headers=bearer_headers(employee))
        grants = client.get(f"/documents/{document.id}/permissions", headers=bearer_headers(admin))
        revoked = client.delete(
            f"/documents/{document.id}/permissions/{employee.id}",
            headers=bearer_headers(admin),
        )
        hidden = client.get(f"/documents/{document.id}", headers=bearer_headers(employee))
        listed = client.get("/documents", headers=bearer_headers(employee))
    finally:
        _clear_overrides()

    assert denied.status_code == 404
    assert granted.status_code == 200
    assert granted.json()["privilege"] == "read"
    assert granted.json()["user_id"] == str(employee.id)
    assert allowed.status_code == 200
    assert allowed.json()["id"] == str(document.id)
    assert grants.status_code == 200
    assert [item["user_id"] for item in grants.json()] == [str(employee.id)]
    assert revoked.status_code == 204
    assert hidden.status_code == 404
    assert listed.json() == []
    actions = [event.action for event in session.audits]
    assert "permission.grant" in actions
    assert "permission.revoke" in actions
    grant_event = next(event for event in session.audits if event.action == "permission.grant")
    assert grant_event.actor_user_id == admin.id
    assert grant_event.actor_tenant_id == tenant.id
    assert grant_event.resource_id == document.id
    assert grant_event.outcome == "success"


def test_employee_cannot_grant_or_revoke_permissions_on_someone_elses_document() -> None:
    session = FakeSession()
    tenant = seed_tenant(session)
    admin = seed_user(session, role=ADMIN, tenant_id=tenant.id, password=TEST_PASSWORD)
    employee = seed_user(session, role=EMPLOYEE, tenant_id=tenant.id, password=TEST_PASSWORD)
    colleague = seed_user(session, role=EMPLOYEE, tenant_id=tenant.id, password=TEST_PASSWORD)
    document = _owned_document(session, admin)
    _override_db(session)
    try:
        grant = client.put(
            f"/documents/{document.id}/permissions",
            json={"user_id": str(colleague.id), "privilege": "read"},
            headers=bearer_headers(employee),
        )
        listed = client.get(f"/documents/{document.id}/permissions", headers=bearer_headers(employee))
        revoke = client.delete(
            f"/documents/{document.id}/permissions/{colleague.id}",
            headers=bearer_headers(employee),
        )
    finally:
        _clear_overrides()

    assert grant.status_code == 404
    assert listed.status_code == 404
    assert revoke.status_code == 404
    assert session.acls == []


def test_read_only_permission_cannot_modify_or_manage_document() -> None:
    session = FakeSession()
    tenant = seed_tenant(session)
    admin = seed_user(session, role=ADMIN, tenant_id=tenant.id, password=TEST_PASSWORD)
    employee = seed_user(session, role=EMPLOYEE, tenant_id=tenant.id, password=TEST_PASSWORD)
    colleague = seed_user(session, role=EMPLOYEE, tenant_id=tenant.id, password=TEST_PASSWORD)
    document = _owned_document(session, admin, "readonly.txt")
    _override_db(session)
    try:
        client.put(
            f"/documents/{document.id}/permissions",
            json={"user_id": str(employee.id), "privilege": "read"},
            headers=bearer_headers(admin),
        )
        readable = client.get(f"/documents/{document.id}", headers=bearer_headers(employee))
        patched = client.patch(
            f"/documents/{document.id}",
            json={"filename": "hijacked.txt"},
            headers=bearer_headers(employee),
        )
        processed = client.post(
            f"/documents/{document.id}/process",
            headers=bearer_headers(employee),
        )
        grant = client.put(
            f"/documents/{document.id}/permissions",
            json={"user_id": str(colleague.id), "privilege": "read"},
            headers=bearer_headers(employee),
        )
        deleted = client.delete(f"/documents/{document.id}", headers=bearer_headers(employee))
        listed = client.get(f"/documents/{document.id}/permissions", headers=bearer_headers(employee))
    finally:
        _clear_overrides()

    assert readable.status_code == 200
    assert patched.status_code == 404
    assert processed.status_code == 404
    assert grant.status_code == 404
    assert deleted.status_code == 404
    assert listed.status_code == 404
    assert document.filename == "readonly.txt"
    assert any(item.id == document.id for item in session.documents)


def test_write_permission_can_update_but_cannot_manage_acl() -> None:
    session = FakeSession()
    tenant = seed_tenant(session)
    admin = seed_user(session, role=ADMIN, tenant_id=tenant.id, password=TEST_PASSWORD)
    employee = seed_user(session, role=EMPLOYEE, tenant_id=tenant.id, password=TEST_PASSWORD)
    colleague = seed_user(session, role=EMPLOYEE, tenant_id=tenant.id, password=TEST_PASSWORD)
    document = _owned_document(session, admin, "writable.txt")
    _override_db(session)
    try:
        granted = client.put(
            f"/documents/{document.id}/permissions",
            json={"user_id": str(employee.id), "privilege": "write"},
            headers=bearer_headers(admin),
        )
        patched = client.patch(
            f"/documents/{document.id}",
            json={"filename": "updated.txt"},
            headers=bearer_headers(employee),
        )
        grant = client.put(
            f"/documents/{document.id}/permissions",
            json={"user_id": str(colleague.id), "privilege": "read"},
            headers=bearer_headers(employee),
        )
        deleted = client.delete(f"/documents/{document.id}", headers=bearer_headers(employee))
    finally:
        _clear_overrides()

    assert granted.status_code == 200
    assert patched.status_code == 200
    assert patched.json()["filename"] == "updated.txt"
    assert grant.status_code == 404
    assert deleted.status_code == 404


def test_admin_can_upgrade_existing_acl_grant() -> None:
    session = FakeSession()
    tenant = seed_tenant(session)
    admin = seed_user(session, role=ADMIN, tenant_id=tenant.id, password=TEST_PASSWORD)
    employee = seed_user(session, role=EMPLOYEE, tenant_id=tenant.id, password=TEST_PASSWORD)
    document = _owned_document(session, admin)
    _override_db(session)
    try:
        first = client.put(
            f"/documents/{document.id}/permissions",
            json={"user_id": str(employee.id), "privilege": "read"},
            headers=bearer_headers(admin),
        )
        second = client.put(
            f"/documents/{document.id}/permissions",
            json={"user_id": str(employee.id), "privilege": "write"},
            headers=bearer_headers(admin),
        )
        patched = client.patch(
            f"/documents/{document.id}",
            json={"filename": "now-writable.txt"},
            headers=bearer_headers(employee),
        )
    finally:
        _clear_overrides()

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["privilege"] == "write"
    assert patched.status_code == 200
    assert len(session.acls) == 1


def test_quarantined_document_is_hidden_from_normal_endpoints_including_tenant_admin() -> None:
    session = FakeSession()
    tenant = seed_tenant(session)
    admin = seed_user(session, role=ADMIN, tenant_id=tenant.id, password=TEST_PASSWORD)
    employee = seed_user(session, role=EMPLOYEE, tenant_id=tenant.id, password=TEST_PASSWORD)
    document = _sample_document("employee_handbook.pdf")
    document.tenant_id = tenant.id
    document.owner_id = None
    session.add(document)
    session.chunks = [_chunk(document.id, "QUARANTINED_SECRET must not leak.")]
    _override_db(session)
    app.dependency_overrides[get_embedder] = lambda: HashEmbedder()
    app.dependency_overrides[get_llm] = lambda: SequenceLLM(["QUARANTINED_SECRET [S1]."])
    try:
        listed = client.get("/documents", headers=bearer_headers(admin))
        fetched = client.get(f"/documents/{document.id}", headers=bearer_headers(admin))
        grant = client.put(
            f"/documents/{document.id}/permissions",
            json={"user_id": str(employee.id), "privilege": "read"},
            headers=bearer_headers(admin),
        )
        searched = client.post(
            "/search",
            json={"query": "QUARANTINED_SECRET", "top_k": 5},
            headers=bearer_headers(admin),
        )
        rag = client.post(
            "/rag/ask",
            json={"question": "What is quarantined?", "document_id": str(document.id)},
            headers=bearer_headers(admin),
        )
        review = client.get("/admin/documents/quarantine", headers=bearer_headers(admin))
    finally:
        _clear_overrides()

    assert listed.json() == []
    assert fetched.status_code == 404
    assert grant.status_code == 404
    assert searched.status_code == 200
    assert searched.json() == []
    assert rag.status_code == 404
    assert review.status_code == 200
    assert review.json()[0]["id"] == str(document.id)
    assert "storage_path" not in review.json()[0]


def test_cross_tenant_document_access_grants_and_search_are_denied() -> None:
    session = FakeSession()
    tenant_a = seed_tenant(session, name="Alpha")
    tenant_b = seed_tenant(session, name="Beta")
    admin_a = seed_user(session, role=ADMIN, tenant_id=tenant_a.id, password=TEST_PASSWORD)
    admin_b = seed_user(session, role=ADMIN, tenant_id=tenant_b.id, password=TEST_PASSWORD)
    employee_b = seed_user(session, role=EMPLOYEE, tenant_id=tenant_b.id, password=TEST_PASSWORD)
    document = _owned_document(session, admin_a, "alpha-only.txt")
    session.chunks = [_chunk(document.id, "ALPHA_TENANT_SECRET vacation policy.")]
    _override_db(session)
    app.dependency_overrides[get_embedder] = lambda: HashEmbedder()
    try:
        fetched = client.get(f"/documents/{document.id}", headers=bearer_headers(admin_b))
        grant_cross_user = client.put(
            f"/documents/{document.id}/permissions",
            json={"user_id": str(employee_b.id), "privilege": "read"},
            headers=bearer_headers(admin_a),
        )
        grant_from_other_admin = client.put(
            f"/documents/{document.id}/permissions",
            json={"user_id": str(employee_b.id), "privilege": "read"},
            headers=bearer_headers(admin_b),
        )
        searched = client.post(
            "/search",
            json={"query": "ALPHA_TENANT_SECRET", "top_k": 5},
            headers=bearer_headers(employee_b),
        )
    finally:
        _clear_overrides()

    assert fetched.status_code == 404
    assert grant_cross_user.status_code == 404
    assert grant_from_other_admin.status_code == 404
    assert searched.status_code == 200
    assert searched.json() == []
    assert session.acls == []


def test_search_and_rag_exclude_unauthorized_and_quarantined_content() -> None:
    session = FakeSession()
    tenant = seed_tenant(session)
    admin = seed_user(session, role=ADMIN, tenant_id=tenant.id, password=TEST_PASSWORD)
    employee = seed_user(session, role=EMPLOYEE, tenant_id=tenant.id, password=TEST_PASSWORD)
    owned = _owned_document(session, employee, "mine.txt")
    hidden = _owned_document(session, admin, "admin-secret.txt")
    quarantined = _sample_document("legacy.pdf")
    quarantined.tenant_id = tenant.id
    quarantined.owner_id = None
    session.add(quarantined)
    session.chunks = [
        _chunk(owned.id, "Employees receive 15 vacation days each year."),
        _chunk(hidden.id, "ADMIN_SECRET_TOKEN must never appear."),
        _chunk(quarantined.id, "QUARANTINE_TOKEN must never appear."),
    ]
    _override_db(session)
    app.dependency_overrides[get_embedder] = lambda: HashEmbedder()
    app.dependency_overrides[get_llm] = lambda: SequenceLLM(
        ["Employees receive 15 vacation days [S1]."]
    )
    try:
        searched = client.post(
            "/search",
            json={"query": "vacation days", "top_k": 10},
            headers=bearer_headers(employee),
        )
        rag = client.post(
            "/rag/ask",
            json={"question": "How many vacation days do employees receive?"},
            headers=bearer_headers(employee),
        )
    finally:
        _clear_overrides()

    assert searched.status_code == 200
    assert [item["document_id"] for item in searched.json()] == [str(owned.id)]
    assert all("ADMIN_SECRET_TOKEN" not in item["content"] for item in searched.json())
    assert all("QUARANTINE_TOKEN" not in item["content"] for item in searched.json())
    assert rag.status_code == 200
    body = rag.json()
    assert "ADMIN_SECRET_TOKEN" not in body["answer"]
    assert "QUARANTINE_TOKEN" not in body["answer"]
    assert all(item["document_id"] == str(owned.id) for item in body["sources"])


def test_tenant_admin_cannot_manage_other_tenant_and_audit_stays_local() -> None:
    session = FakeSession()
    tenant_a = seed_tenant(session, name="Alpha")
    tenant_b = seed_tenant(session, name="Beta")
    admin_a = seed_user(session, role=ADMIN, tenant_id=tenant_a.id, password=TEST_PASSWORD)
    admin_b = seed_user(session, role=ADMIN, tenant_id=tenant_b.id, password=TEST_PASSWORD)
    owner = seed_user(session, role=EMPLOYEE, tenant_id=tenant_a.id, password=TEST_PASSWORD)
    colleague = seed_user(session, role=EMPLOYEE, tenant_id=tenant_a.id, password=TEST_PASSWORD)
    document = _owned_document(session, owner, "alpha.txt")
    _override_db(session)
    try:
        local = client.get(f"/documents/{document.id}", headers=bearer_headers(admin_a))
        foreign = client.get(f"/documents/{document.id}", headers=bearer_headers(admin_b))
        grant = client.put(
            f"/documents/{document.id}/permissions",
            json={"user_id": str(colleague.id), "privilege": "read"},
            headers=bearer_headers(admin_a),
        )
    finally:
        _clear_overrides()

    assert local.status_code == 200
    assert foreign.status_code == 404
    assert grant.status_code == 200
    grant_audits = [event for event in session.audits if event.action == "permission.grant"]
    assert len(grant_audits) == 1
    assert grant_audits[0].actor_tenant_id == tenant_a.id
    assert grant_audits[0].actor_user_id == admin_a.id
    denied = [event for event in session.audits if event.action == "document.access_denied"]
    assert denied
    assert denied[-1].actor_tenant_id == tenant_b.id
    assert denied[-1].actor_user_id == admin_b.id


def test_manager_has_no_automatic_document_access() -> None:
    session = FakeSession()
    tenant = seed_tenant(session)
    owner = seed_user(session, role=EMPLOYEE, tenant_id=tenant.id, password=TEST_PASSWORD)
    manager = seed_user(session, role=MANAGER, tenant_id=tenant.id, password=TEST_PASSWORD)
    document = _owned_document(session, owner, "employee-notes.txt")
    _override_db(session)
    try:
        fetched = client.get(f"/documents/{document.id}", headers=bearer_headers(manager))
        grant = client.put(
            f"/documents/{document.id}/permissions",
            json={"user_id": str(manager.id), "privilege": "read"},
            headers=bearer_headers(manager),
        )
    finally:
        _clear_overrides()

    assert fetched.status_code == 404
    assert grant.status_code == 404
