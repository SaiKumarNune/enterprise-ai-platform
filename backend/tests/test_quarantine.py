import uuid
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.models.chunk import DocumentChunk
from app.services.embeddings import get_embedder
from app.services.llm import get_llm
from tests.auth_helpers import bearer_headers, seed_tenant, seed_user
from tests.fake_embeddings import HashEmbedder
from tests.fake_llm import SequenceLLM
from tests.test_documents import FakeSession, _clear_overrides, _override_db, _sample_document

client = TestClient(app)


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


def test_quarantined_document_is_hidden_from_ordinary_apis() -> None:
    session = FakeSession()
    tenant = seed_tenant(session)
    admin = seed_user(session, role="admin", tenant_id=tenant.id)
    employee = seed_user(session, role="employee", tenant_id=tenant.id)
    document = _sample_document("employee_handbook.pdf")
    document.tenant_id = tenant.id
    document.owner_id = None
    session.add(document)
    session.chunks = [_chunk(document.id, "SECRET handbook text that must not leak.")]
    _override_db(session)
    app.dependency_overrides[get_embedder] = lambda: HashEmbedder()
    app.dependency_overrides[get_llm] = lambda: SequenceLLM(["SECRET handbook text [S1]."])
    try:
        listed = client.get("/documents", headers=bearer_headers(admin))
        fetched = client.get(f"/documents/{document.id}", headers=bearer_headers(admin))
        downloaded = client.get(f"/documents/{document.id}/download", headers=bearer_headers(employee))
        searched = client.post(
            "/search",
            json={"query": "handbook", "top_k": 5},
            headers=bearer_headers(admin),
        )
        rag = client.post(
            "/rag/ask",
            json={"question": "What does the handbook say?"},
            headers=bearer_headers(admin),
        )
        agent = client.post(
            "/agents/run",
            json={"request": "Summarize the handbook", "document_id": str(document.id)},
            headers=bearer_headers(admin),
        )
    finally:
        _clear_overrides()

    assert listed.status_code == 200
    assert listed.json() == []
    assert fetched.status_code == 404
    assert downloaded.status_code == 404
    assert searched.status_code == 200
    assert searched.json() == []
    assert rag.status_code == 200
    assert rag.json()["sources"] == []
    assert "SECRET handbook" not in rag.json()["answer"]
    assert agent.status_code == 404


def test_tenant_admin_can_review_metadata_then_assign_owner() -> None:
    session = FakeSession()
    tenant = seed_tenant(session)
    admin = seed_user(session, role="admin", tenant_id=tenant.id)
    employee = seed_user(session, role="employee", tenant_id=tenant.id)
    document = _sample_document("employee_handbook.pdf")
    document.tenant_id = tenant.id
    document.owner_id = None
    document.storage_path = "must-not-appear.bin"
    session.add(document)
    _override_db(session)
    try:
        review = client.get("/admin/documents/quarantine", headers=bearer_headers(admin))
        assigned = client.post(
            f"/admin/documents/{document.id}/assign-owner",
            json={"user_id": str(employee.id)},
            headers=bearer_headers(admin),
        )
        after = client.get(f"/documents/{document.id}", headers=bearer_headers(employee))
    finally:
        _clear_overrides()

    assert review.status_code == 200
    assert len(review.json()) == 1
    item = review.json()[0]
    assert item["filename"] == "employee_handbook.pdf"
    assert "storage_path" not in item
    assert "content" not in item
    assert assigned.status_code == 200
    assert assigned.json()["filename"] == "employee_handbook.pdf"
    assert "storage_path" not in assigned.json()
    assert after.status_code == 200
    assert after.json()["owner_id"] == str(employee.id)
    assert session.audits
    assert session.audits[-1].action == "document.ownership.assign"


def test_cross_tenant_admin_cannot_review_or_assign_quarantine() -> None:
    session = FakeSession()
    tenant_a = seed_tenant(session, name="A")
    tenant_b = seed_tenant(session, name="B")
    admin_a = seed_user(session, role="admin", tenant_id=tenant_a.id)
    admin_b = seed_user(session, role="admin", tenant_id=tenant_b.id)
    owner_b = seed_user(session, role="employee", tenant_id=tenant_b.id)
    document = _sample_document("employee_handbook.pdf")
    document.tenant_id = tenant_a.id
    document.owner_id = None
    session.add(document)
    _override_db(session)
    try:
        listed = client.get("/admin/documents/quarantine", headers=bearer_headers(admin_b))
        assigned = client.post(
            f"/admin/documents/{document.id}/assign-owner",
            json={"user_id": str(owner_b.id)},
            headers=bearer_headers(admin_b),
        )
        local = client.get("/admin/documents/quarantine", headers=bearer_headers(admin_a))
    finally:
        _clear_overrides()

    assert listed.status_code == 200
    assert listed.json() == []
    assert assigned.status_code == 404
    assert local.status_code == 200
    assert local.json()[0]["id"] == str(document.id)


def test_employee_cannot_use_quarantine_admin_api() -> None:
    session = FakeSession()
    tenant = seed_tenant(session)
    employee = seed_user(session, role="employee", tenant_id=tenant.id)
    document = _sample_document("employee_handbook.pdf")
    document.tenant_id = tenant.id
    document.owner_id = None
    session.add(document)
    _override_db(session)
    try:
        listed = client.get("/admin/documents/quarantine", headers=bearer_headers(employee))
        assigned = client.post(
            f"/admin/documents/{document.id}/assign-owner",
            json={"user_id": str(employee.id)},
            headers=bearer_headers(employee),
        )
    finally:
        _clear_overrides()

    assert listed.status_code == 403
    assert assigned.status_code == 403


def test_new_uploads_still_require_authenticated_owner() -> None:
    session = FakeSession()
    user = seed_user(session)
    _override_db(session)
    try:
        unauthenticated = client.post(
            "/documents",
            json={"filename": "new.txt", "content_type": "text/plain"},
        )
        created = client.post(
            "/documents",
            json={"filename": "new.txt", "content_type": "text/plain"},
            headers=bearer_headers(user),
        )
    finally:
        _clear_overrides()

    assert unauthenticated.status_code == 401
    assert created.status_code == 201
    assert created.json()["owner_id"] == str(user.id)
    assert created.json()["tenant_id"] == str(user.tenant_id)
