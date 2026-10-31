import uuid
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.models.chunk import DocumentChunk
from app.services.embeddings import get_embedder
from app.services.llm import get_llm
from tests.auth_helpers import bearer_headers, own_session_documents, seed_tenant, seed_user
from tests.fake_embeddings import HashEmbedder
from tests.fake_llm import SequenceLLM
from tests.test_agents import ANSWER, PLAN_INFO
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


def test_unauthenticated_document_search_and_rag_return_401() -> None:
    session = FakeSession()
    _override_db(session)
    try:
        documents = client.get("/documents")
        search = client.post("/search", json={"query": "policy"})
        rag = client.post("/rag/ask", json={"question": "How many days?"})
        upload = client.post("/documents", json={"filename": "a.txt", "content_type": "text/plain"})
    finally:
        _clear_overrides()

    assert documents.status_code == 401
    assert search.status_code == 401
    assert rag.status_code == 401
    assert upload.status_code == 401


def test_same_tenant_owner_can_read_and_cross_tenant_admin_cannot() -> None:
    session = FakeSession()
    tenant_a = seed_tenant(session, name="Alpha")
    tenant_b = seed_tenant(session, name="Beta")
    owner = seed_user(session, role="employee", tenant_id=tenant_a.id)
    outsider_admin = seed_user(session, role="admin", tenant_id=tenant_b.id)
    document = _sample_document("secret.txt")
    session.add(document)
    own_session_documents(session, owner)
    _override_db(session)
    try:
        allowed = client.get(f"/documents/{document.id}", headers=bearer_headers(owner))
        denied = client.get(f"/documents/{document.id}", headers=bearer_headers(outsider_admin))
    finally:
        _clear_overrides()

    assert allowed.status_code == 200
    assert denied.status_code == 404
    assert denied.json() == {"detail": "Document not found"}


def test_document_owner_cannot_be_spoofed_in_create_payload() -> None:
    session = FakeSession()
    user = seed_user(session)
    other = seed_user(session, tenant_id=user.tenant_id)
    _override_db(session)
    try:
        response = client.post(
            "/documents",
            json={
                "filename": "mine.txt",
                "content_type": "text/plain",
                "owner_id": str(other.id),
                "tenant_id": str(uuid.uuid4()),
            },
            headers=bearer_headers(user),
        )
    finally:
        _clear_overrides()

    assert response.status_code == 201
    body = response.json()
    assert body["owner_id"] == str(user.id)
    assert body["tenant_id"] == str(user.tenant_id)


def test_permission_grant_and_revoke() -> None:
    session = FakeSession()
    tenant = seed_tenant(session)
    owner = seed_user(session, role="employee", tenant_id=tenant.id)
    colleague = seed_user(session, role="employee", tenant_id=tenant.id)
    document = _sample_document("shared.txt")
    session.add(document)
    own_session_documents(session, owner)
    _override_db(session)
    try:
        before = client.get(f"/documents/{document.id}", headers=bearer_headers(colleague))
        granted = client.put(
            f"/documents/{document.id}/permissions",
            json={"user_id": str(colleague.id), "privilege": "read"},
            headers=bearer_headers(owner),
        )
        after = client.get(f"/documents/{document.id}", headers=bearer_headers(colleague))
        revoked = client.delete(
            f"/documents/{document.id}/permissions/{colleague.id}",
            headers=bearer_headers(owner),
        )
        gone = client.get(f"/documents/{document.id}", headers=bearer_headers(colleague))
        self_grant = client.put(
            f"/documents/{document.id}/permissions",
            json={"user_id": str(owner.id), "privilege": "manage"},
            headers=bearer_headers(owner),
        )
    finally:
        _clear_overrides()

    assert before.status_code == 404
    assert granted.status_code == 200
    assert granted.json()["privilege"] == "read"
    assert after.status_code == 200
    assert revoked.status_code == 204
    assert gone.status_code == 404
    assert self_grant.status_code == 403


def test_pagination_excludes_unauthorized_documents() -> None:
    session = FakeSession()
    tenant = seed_tenant(session)
    owner = seed_user(session, role="employee", tenant_id=tenant.id)
    other = seed_user(session, role="employee", tenant_id=tenant.id)
    visible = _sample_document("visible.txt")
    hidden = _sample_document("hidden.txt")
    visible.created_at = datetime.now(timezone.utc)
    hidden.created_at = datetime.now(timezone.utc)
    session.documents = [visible, hidden]
    visible.owner_id = owner.id
    visible.tenant_id = tenant.id
    hidden.owner_id = other.id
    hidden.tenant_id = tenant.id
    _override_db(session)
    try:
        response = client.get("/documents", headers=bearer_headers(owner))
    finally:
        _clear_overrides()

    assert response.status_code == 200
    names = [item["filename"] for item in response.json()]
    assert names == ["visible.txt"]


def test_vector_retrieval_excludes_unauthorized_chunks_before_limit() -> None:
    session = FakeSession()
    tenant_a = seed_tenant(session, name="A")
    tenant_b = seed_tenant(session, name="B")
    user = seed_user(session, tenant_id=tenant_a.id)
    own = _sample_document("own.txt")
    secret = _sample_document("secret.txt")
    session.documents = [own, secret]
    own.owner_id = user.id
    own.tenant_id = tenant_a.id
    secret.owner_id = seed_user(session, tenant_id=tenant_b.id).id
    secret.tenant_id = tenant_b.id
    own_chunk = _chunk(own.id, "weak match apple")
    secret_chunk = _chunk(secret.id, "apple pie dessert recipe extra apple apple")
    session.chunks = [secret_chunk, own_chunk]
    _override_db(session)
    try:
        response = client.post(
            "/search",
            json={"query": "apple pie", "top_k": 1},
            headers=bearer_headers(user),
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["document_id"] == str(own.id)
    assert body[0]["content"] == "weak match apple"


def test_rag_and_agents_exclude_unauthorized_material() -> None:
    session = FakeSession()
    tenant_a = seed_tenant(session, name="A")
    tenant_b = seed_tenant(session, name="B")
    user = seed_user(session, tenant_id=tenant_a.id)
    own = _sample_document("own.txt")
    secret = _sample_document("secret.txt")
    session.documents = [own, secret]
    own.owner_id = user.id
    own.tenant_id = tenant_a.id
    secret.owner_id = seed_user(session, tenant_id=tenant_b.id).id
    secret.tenant_id = tenant_b.id
    session.chunks = [
        _chunk(own.id, "Employees receive 15 vacation days each year."),
        _chunk(secret.id, "SECRET_TOKEN_SHOULD_NEVER_APPEAR in answers."),
    ]
    app.dependency_overrides[get_llm] = lambda: SequenceLLM(
        [PLAN_INFO, ANSWER, "SECRET_TOKEN_SHOULD_NEVER_APPEAR [S1]."]
    )
    app.dependency_overrides[get_embedder] = lambda: HashEmbedder()
    _override_db(session)
    try:
        rag = client.post(
            "/rag/ask",
            json={"question": "How many vacation days do full-time employees receive?"},
            headers=bearer_headers(user),
        )
        secret_filter = client.post(
            "/rag/ask",
            json={"question": "leak", "document_id": str(secret.id)},
            headers=bearer_headers(user),
        )
        agent = client.post(
            "/agents/run",
            json={"request": "How many vacation days do full-time employees receive?"},
            headers=bearer_headers(user),
        )
    finally:
        _clear_overrides()

    assert rag.status_code == 200
    assert "SECRET_TOKEN_SHOULD_NEVER_APPEAR" not in rag.json()["answer"]
    assert all(item["document_id"] == str(own.id) for item in rag.json()["sources"])
    assert secret_filter.status_code == 404
    assert agent.status_code == 200
    assert "SECRET_TOKEN_SHOULD_NEVER_APPEAR" not in (agent.json().get("answer") or "")


def test_cross_tenant_admin_cannot_approve_workflow() -> None:
    session = FakeSession()
    tenant_a = seed_tenant(session, name="A")
    tenant_b = seed_tenant(session, name="B")
    employee = seed_user(session, role="employee", tenant_id=tenant_a.id)
    foreign_admin = seed_user(session, role="admin", tenant_id=tenant_b.id)
    local_manager = seed_user(session, role="manager", tenant_id=tenant_a.id)
    document = _sample_document("policy.txt")
    session.add(document)
    own_session_documents(session, employee)
    session.chunks = [_chunk(document.id, "Employees receive 15 vacation days.")]
    app.dependency_overrides[get_llm] = lambda: SequenceLLM(
        [
            '{"intent":"draft_action","tool":"prepare_approval_request","action_type":"leave_request"}',
            '{"action_type":"leave_request","days":3,"source_ids":["S1"]}',
        ]
    )
    app.dependency_overrides[get_embedder] = lambda: HashEmbedder()
    _override_db(session)
    try:
        started = client.post(
            "/agents/run",
            json={"request": "Prepare an approval request for three days off."},
            headers=bearer_headers(employee),
        )
        workflow_id = started.json()["workflow_id"]
        denied = client.post(
            f"/agents/{workflow_id}/approve",
            json={},
            headers=bearer_headers(foreign_admin),
        )
        allowed = client.post(
            f"/agents/{workflow_id}/approve",
            json={},
            headers=bearer_headers(local_manager),
        )
    finally:
        _clear_overrides()

    assert started.status_code == 200
    assert denied.status_code == 404
    assert allowed.status_code in {200, 409}


def test_disabled_user_cannot_read_own_document() -> None:
    session = FakeSession()
    user = seed_user(session, is_active=False)
    document = _sample_document("notes.txt")
    session.add(document)
    own_session_documents(session, user)
    _override_db(session)
    try:
        response = client.get(f"/documents/{document.id}", headers=bearer_headers(user))
    finally:
        _clear_overrides()

    assert response.status_code == 401


def test_register_ignores_client_supplied_tenant() -> None:
    from app.core.config import settings
    from app.services.legacy_backfill import GREENFIELD_TENANT_ID

    session = FakeSession()
    _override_db(session)
    spoofed = str(uuid.uuid4())
    try:
        response = client.post(
            "/auth/register",
            json={
                "email": "tenant-spoof@example.com",
                "password": "correct-horse-battery",
                "tenant_id": spoofed,
            },
        )
    finally:
        _clear_overrides()

    assert response.status_code == 201
    expected = settings.default_tenant_id.strip() or str(GREENFIELD_TENANT_ID)
    assert response.json()["tenant_id"] == expected
    assert response.json()["tenant_id"] != spoofed
