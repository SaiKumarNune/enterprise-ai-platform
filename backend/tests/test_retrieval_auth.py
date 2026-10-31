"""Authorization regressions for hybrid retrieval (in-memory FakeSession)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.models.chunk import DocumentChunk
from app.services.embeddings import get_embedder
from app.services.llm import get_llm
from tests.auth_helpers import bearer_headers, seed_tenant, seed_user
from tests.fake_embeddings import HashEmbedder
from tests.fake_llm import ScriptedLLM, SequenceLLM
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


def test_employee_manager_admin_hybrid_visibility_and_acl_revoke() -> None:
    session = FakeSession()
    tenant = seed_tenant(session)
    admin = seed_user(session, role="admin", tenant_id=tenant.id)
    manager = seed_user(session, role="manager", tenant_id=tenant.id)
    employee = seed_user(session, role="employee", tenant_id=tenant.id)
    document = _sample_document("admin-policy.txt")
    document.owner_id = admin.id
    document.tenant_id = tenant.id
    session.add(document)
    session.chunks = [_chunk(document.id, "SECRET_ACL_TOKEN hybrid retrieval probe.")]
    _override_db(session)
    app.dependency_overrides[get_embedder] = lambda: HashEmbedder()
    try:
        payload = {"query": "SECRET_ACL_TOKEN", "top_k": 5, "retrieval_mode": "hybrid"}
        employee_before = client.post("/search", json=payload, headers=bearer_headers(employee))
        manager_before = client.post("/search", json=payload, headers=bearer_headers(manager))
        admin_ok = client.post("/search", json=payload, headers=bearer_headers(admin))
        client.put(
            f"/documents/{document.id}/permissions",
            json={"user_id": str(employee.id), "privilege": "read"},
            headers=bearer_headers(admin),
        )
        employee_granted = client.post("/search", json=payload, headers=bearer_headers(employee))
        client.delete(
            f"/documents/{document.id}/permissions/{employee.id}",
            headers=bearer_headers(admin),
        )
        employee_revoked = client.post("/search", json=payload, headers=bearer_headers(employee))
    finally:
        _clear_overrides()

    assert employee_before.json() == []
    assert manager_before.json() == []
    assert [item["document_id"] for item in admin_ok.json()] == [str(document.id)]
    assert [item["document_id"] for item in employee_granted.json()] == [str(document.id)]
    assert employee_revoked.json() == []


def test_cross_tenant_and_quarantine_excluded_from_all_modes() -> None:
    session = FakeSession()
    tenant_a = seed_tenant(session, name="A")
    tenant_b = seed_tenant(session, name="B")
    user_a = seed_user(session, tenant_id=tenant_a.id)
    user_b = seed_user(session, tenant_id=tenant_b.id)
    admin_a = seed_user(session, role="admin", tenant_id=tenant_a.id)
    owned = _sample_document("a.txt")
    owned.owner_id = user_a.id
    owned.tenant_id = tenant_a.id
    quarantined = _sample_document("legacy.txt")
    quarantined.owner_id = None
    quarantined.tenant_id = tenant_a.id
    session.documents = [owned, quarantined]
    session.chunks = [
        _chunk(owned.id, "ALPHAONLYTOKEN vacation."),
        _chunk(quarantined.id, "QUARANTINEFTSXYZ handbook secret."),
    ]
    _override_db(session)
    app.dependency_overrides[get_embedder] = lambda: HashEmbedder()
    app.dependency_overrides[get_llm] = lambda: SequenceLLM([PLAN_INFO, ANSWER])
    try:
        for mode in ("semantic", "lexical", "hybrid"):
            foreign = client.post(
                "/search",
                json={"query": "ALPHAONLYTOKEN", "retrieval_mode": mode},
                headers=bearer_headers(user_b),
            )
            admin_search = client.post(
                "/search",
                json={"query": "QUARANTINEFTSXYZ", "retrieval_mode": mode},
                headers=bearer_headers(admin_a),
            )
            assert foreign.json() == []
            assert all(item["document_id"] != str(quarantined.id) for item in admin_search.json())
            assert all("QUARANTINEFTSXYZ" not in item["content"] for item in admin_search.json())
        rag = client.post(
            "/rag/ask",
            json={"question": "QUARANTINEFTSXYZ", "retrieval_mode": "hybrid"},
            headers=bearer_headers(admin_a),
        )
        agent = client.post(
            "/agents/run",
            json={"request": "Summarize QUARANTINEFTSXYZ", "document_id": str(quarantined.id)},
            headers=bearer_headers(admin_a),
        )
    finally:
        _clear_overrides()

    assert rag.status_code == 200
    assert rag.json()["sources"] == []
    assert "QUARANTINEFTSXYZ" not in rag.json()["answer"]
    assert agent.status_code == 404


def test_unauthorized_lexical_chunk_is_not_returned_before_limit() -> None:
    session = FakeSession()
    tenant = seed_tenant(session)
    owner = seed_user(session, tenant_id=tenant.id)
    other = seed_user(session, tenant_id=tenant.id)
    visible = _sample_document("own.txt")
    hidden = _sample_document("secret.txt")
    visible.owner_id = owner.id
    visible.tenant_id = tenant.id
    hidden.owner_id = other.id
    hidden.tenant_id = tenant.id
    session.documents = [visible, hidden]
    session.chunks = [
        _chunk(hidden.id, "apple pie dessert recipe extra apple apple"),
        _chunk(visible.id, "weak match apple"),
    ]
    _override_db(session)
    try:
        response = client.post(
            "/search",
            json={"query": "apple pie", "top_k": 1, "retrieval_mode": "lexical"},
            headers=bearer_headers(owner),
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200
    assert response.json()[0]["document_id"] == str(visible.id)
    assert "dessert recipe" not in response.json()[0]["content"]


def test_all_modes_respect_grant_revoke_and_hide_diagnostics_for_forbidden_docs() -> None:
    session = FakeSession()
    tenant = seed_tenant(session)
    admin = seed_user(session, role="admin", tenant_id=tenant.id)
    employee = seed_user(session, role="employee", tenant_id=tenant.id)
    manager = seed_user(session, role="manager", tenant_id=tenant.id)
    document = _sample_document("admin-secret.txt")
    document.owner_id = admin.id
    document.tenant_id = tenant.id
    session.add(document)
    session.chunks = [_chunk(document.id, "FORBIDDENDIAGTOKEN confidential policy.")]
    _override_db(session)
    app.dependency_overrides[get_embedder] = lambda: HashEmbedder()
    app.dependency_overrides[get_llm] = lambda: ScriptedLLM(
        "The confidential policy mentions FORBIDDENDIAGTOKEN [S1]."
    )
    try:
        for mode in ("semantic", "lexical", "hybrid"):
            payload = {
                "query": "FORBIDDENDIAGTOKEN",
                "top_k": 5,
                "retrieval_mode": mode,
                "include_diagnostics": True,
            }
            denied = client.post("/search", json=payload, headers=bearer_headers(employee))
            manager_denied = client.post("/search", json=payload, headers=bearer_headers(manager))
            rag_denied = client.post(
                "/rag/ask",
                json={"question": "FORBIDDENDIAGTOKEN", "top_k": 5, "retrieval_mode": mode},
                headers=bearer_headers(employee),
            )
            assert denied.status_code == 200
            assert denied.json() == []
            assert manager_denied.json() == []
            assert rag_denied.status_code == 200
            assert rag_denied.json()["sources"] == []
            assert "FORBIDDENDIAGTOKEN" not in rag_denied.json()["answer"]
            assert str(document.id) not in denied.text

            granted = client.put(
                f"/documents/{document.id}/permissions",
                json={"user_id": str(employee.id), "privilege": "read"},
                headers=bearer_headers(admin),
            )
            assert granted.status_code == 200
            allowed = client.post("/search", json=payload, headers=bearer_headers(employee))
            rag_allowed = client.post(
                "/rag/ask",
                json={"question": "FORBIDDENDIAGTOKEN", "top_k": 5, "retrieval_mode": mode},
                headers=bearer_headers(employee),
            )
            assert [item["document_id"] for item in allowed.json()] == [str(document.id)]
            assert allowed.json()[0]["retrieval_mode"] == mode
            assert rag_allowed.status_code == 200
            assert all(item["document_id"] == str(document.id) for item in rag_allowed.json()["sources"])

            revoked = client.delete(
                f"/documents/{document.id}/permissions/{employee.id}",
                headers=bearer_headers(admin),
            )
            assert revoked.status_code == 204
            hidden = client.post("/search", json=payload, headers=bearer_headers(employee))
            rag_hidden = client.post(
                "/rag/ask",
                json={"question": "FORBIDDENDIAGTOKEN", "retrieval_mode": mode},
                headers=bearer_headers(employee),
            )
            assert hidden.json() == []
            assert rag_hidden.json()["sources"] == []
    finally:
        _clear_overrides()


def test_hybrid_and_semantic_exclude_unauthorized_chunks_before_limit() -> None:
    session = FakeSession()
    tenant = seed_tenant(session)
    owner = seed_user(session, tenant_id=tenant.id)
    other = seed_user(session, tenant_id=tenant.id)
    visible = _sample_document("own.txt")
    hidden = _sample_document("secret.txt")
    visible.owner_id = owner.id
    visible.tenant_id = tenant.id
    hidden.owner_id = other.id
    hidden.tenant_id = tenant.id
    session.documents = [visible, hidden]
    session.chunks = [
        _chunk(hidden.id, "apple pie dessert recipe extra apple apple"),
        _chunk(visible.id, "weak match apple"),
    ]
    _override_db(session)
    try:
        for mode in ("semantic", "hybrid"):
            response = client.post(
                "/search",
                json={"query": "apple pie", "top_k": 1, "retrieval_mode": mode, "include_diagnostics": True},
                headers=bearer_headers(owner),
            )
            assert response.status_code == 200
            assert len(response.json()) == 1
            assert response.json()[0]["document_id"] == str(visible.id)
            assert "dessert recipe" not in response.json()[0]["content"]
            assert str(hidden.id) not in response.text
    finally:
        _clear_overrides()


def test_agent_corpus_search_excludes_quarantined_chunks() -> None:
    session = FakeSession()
    tenant = seed_tenant(session)
    admin = seed_user(session, role="admin", tenant_id=tenant.id)
    owned = _sample_document("visible.txt")
    owned.owner_id = admin.id
    owned.tenant_id = tenant.id
    quarantined = _sample_document("legacy.txt")
    quarantined.owner_id = None
    quarantined.tenant_id = tenant.id
    session.documents = [owned, quarantined]
    session.chunks = [
        _chunk(owned.id, "Employees receive 15 vacation days each year."),
        _chunk(quarantined.id, "QUARANTINEAGENTXYZ must never appear in agents."),
    ]
    _override_db(session)
    app.dependency_overrides[get_embedder] = lambda: HashEmbedder()
    app.dependency_overrides[get_llm] = lambda: SequenceLLM(
        [PLAN_INFO, "Employees receive 15 vacation days each year [S1]."]
    )
    try:
        response = client.post(
            "/agents/run",
            json={"request": "QUARANTINEAGENTXYZ vacation days", "top_k": 5},
            headers=bearer_headers(admin),
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200
    body = response.json()
    assert "QUARANTINEAGENTXYZ" not in (body.get("answer") or "")
    assert all(item["document_id"] != str(quarantined.id) for item in body.get("sources") or [])
