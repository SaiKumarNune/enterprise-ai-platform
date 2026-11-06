from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app
from app.services.embeddings import get_embedder
from app.services.llm import get_llm
from app.services.storage import LocalStorage, get_storage
from tests.auth_helpers import seed_agent_users, seed_user
from tests.fake_embeddings import HashEmbedder
from tests.fake_llm import SequenceLLM
from tests.test_agents import DRAFT, PLAN_DRAFT, _session_with_policy
from tests.test_documents import FakeSession, _clear_overrides, _override_db, _sample_document
from tests.test_search import _chunk

client = TestClient(app)
PASSWORD = "correct-horse-battery"


def _login(email: str, password: str = PASSWORD):
    response = client.post(
        "/app/login",
        data={"email": email, "password": password},
        follow_redirects=False,
    )
    assert response.status_code == 303
    return response


def _override_storage(directory: Path) -> LocalStorage:
    storage = LocalStorage(root=directory, max_bytes=10 * 1024 * 1024)
    app.dependency_overrides[get_storage] = lambda: storage
    return storage


def test_root_redirects_to_ui() -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/app/"


def test_login_page_and_failure_and_success() -> None:
    session = FakeSession()
    seed_user(session, email="ui@example.com", password=PASSWORD)
    _override_db(session)
    try:
        page = client.get("/app/login")
        denied = client.post(
            "/app/login",
            data={"email": "ui@example.com", "password": "wrong-password-9"},
        )
        client.cookies.clear()
        ok = client.post(
            "/app/login",
            data={"email": "ui@example.com", "password": PASSWORD},
            follow_redirects=False,
        )
        dash = client.get("/app/")
        locked = client.get("/app/documents")
    finally:
        client.cookies.clear()
        _clear_overrides()

    assert page.status_code == 200
    assert "Sign in" in page.text
    assert denied.status_code == 401
    assert "Invalid email or password" in denied.text
    assert ok.status_code == 303
    assert dash.status_code == 200
    assert "Workspace" in dash.text
    assert "ui@example.com" in dash.text
    assert locked.status_code == 200
    assert "No documents yet" in locked.text


def test_unauthenticated_ui_redirects_to_login() -> None:
    client.cookies.clear()
    response = client.get("/app/documents", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/app/login"


def test_upload_enqueues_async_job(tmp_path: Path) -> None:
    session = FakeSession()
    seed_user(session, email="docs@example.com", password=PASSWORD)
    _override_storage(tmp_path)
    _override_db(session)
    try:
        _login("docs@example.com")
        empty = client.get("/app/documents")
        uploaded = client.post(
            "/app/documents/upload",
            files={"file": ("notes.txt", b"Vacation policy text.\n", "text/plain")},
        )
    finally:
        client.cookies.clear()
        _clear_overrides()

    assert empty.status_code == 200
    assert "No documents yet" in empty.text
    assert uploaded.status_code == 200
    assert "notes.txt" in uploaded.text
    assert "queued" in uploaded.text
    assert session.jobs
    assert session.jobs[0].status == "queued"


def test_search_empty_and_results() -> None:
    document = _sample_document("kb.txt")
    apple = _chunk(document.id, 0, "apple pie dessert recipe")
    session = FakeSession(documents=[document])
    session.chunks = [apple]
    user = seed_user(session, email="search@example.com", password=PASSWORD)
    document.owner_id = user.id
    document.tenant_id = user.tenant_id
    _override_db(session)
    try:
        _login("search@example.com")
        blank = client.get("/app/search")
        empty = client.post("/app/search", data={"query": "   ", "retrieval_mode": "lexical"})
        found = client.post("/app/search", data={"query": "apple pie", "retrieval_mode": "lexical"})
    finally:
        client.cookies.clear()
        _clear_overrides()

    assert blank.status_code == 200
    assert "Submit a query" in blank.text
    assert empty.status_code == 200
    assert "Enter a search query" in empty.text
    assert found.status_code == 200
    assert "apple pie dessert recipe" in found.text


def test_rag_ask_renders_answer_and_sources() -> None:
    document = _sample_document("policy.txt")
    chunk = _chunk(document.id, 0, "Full-time employees receive 15 vacation days each year.")
    session = FakeSession(documents=[document])
    session.chunks = [chunk]
    user = seed_user(session, email="rag@example.com", password=PASSWORD)
    document.owner_id = user.id
    document.tenant_id = user.tenant_id
    _override_db(session)
    try:
        _login("rag@example.com")
        response = client.post(
            "/app/ask",
            data={"question": "How many vacation days do employees receive?"},
        )
    finally:
        client.cookies.clear()
        _clear_overrides()

    assert response.status_code == 200
    assert "Answer" in response.text or "abstained" in response.text.lower()
    assert "Sources" in response.text or "No sources" in response.text


def test_agent_employee_cannot_approve_manager_can() -> None:
    session, _document = _session_with_policy()
    emp_headers, mgr_headers, _admin = seed_agent_users(session)
    _ = emp_headers, mgr_headers
    app.dependency_overrides[get_llm] = lambda: SequenceLLM([PLAN_DRAFT, DRAFT])
    app.dependency_overrides[get_embedder] = lambda: HashEmbedder()
    _override_db(session)
    employee = TestClient(app)
    manager = TestClient(app)
    try:
        employee.post(
            "/app/login",
            data={"email": "employee@example.com", "password": PASSWORD},
            follow_redirects=False,
        )
        started = employee.post(
            "/app/agents/run",
            data={
                "request_text": "Review the vacation policy and prepare an approval request for three days off."
            },
        )
        workflow_id = str(session.workflows[-1].id)
        assert "Approve" not in started.text
        forbidden = employee.post(f"/app/agents/{workflow_id}/approve", follow_redirects=False)
        manager.post(
            "/app/login",
            data={"email": "manager@example.com", "password": PASSWORD},
            follow_redirects=False,
        )
        opened = manager.post("/app/agents/lookup", data={"workflow_id": workflow_id})
        approved = manager.post(f"/app/agents/{workflow_id}/approve")
        unknown = manager.post("/app/agents/lookup", data={"workflow_id": str(uuid4())})
    finally:
        _clear_overrides()

    assert started.status_code == 200
    assert "pending_approval" in started.text
    assert forbidden.status_code == 403
    assert "Insufficient permissions" in forbidden.text
    assert opened.status_code == 200
    assert "Approve" in opened.text
    assert approved.status_code == 200
    assert "approved" in approved.text.lower() or "completed" in approved.text
    assert unknown.status_code == 200
    assert "Workflow not found" in unknown.text
