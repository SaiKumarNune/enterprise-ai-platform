import uuid
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.core.database import get_db
from app.main import app
from app.models.document import Document

client = TestClient(app)


class FakeSession:
    """In-memory stand-in so API tests never touch PostgreSQL."""

    def __init__(self, documents: list[Document] | None = None) -> None:
        self.documents = list(documents or [])
        self.added: list[Document] = []
        self.commit_called = False
        self.refreshed: list[Document] = []
        self.last_statement = None

    def add(self, document: Document) -> None:
        now = datetime.now(timezone.utc)
        if document.id is None:
            document.id = uuid.uuid4()
        if document.status is None:
            document.status = "uploaded"
        if document.created_at is None:
            document.created_at = now
        if document.updated_at is None:
            document.updated_at = now
        self.added.append(document)
        self.documents.append(document)

    def commit(self) -> None:
        self.commit_called = True

    def refresh(self, document: Document) -> None:
        self.refreshed.append(document)

    def scalars(self, statement):
        self.last_statement = statement

        class Result:
            def all(inner_self) -> list[Document]:
                return list(self.documents)

        return Result()


def _override_db(session: FakeSession):
    def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db


def _clear_overrides() -> None:
    app.dependency_overrides.clear()


def test_create_document_returns_201() -> None:
    session = FakeSession()
    _override_db(session)
    try:
        response = client.post(
            "/documents",
            json={"filename": "handbook.pdf", "content_type": "application/pdf"},
        )
    finally:
        _clear_overrides()

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "handbook.pdf"
    assert body["content_type"] == "application/pdf"
    assert body["status"] == "uploaded"
    assert body["id"]
    assert body["created_at"]
    assert body["updated_at"]
    assert session.commit_called
    assert len(session.added) == 1
    assert len(session.refreshed) == 1


def test_create_document_rejects_blank_filename() -> None:
    session = FakeSession()
    _override_db(session)
    try:
        response = client.post(
            "/documents",
            json={"filename": "   ", "content_type": "application/pdf"},
        )
    finally:
        _clear_overrides()

    assert response.status_code == 422
    assert session.added == []
    assert not session.commit_called


def test_create_document_rejects_oversized_content_type() -> None:
    session = FakeSession()
    _override_db(session)
    try:
        response = client.post(
            "/documents",
            json={"filename": "handbook.pdf", "content_type": "x" * 101},
        )
    finally:
        _clear_overrides()

    assert response.status_code == 422
    assert session.added == []
    assert not session.commit_called


def test_list_documents_returns_records_in_query_order() -> None:
    newer = Document(
        id=uuid.uuid4(),
        filename="newer.pdf",
        content_type="application/pdf",
        status="uploaded",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    older = Document(
        id=uuid.uuid4(),
        filename="older.pdf",
        content_type="application/pdf",
        status="uploaded",
        created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        updated_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    session = FakeSession(documents=[newer, older])
    _override_db(session)
    try:
        response = client.get("/documents")
    finally:
        _clear_overrides()

    assert response.status_code == 200
    body = response.json()
    assert [item["filename"] for item in body] == ["newer.pdf", "older.pdf"]
    compiled = str(session.last_statement.compile())
    assert "created_at" in compiled
    assert "id" in compiled
    assert compiled.count("DESC") >= 2


def test_list_documents_returns_empty_list() -> None:
    session = FakeSession()
    _override_db(session)
    try:
        response = client.get("/documents")
    finally:
        _clear_overrides()

    assert response.status_code == 200
    assert response.json() == []
