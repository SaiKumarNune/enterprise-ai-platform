import uuid
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.core.database import get_db
from app.main import app
from app.models.chunk import DocumentChunk
from app.models.document import Document

client = TestClient(app)


def _clause_int(clause) -> int | None:
    if clause is None:
        return None
    value = getattr(clause, "value", None)
    if value is None:
        return None
    return int(value)


class FakeSession:
    """In-memory stand-in so API tests never touch PostgreSQL."""

    def __init__(
        self,
        documents: list[Document] | None = None,
        fail_commit: bool = False,
    ) -> None:
        self.documents = list(documents or [])
        self.chunks: list[DocumentChunk] = []
        self.added: list[object] = []
        self.deleted: list[object] = []
        self.commit_called = False
        self.rollback_called = False
        self.fail_commit = fail_commit
        self.refreshed: list[Document] = []
        self.last_statement = None

    def add(self, obj: object) -> None:
        now = datetime.now(timezone.utc)
        if isinstance(obj, DocumentChunk):
            if obj.id is None:
                obj.id = uuid.uuid4()
            if obj.created_at is None:
                obj.created_at = now
            self.added.append(obj)
            self.chunks.append(obj)
            return
        document = obj
        assert isinstance(document, Document)
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
        if self.fail_commit:
            raise RuntimeError("commit failed")
        self.commit_called = True

    def rollback(self) -> None:
        self.rollback_called = True

    def refresh(self, obj: object) -> None:
        self.refreshed.append(obj)

    def flush(self) -> None:
        return None

    def get(self, model, ident):
        rows = self.chunks if model is DocumentChunk else self.documents
        for item in rows:
            if item.id == ident:
                return item
        return None

    def delete(self, obj: object) -> None:
        self.deleted.append(obj)
        if isinstance(obj, DocumentChunk):
            self.chunks = [item for item in self.chunks if item.id != obj.id]
            return
        document = obj
        assert isinstance(document, Document)
        self.documents = [item for item in self.documents if item.id != document.id]
        self.chunks = [item for item in self.chunks if item.document_id != document.id]

    def execute(self, statement):
        compiled = statement.compile()
        sql = str(compiled).lower()
        params = compiled.params
        if sql.strip().startswith("delete"):
            document_id = next(
                (value for value in params.values() if isinstance(value, uuid.UUID)),
                None,
            )
            if document_id is None:
                self.chunks = []
            else:
                self.chunks = [
                    item for item in self.chunks if item.document_id != document_id
                ]
        return statement

    def scalars(self, statement):
        self.last_statement = statement
        entity = statement.column_descriptions[0]["entity"]
        offset = _clause_int(getattr(statement, "_offset_clause", None)) or 0
        limit = _clause_int(getattr(statement, "_limit_clause", None))
        if entity is DocumentChunk:
            rows = sorted(self.chunks, key=lambda item: item.chunk_index)
            params = statement.compile().params
            document_id = next(
                (value for value in params.values() if isinstance(value, uuid.UUID)),
                None,
            )
            if document_id is not None:
                rows = [item for item in rows if item.document_id == document_id]
        else:
            rows = list(self.documents)
        selected = rows[offset:] if limit is None else rows[offset : offset + limit]

        class Result:
            def all(inner_self) -> list:
                return selected

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


def _sample_document(filename: str = "handbook.pdf") -> Document:
    now = datetime.now(timezone.utc)
    return Document(
        id=uuid.uuid4(),
        filename=filename,
        content_type="application/pdf",
        status="uploaded",
        created_at=now,
        updated_at=now,
    )


def test_get_document_returns_200() -> None:
    document = _sample_document()
    session = FakeSession(documents=[document])
    _override_db(session)
    try:
        response = client.get(f"/documents/{document.id}")
    finally:
        _clear_overrides()

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(document.id)
    assert body["filename"] == "handbook.pdf"
    assert body["content_type"] == "application/pdf"
    assert body["status"] == "uploaded"


def test_get_document_returns_404_when_missing() -> None:
    session = FakeSession()
    _override_db(session)
    try:
        response = client.get(f"/documents/{uuid.uuid4()}")
    finally:
        _clear_overrides()

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found"}


def test_get_document_returns_422_for_invalid_uuid() -> None:
    session = FakeSession()
    _override_db(session)
    try:
        response = client.get("/documents/not-a-uuid")
    finally:
        _clear_overrides()

    assert response.status_code == 422


def test_delete_document_returns_204() -> None:
    document = _sample_document()
    session = FakeSession(documents=[document])
    _override_db(session)
    try:
        response = client.delete(f"/documents/{document.id}")
    finally:
        _clear_overrides()

    assert response.status_code == 204
    assert response.content == b""
    assert session.commit_called
    assert session.deleted == [document]
    assert session.documents == []


def test_delete_document_returns_404_when_missing() -> None:
    session = FakeSession()
    _override_db(session)
    try:
        response = client.delete(f"/documents/{uuid.uuid4()}")
    finally:
        _clear_overrides()

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found"}
    assert not session.commit_called
    assert session.deleted == []


def test_delete_document_returns_422_for_invalid_uuid() -> None:
    session = FakeSession()
    _override_db(session)
    try:
        response = client.delete("/documents/not-a-uuid")
    finally:
        _clear_overrides()

    assert response.status_code == 422
    assert not session.commit_called


def test_list_documents_paginates_with_skip_and_limit() -> None:
    first = _sample_document("first.pdf")
    second = _sample_document("second.pdf")
    third = _sample_document("third.pdf")
    session = FakeSession(documents=[first, second, third])
    _override_db(session)
    try:
        response = client.get("/documents", params={"skip": 1, "limit": 1})
    finally:
        _clear_overrides()

    assert response.status_code == 200
    body = response.json()
    assert [item["filename"] for item in body] == ["second.pdf"]
    compiled = str(session.last_statement.compile())
    assert "LIMIT" in compiled
    assert "OFFSET" in compiled


def test_list_documents_rejects_invalid_pagination() -> None:
    session = FakeSession()
    _override_db(session)
    try:
        too_large = client.get("/documents", params={"limit": 101})
        negative = client.get("/documents", params={"skip": -1})
    finally:
        _clear_overrides()

    assert too_large.status_code == 422
    assert negative.status_code == 422


def test_patch_document_updates_metadata() -> None:
    document = _sample_document()
    session = FakeSession(documents=[document])
    _override_db(session)
    try:
        response = client.patch(
            f"/documents/{document.id}",
            json={"filename": "renamed.txt"},
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "renamed.txt"
    assert body["status"] == "uploaded"
    assert body["content_type"] == "application/pdf"
    assert session.commit_called


def test_patch_document_returns_404_when_missing() -> None:
    session = FakeSession()
    _override_db(session)
    try:
        response = client.patch(
            f"/documents/{uuid.uuid4()}",
            json={"filename": "renamed.txt"},
        )
    finally:
        _clear_overrides()

    assert response.status_code == 404


def test_patch_document_returns_422_for_invalid_input() -> None:
    document = _sample_document()
    session = FakeSession(documents=[document])
    _override_db(session)
    try:
        invalid_uuid = client.patch("/documents/not-a-uuid", json={"filename": "x"})
        blank = client.patch(f"/documents/{document.id}", json={"filename": "  "})
        bad_status = client.patch(
            f"/documents/{document.id}",
            json={"status": "not-a-status"},
        )
    finally:
        _clear_overrides()

    assert invalid_uuid.status_code == 422
    assert blank.status_code == 422
    assert bad_status.status_code == 422
    assert not session.commit_called


def test_list_documents_returns_empty_list() -> None:
    session = FakeSession()
    _override_db(session)
    try:
        response = client.get("/documents")
    finally:
        _clear_overrides()

    assert response.status_code == 200
    assert response.json() == []
