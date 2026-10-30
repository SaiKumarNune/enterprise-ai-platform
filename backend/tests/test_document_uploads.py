import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.database import get_db
from app.main import app
from app.models.document import Document
from app.services.storage import LocalStorage, get_storage
from tests.auth_helpers import auth_headers
from tests.test_documents import FakeSession, _clear_overrides, _override_db, _sample_document

client = TestClient(app)

MINIMAL_PDF = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"


def _override_storage(directory: Path, max_bytes: int = 10 * 1024 * 1024) -> LocalStorage:
    storage = LocalStorage(root=directory, max_bytes=max_bytes)

    def override_get_storage() -> LocalStorage:
        return storage

    app.dependency_overrides[get_storage] = override_get_storage
    return storage


def test_upload_pdf_returns_201(tmp_path: Path) -> None:
    session = FakeSession()
    storage = _override_storage(tmp_path)
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        response = client.post(
            "/documents/upload",
            files={"file": ("handbook.pdf", MINIMAL_PDF, "application/octet-stream")}, headers=headers)
    finally:
        _clear_overrides()

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "handbook.pdf"
    assert body["content_type"] == "application/pdf"
    assert body["status"] == "uploaded"
    assert body["storage_path"].endswith(".pdf")
    assert (storage.root / body["storage_path"]).is_file()
    assert session.commit_called


def test_upload_txt_returns_201(tmp_path: Path) -> None:
    session = FakeSession()
    _override_storage(tmp_path)
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        response = client.post(
            "/documents/upload",
            files={"file": ("notes.txt", b"hello knowledge base\n", "application/octet-stream")}, headers=headers)
    finally:
        _clear_overrides()

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "notes.txt"
    assert body["content_type"] == "text/plain"
    assert body["storage_path"].endswith(".txt")


def test_upload_rejects_unsupported_type(tmp_path: Path) -> None:
    session = FakeSession()
    storage = _override_storage(tmp_path)
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        response = client.post(
            "/documents/upload",
            files={"file": ("image.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 16, "image/png")}, headers=headers)
    finally:
        _clear_overrides()

    assert response.status_code == 415
    assert not any(isinstance(item, Document) for item in session.added)
    assert list(storage.root.glob("*")) == []


def test_upload_rejects_invalid_pdf(tmp_path: Path) -> None:
    session = FakeSession()
    _override_storage(tmp_path)
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        response = client.post(
            "/documents/upload",
            files={"file": ("bad.pdf", b"%PDF-1.4\nthis is not a finished pdf", "application/pdf")}, headers=headers)
    finally:
        _clear_overrides()

    assert response.status_code == 415
    assert not any(isinstance(item, Document) for item in session.added)


def test_upload_rejects_oversized_file(tmp_path: Path) -> None:
    session = FakeSession()
    storage = _override_storage(tmp_path, max_bytes=32)
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        response = client.post(
            "/documents/upload",
            files={"file": ("big.txt", b"a" * 200, "text/plain")}, headers=headers)
    finally:
        _clear_overrides()

    assert response.status_code == 413
    assert not any(isinstance(item, Document) for item in session.added)
    assert list(tmp_path.glob("*")) == []
    assert storage.root == tmp_path


def test_upload_removes_file_when_commit_fails(tmp_path: Path) -> None:
    session = FakeSession(fail_commit=True)
    storage = _override_storage(tmp_path)
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        response = client.post(
            "/documents/upload",
            files={"file": ("handbook.pdf", MINIMAL_PDF, "application/pdf")}, headers=headers)
    finally:
        _clear_overrides()

    assert response.status_code == 500
    assert session.rollback_called
    assert list(storage.root.glob("*")) == []


def test_download_returns_stored_file(tmp_path: Path) -> None:
    stored_name = f"{uuid.uuid4()}.txt"
    (tmp_path / stored_name).write_text("download me", encoding="utf-8")
    document = _sample_document("notes.txt")
    document.content_type = "text/plain"
    document.storage_path = stored_name
    session = FakeSession(documents=[document])
    _override_storage(tmp_path)
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        response = client.get(f"/documents/{document.id}/download", headers=headers)
    finally:
        _clear_overrides()

    assert response.status_code == 200
    assert response.content == b"download me"
    assert response.headers["content-type"].startswith("text/plain")
    assert "attachment" in response.headers["content-disposition"]
    assert "notes.txt" in response.headers["content-disposition"]


def test_download_returns_404_when_file_missing(tmp_path: Path) -> None:
    document = _sample_document()
    document.storage_path = f"{uuid.uuid4()}.pdf"
    session = FakeSession(documents=[document])
    _override_storage(tmp_path)
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        response = client.get(f"/documents/{document.id}/download", headers=headers)
    finally:
        _clear_overrides()

    assert response.status_code == 404
    assert response.json() == {"detail": "Stored file not found"}


def test_download_returns_404_for_metadata_only_document(tmp_path: Path) -> None:
    document = _sample_document()
    session = FakeSession(documents=[document])
    _override_storage(tmp_path)
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        response = client.get(f"/documents/{document.id}/download", headers=headers)
    finally:
        _clear_overrides()

    assert response.status_code == 404
    assert response.json() == {"detail": "Stored file not found"}


def test_delete_removes_stored_file(tmp_path: Path) -> None:
    stored_name = f"{uuid.uuid4()}.txt"
    stored_file = tmp_path / stored_name
    stored_file.write_text("remove me", encoding="utf-8")
    document = Document(
        id=uuid.uuid4(),
        filename="notes.txt",
        content_type="text/plain",
        status="uploaded",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        storage_path=stored_name,
    )
    session = FakeSession(documents=[document])
    _override_storage(tmp_path)
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        response = client.delete(f"/documents/{document.id}", headers=headers)
    finally:
        _clear_overrides()

    assert response.status_code == 204
    assert not stored_file.exists()
    assert session.documents == []


def test_delete_preserves_metadata_only_records(tmp_path: Path) -> None:
    document = _sample_document()
    session = FakeSession(documents=[document])
    _override_storage(tmp_path)
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        response = client.delete(f"/documents/{document.id}", headers=headers)
    finally:
        _clear_overrides()

    assert response.status_code == 204
    assert session.deleted == [document]
