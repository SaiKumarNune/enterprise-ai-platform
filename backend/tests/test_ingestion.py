import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.models.chunk import DocumentChunk
from app.services.embeddings import get_embedder
from app.services.storage import LocalStorage, get_storage
from tests.fake_embeddings import FailingEmbedder, HashEmbedder
from tests.pdf_fixtures import pdf_with_pages
from tests.test_documents import (
    FakeSession,
    _clear_overrides,
    _override_db,
    _sample_document,
)

client = TestClient(app)


def _override_storage(directory: Path) -> LocalStorage:
    storage = LocalStorage(root=directory, max_bytes=10 * 1024 * 1024)

    def override_get_storage() -> LocalStorage:
        return storage

    app.dependency_overrides[get_storage] = override_get_storage
    return storage


def test_process_txt_document_creates_chunks(tmp_path: Path) -> None:
    stored_name = f"{uuid.uuid4()}.txt"
    (tmp_path / stored_name).write_text("alpha " * 400, encoding="utf-8")
    document = _sample_document("notes.txt")
    document.content_type = "text/plain"
    document.storage_path = stored_name
    session = FakeSession(documents=[document])
    _override_storage(tmp_path)
    _override_db(session)
    try:
        response = client.post(f"/documents/{document.id}/process")
        chunks = client.get(f"/documents/{document.id}/chunks")
    finally:
        _clear_overrides()

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    body = chunks.json()
    assert chunks.status_code == 200
    assert len(body) >= 1
    assert body[0]["chunk_index"] == 0
    assert body[0]["content"]
    assert session.chunks
    assert session.chunks[0].embedding is not None
    assert len(session.chunks[0].embedding) == 384


def test_process_pdf_document_sets_page_numbers(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "app.api.documents.settings",
        replace(settings, chunk_size=24, chunk_overlap=0),
    )
    stored_name = f"{uuid.uuid4()}.pdf"
    (tmp_path / stored_name).write_bytes(
        pdf_with_pages("Hello World Page One", "Hello World Page Two")
    )
    document = _sample_document("handbook.pdf")
    document.storage_path = stored_name
    session = FakeSession(documents=[document])
    _override_storage(tmp_path)
    _override_db(session)
    try:
        response = client.post(f"/documents/{document.id}/process")
        chunks = client.get(f"/documents/{document.id}/chunks")
    finally:
        _clear_overrides()

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    pages = {item["page_number"] for item in chunks.json()}
    assert 1 in pages
    assert 2 in pages


def test_reprocess_replaces_existing_chunks(tmp_path: Path) -> None:
    stored_name = f"{uuid.uuid4()}.txt"
    (tmp_path / stored_name).write_text("replacement text", encoding="utf-8")
    document = _sample_document("notes.txt")
    document.content_type = "text/plain"
    document.storage_path = stored_name
    document.status = "ready"
    old = DocumentChunk(
        id=uuid.uuid4(),
        document_id=document.id,
        chunk_index=0,
        content="old chunk",
        page_number=None,
    )
    session = FakeSession(documents=[document])
    session.chunks = [old]
    _override_storage(tmp_path)
    _override_db(session)
    try:
        response = client.post(f"/documents/{document.id}/process")
        chunks = client.get(f"/documents/{document.id}/chunks")
    finally:
        _clear_overrides()

    assert response.status_code == 200
    body = chunks.json()
    assert len(body) == 1
    assert body[0]["content"] == "replacement text"
    assert all(item["content"] != "old chunk" for item in body)


def test_process_rejects_when_already_processing(tmp_path: Path) -> None:
    document = _sample_document()
    document.status = "processing"
    document.storage_path = f"{uuid.uuid4()}.txt"
    session = FakeSession(documents=[document])
    _override_storage(tmp_path)
    _override_db(session)
    try:
        response = client.post(f"/documents/{document.id}/process")
    finally:
        _clear_overrides()

    assert response.status_code == 409


def test_process_rejects_metadata_only_document(tmp_path: Path) -> None:
    document = _sample_document()
    session = FakeSession(documents=[document])
    _override_storage(tmp_path)
    _override_db(session)
    try:
        response = client.post(f"/documents/{document.id}/process")
    finally:
        _clear_overrides()

    assert response.status_code == 409
    assert "path" not in response.text.lower()


def test_process_returns_404_when_stored_file_missing(tmp_path: Path) -> None:
    document = _sample_document()
    document.storage_path = f"{uuid.uuid4()}.txt"
    session = FakeSession(documents=[document])
    _override_storage(tmp_path)
    _override_db(session)
    try:
        response = client.post(f"/documents/{document.id}/process")
    finally:
        _clear_overrides()

    assert response.status_code == 404
    assert response.json() == {"detail": "Stored file not found"}
    assert document.status == "uploaded"


def test_process_keeps_old_chunks_when_embedding_fails(tmp_path: Path) -> None:
    stored_name = f"{uuid.uuid4()}.txt"
    (tmp_path / stored_name).write_text("new content to embed", encoding="utf-8")
    document = _sample_document("notes.txt")
    document.content_type = "text/plain"
    document.storage_path = stored_name
    old = DocumentChunk(
        id=uuid.uuid4(),
        document_id=document.id,
        chunk_index=0,
        content="keep me",
        page_number=None,
        created_at=datetime.now(timezone.utc),
        embedding=HashEmbedder().embed_texts(["keep me"])[0],
    )
    session = FakeSession(documents=[document])
    session.chunks = [old]
    _override_storage(tmp_path)
    _override_db(session)
    app.dependency_overrides[get_embedder] = lambda: FailingEmbedder()
    try:
        response = client.post(f"/documents/{document.id}/process")
    finally:
        _clear_overrides()

    assert response.status_code == 500
    assert document.status == "failed"
    assert session.chunks == [old]


def test_process_empty_file_marks_failed(tmp_path: Path) -> None:
    stored_name = f"{uuid.uuid4()}.txt"
    (tmp_path / stored_name).write_text("   \n", encoding="utf-8")
    document = _sample_document("empty.txt")
    document.content_type = "text/plain"
    document.storage_path = stored_name
    old = DocumentChunk(
        id=uuid.uuid4(),
        document_id=document.id,
        chunk_index=0,
        content="keep me",
        page_number=None,
    )
    session = FakeSession(documents=[document])
    session.chunks = [old]
    _override_storage(tmp_path)
    _override_db(session)
    try:
        response = client.post(f"/documents/{document.id}/process")
    finally:
        _clear_overrides()

    assert response.status_code == 422
    assert document.status == "failed"
    assert session.chunks == [old]


def test_list_chunks_paginates_in_index_order(tmp_path: Path) -> None:
    document = _sample_document()
    now = datetime.now(timezone.utc)
    chunks = [
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=document.id,
            chunk_index=index,
            content=f"chunk-{index}",
            page_number=None,
            created_at=now,
        )
        for index in range(3)
    ]
    session = FakeSession(documents=[document])
    session.chunks = chunks
    _override_storage(tmp_path)
    _override_db(session)
    try:
        response = client.get(
            f"/documents/{document.id}/chunks",
            params={"skip": 1, "limit": 1},
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200
    body = response.json()
    assert [item["content"] for item in body] == ["chunk-1"]


def test_process_and_chunks_return_404_and_422_for_bad_ids(tmp_path: Path) -> None:
    session = FakeSession()
    _override_storage(tmp_path)
    _override_db(session)
    try:
        missing = client.post(f"/documents/{uuid.uuid4()}/process")
        invalid = client.get("/documents/not-a-uuid/chunks")
    finally:
        _clear_overrides()

    assert missing.status_code == 404
    assert invalid.status_code == 422


def test_patch_cannot_change_processing_status() -> None:
    document = _sample_document()
    session = FakeSession(documents=[document])
    _override_db(session)
    try:
        response = client.patch(
            f"/documents/{document.id}",
            json={"status": "ready"},
        )
    finally:
        _clear_overrides()

    assert response.status_code == 422
    assert document.status == "uploaded"
