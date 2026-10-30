import uuid
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.models.chunk import DocumentChunk
from app.services.embeddings import get_embedder
from tests.fake_embeddings import FailingEmbedder, HashEmbedder
from tests.auth_helpers import auth_headers
from tests.test_documents import FakeSession, _clear_overrides, _override_db, _sample_document

client = TestClient(app)


def _chunk(document_id, index: int, content: str) -> DocumentChunk:
    embedder = HashEmbedder()
    return DocumentChunk(
        id=uuid.uuid4(),
        document_id=document_id,
        chunk_index=index,
        content=content,
        page_number=1,
        created_at=datetime.now(timezone.utc),
        embedding=embedder.embed_texts([content])[0],
    )


def test_search_ranks_semantic_matches_first() -> None:
    document = _sample_document("kb.txt")
    apple = _chunk(document.id, 0, "apple pie dessert recipe")
    engine = _chunk(document.id, 1, "car engine maintenance manual")
    session = FakeSession(documents=[document])
    session.chunks = [engine, apple]
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        response = client.post("/search", json={"query": "apple pie", "top_k": 2}, headers=headers)
    finally:
        _clear_overrides()

    assert response.status_code == 200
    body = response.json()
    assert [item["content"] for item in body] == [
        "apple pie dessert recipe",
        "car engine maintenance manual",
    ]
    assert body[0]["similarity"] >= body[1]["similarity"]
    assert body[0]["chunk_id"] == str(apple.id)
    assert body[0]["document_id"] == str(document.id)
    assert body[0]["chunk_index"] == 0
    assert body[0]["page_number"] == 1


def test_search_filters_by_document_id() -> None:
    first = _sample_document("one.txt")
    second = _sample_document("two.txt")
    keep = _chunk(first.id, 0, "apple orchard notes")
    other = _chunk(second.id, 0, "apple tart notes")
    session = FakeSession(documents=[first, second])
    session.chunks = [keep, other]
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        response = client.post(
            "/search",
            json={"query": "apple", "top_k": 5, "document_id": str(first.id)}, headers=headers)
    finally:
        _clear_overrides()

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["document_id"] == str(first.id)


def test_search_excludes_chunks_without_embeddings() -> None:
    document = _sample_document("notes.txt")
    with_vector = _chunk(document.id, 0, "searchable text")
    bare = DocumentChunk(
        id=uuid.uuid4(),
        document_id=document.id,
        chunk_index=1,
        content="no embedding yet",
        page_number=None,
        created_at=datetime.now(timezone.utc),
        embedding=None,
    )
    session = FakeSession(documents=[document])
    session.chunks = [bare, with_vector]
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        response = client.post("/search", json={"query": "searchable", "top_k": 5}, headers=headers)
    finally:
        _clear_overrides()

    assert response.status_code == 200
    assert [item["content"] for item in response.json()] == ["searchable text"]


def test_search_returns_empty_list_when_no_matches() -> None:
    session = FakeSession()
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        response = client.post("/search", json={"query": "anything", "top_k": 3}, headers=headers)
    finally:
        _clear_overrides()

    assert response.status_code == 200
    assert response.json() == []


def test_search_validates_query_and_top_k() -> None:
    session = FakeSession()
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        blank = client.post("/search", json={"query": "   ", "top_k": 3}, headers=headers)
        too_large = client.post("/search", json={"query": "hello", "top_k": 51}, headers=headers)
        too_small = client.post("/search", json={"query": "hello", "top_k": 0}, headers=headers)
    finally:
        _clear_overrides()

    assert blank.status_code == 422
    assert too_large.status_code == 422
    assert too_small.status_code == 422


def test_search_returns_404_for_unknown_document_filter() -> None:
    session = FakeSession()
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        response = client.post(
            "/search",
            json={"query": "hello", "document_id": str(uuid.uuid4())},
            headers=headers,
        )
    finally:
        _clear_overrides()

    assert response.status_code == 404


def test_search_handles_embedding_failure() -> None:
    session = FakeSession()
    _override_db(session)
    _user, headers = auth_headers(session)
    app.dependency_overrides[get_embedder] = lambda: FailingEmbedder()
    try:
        response = client.post("/search", json={"query": "hello"}, headers=headers)
    finally:
        _clear_overrides()

    assert response.status_code == 500
    assert "path" not in response.text.lower()
