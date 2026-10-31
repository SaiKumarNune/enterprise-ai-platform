"""In-memory hybrid retrieval tests. These do not use PostgreSQL FTS or pgvector."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.models.chunk import DocumentChunk
from app.services.authorization import Principal
from app.services.embeddings import get_embedder
from pathlib import Path

from app.services.retrieval import LEXICAL, lexical_tokens, retrieve_chunks
from tests.auth_helpers import auth_headers
from tests.fake_embeddings import FailingEmbedder, HashEmbedder
from tests.test_documents import FakeSession, _clear_overrides, _override_db, _sample_document

client = TestClient(app)


def _chunk(document_id, index: int, content: str, *, embed: bool = True) -> DocumentChunk:
    return DocumentChunk(
        id=uuid.uuid4(),
        document_id=document_id,
        chunk_index=index,
        content=content,
        page_number=1,
        created_at=datetime.now(timezone.utc),
        embedding=HashEmbedder().embed_texts([content])[0] if embed else None,
    )


def test_sql_builders_filter_authorization_before_limit() -> None:
    text = (Path(__file__).resolve().parents[1] / "app/services/retrieval.py").read_text(encoding="utf-8")
    semantic_sql = text.split("distance = DocumentChunk.embedding.cosine_distance", 1)[1].split(
        "def _lexical_rows", 1
    )[0]
    lexical_sql = text.split("tsquery = func.plainto_tsquery", 1)[1].split("def _memory_lexical_score", 1)[0]
    for block, label in ((semantic_sql, "semantic"), (lexical_sql, "lexical")):
        assert "authorized_document_clause" in block, label
        assert ".limit(" in block, label
        assert block.index("authorized_document_clause") < block.index(".limit("), label
        assert block.index(".where(") < block.index(".order_by(") < block.index(".limit("), label


def test_lexical_tokens_drop_stopwords_and_punctuation() -> None:
    assert lexical_tokens("The vacation-policy, POL-4419!") == ["vacation-policy", "pol-4419"]


def test_search_lexical_ranks_exact_terms_ahead_of_unrelated_text() -> None:
    document = _sample_document("kb.txt")
    match = _chunk(document.id, 0, "Employees must cite form POL-4419 before travel.")
    other = _chunk(document.id, 1, "Cafeteria hours and parking permits.")
    session = FakeSession(documents=[document])
    session.chunks = [other, match]
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        response = client.post(
            "/search",
            json={"query": "POL-4419", "top_k": 5, "retrieval_mode": "lexical"},
            headers=headers,
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200
    assert [item["content"] for item in response.json()] == [match.content]


def test_search_semantic_mode_excludes_chunks_without_embeddings() -> None:
    document = _sample_document("notes.txt")
    with_vector = _chunk(document.id, 0, "searchable text")
    bare = _chunk(document.id, 1, "searchable text without a vector", embed=False)
    session = FakeSession(documents=[document])
    session.chunks = [bare, with_vector]
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        response = client.post(
            "/search",
            json={"query": "searchable", "top_k": 5, "retrieval_mode": "semantic"},
            headers=headers,
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200
    assert [item["content"] for item in response.json()] == ["searchable text"]


def test_hybrid_includes_lexical_only_chunk_and_deduplicates() -> None:
    document = _sample_document("hybrid.txt")
    lexical_only = _chunk(document.id, 0, "UniqueIdentifier ZX-9001 appears only here.", embed=False)
    semantic = _chunk(document.id, 1, "Dessert recipes and orchard notes.")
    session = FakeSession(documents=[document])
    session.chunks = [lexical_only, semantic]
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        response = client.post(
            "/search",
            json={
                "query": "ZX-9001",
                "top_k": 5,
                "retrieval_mode": "hybrid",
                "include_diagnostics": True,
            },
            headers=headers,
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200
    body = response.json()
    ids = [item["chunk_id"] for item in body]
    assert str(lexical_only.id) in ids
    assert len(ids) == len(set(ids))
    assert body[0]["retrieval_mode"] == "hybrid"
    assert body[0]["similarity"] > 0


def test_hybrid_respects_top_k_and_empty_results() -> None:
    document = _sample_document("kb.txt")
    session = FakeSession(documents=[document])
    session.chunks = [
        _chunk(document.id, 0, "alpha token one"),
        _chunk(document.id, 1, "beta token two"),
        _chunk(document.id, 2, "gamma token three"),
    ]
    _override_db(session)
    user, headers = auth_headers(session)
    try:
        limited = client.post(
            "/search",
            json={"query": "token", "top_k": 2, "retrieval_mode": "hybrid"},
            headers=headers,
        )
        empty = client.post(
            "/search",
            json={"query": "the and of", "top_k": 5, "retrieval_mode": "lexical"},
            headers=headers,
        )
        missing = retrieve_chunks(
            session,
            query="no-such-term-xyz",
            principal=Principal.from_user(user),
            top_k=5,
            mode=LEXICAL,
        )
    finally:
        _clear_overrides()

    assert limited.status_code == 200
    assert len(limited.json()) == 2
    assert empty.status_code == 200
    assert empty.json() == []
    assert missing == []


def test_search_rejects_unknown_retrieval_mode() -> None:
    session = FakeSession()
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        response = client.post(
            "/search",
            json={"query": "hello", "retrieval_mode": "bm25"},
            headers=headers,
        )
    finally:
        _clear_overrides()

    assert response.status_code == 422


def test_diagnostics_omitted_by_default_and_never_list_other_documents() -> None:
    document = _sample_document("own.txt")
    session = FakeSession(documents=[document])
    session.chunks = [_chunk(document.id, 0, "vacation policy")]
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        response = client.post(
            "/search",
            json={"query": "vacation", "top_k": 1, "retrieval_mode": "lexical"},
            headers=headers,
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200
    assert "retrieval_mode" not in response.json()[0] or response.json()[0]["retrieval_mode"] is None


def test_hybrid_embedding_failure_degrades_to_lexical() -> None:
    document = _sample_document("kb.txt")
    chunk = _chunk(document.id, 0, "travel form POL-4419")
    session = FakeSession(documents=[document])
    session.chunks = [chunk]
    _override_db(session)
    _user, headers = auth_headers(session)
    app.dependency_overrides[get_embedder] = lambda: FailingEmbedder()
    try:
        hybrid = client.post(
            "/search",
            json={"query": "POL-4419", "retrieval_mode": "hybrid"},
            headers=headers,
        )
        semantic = client.post(
            "/search",
            json={"query": "POL-4419", "retrieval_mode": "semantic"},
            headers=headers,
        )
    finally:
        _clear_overrides()

    assert hybrid.status_code == 200
    assert hybrid.json()[0]["chunk_id"] == str(chunk.id)
    assert semantic.status_code == 500
