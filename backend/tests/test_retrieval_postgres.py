"""Isolated PostgreSQL tests for pgvector + full-text retrieval.

Requires TEST_DATABASE_URL to a dedicated *test* database. Never uses enterprise_ai.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.security import create_access_token, hash_password, normalize_email
from app.models.acl import DocumentACL
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.tenant import Tenant
from app.models.user import ADMIN, EMPLOYEE, User
from app.services.authorization import Principal
from app.services.retrieval import retrieve_chunks
from tests.fake_embeddings import HashEmbedder
from tests.postgres_isolation import SKIP_ISOLATED_POSTGRES, postgres_client as postgres_client

pytestmark = SKIP_ISOLATED_POSTGRES


def _headers(user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user.id)}"}


def _seed_user(session: Session, tenant_id, *, role: str, email: str) -> User:
    now = datetime.now(timezone.utc)
    user = User(
        id=uuid4(),
        email=normalize_email(email),
        tenant_id=tenant_id,
        password_hash=hash_password("pytest-only-acl-password"),
        role=role,
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def _seed_doc_with_chunk(
    session: Session,
    *,
    tenant_id,
    owner_id,
    filename: str,
    content: str,
) -> tuple[Document, DocumentChunk]:
    now = datetime.now(timezone.utc)
    document = Document(
        id=uuid4(),
        filename=filename,
        content_type="text/plain",
        status="ready",
        tenant_id=tenant_id,
        owner_id=owner_id,
        created_at=now,
        updated_at=now,
    )
    session.add(document)
    session.flush()
    chunk = DocumentChunk(
        id=uuid4(),
        document_id=document.id,
        chunk_index=0,
        content=content,
        page_number=1,
        created_at=now,
        embedding=HashEmbedder().embed_texts([content])[0],
    )
    session.add(chunk)
    session.commit()
    session.refresh(document)
    session.refresh(chunk)
    return document, chunk


def test_postgres_lexical_and_vector_filter_before_limit(
    postgres_client: tuple,
) -> None:
    client, factory = postgres_client
    session = factory()
    try:
        now = datetime.now(timezone.utc)
        tenant = Tenant(id=uuid4(), name="IR", slug=f"ir-{uuid4().hex[:8]}", created_at=now, updated_at=now)
        other_tenant = Tenant(id=uuid4(), name="Other", slug=f"ot-{uuid4().hex[:8]}", created_at=now, updated_at=now)
        session.add_all([tenant, other_tenant])
        session.commit()
        owner = _seed_user(session, tenant.id, role=EMPLOYEE, email="owner@example.test")
        outsider = _seed_user(session, other_tenant.id, role=ADMIN, email="outsider@example.test")
        colleague = _seed_user(session, tenant.id, role=EMPLOYEE, email="peer@example.test")
        visible, visible_chunk = _seed_doc_with_chunk(
            session,
            tenant_id=tenant.id,
            owner_id=owner.id,
            filename="own.txt",
            content="Employees receive 15 vacation days each year.",
        )
        _secret, secret_chunk = _seed_doc_with_chunk(
            session,
            tenant_id=tenant.id,
            owner_id=colleague.id,
            filename="secret.txt",
            content="vacation vacation vacation days days days SECRET_FTS_TOKEN",
        )
        _foreign, _foreign_chunk = _seed_doc_with_chunk(
            session,
            tenant_id=other_tenant.id,
            owner_id=outsider.id,
            filename="foreign.txt",
            content="vacation days SECRET_FTS_TOKEN",
        )
        principal = Principal.from_user(owner)
        lexical = retrieve_chunks(
            session,
            query="vacation days",
            principal=principal,
            top_k=1,
            mode="lexical",
        )
        semantic = retrieve_chunks(
            session,
            query="vacation days",
            principal=principal,
            top_k=1,
            mode="semantic",
            query_embedding=HashEmbedder().embed_texts(["vacation days"])[0],
        )
        vector_exists = session.scalar(select(DocumentChunk.search_vector).where(DocumentChunk.id == visible_chunk.id))
        owner_id = owner.id
        secret_id = secret_chunk.id
        visible_id = visible.id
        owner_headers = _headers(owner)
        colleague_headers = _headers(colleague)
        document_id = visible.id
        colleague_id = colleague.id
    finally:
        session.close()

    assert vector_exists is not None
    assert lexical[0].chunk.id == visible_chunk.id
    assert semantic[0].chunk.id == visible_chunk.id
    assert secret_id not in {lexical[0].chunk.id, semantic[0].chunk.id}

    searched = client.post(
        "/search",
        json={"query": "SECRET_FTS_TOKEN", "top_k": 5, "retrieval_mode": "lexical"},
        headers=owner_headers,
    )
    assert searched.status_code == 200
    assert searched.json() == []

    granted = client.put(
        f"/documents/{document_id}/permissions",
        json={"user_id": str(colleague_id), "privilege": "read"},
        headers=owner_headers,
    )
    assert granted.status_code == 200
    after_grant = client.post(
        "/search",
        json={"query": "15 vacation days", "retrieval_mode": "hybrid"},
        headers=colleague_headers,
    )
    assert [item["document_id"] for item in after_grant.json()] == [str(visible_id)]
    client.delete(f"/documents/{document_id}/permissions/{colleague_id}", headers=owner_headers)
    after_revoke = client.post(
        "/search",
        json={"query": "15 vacation days", "retrieval_mode": "hybrid"},
        headers=colleague_headers,
    )
    assert after_revoke.json() == []
    assert owner_id


def test_postgres_quarantine_excluded_from_fts(
    postgres_client: tuple,
) -> None:
    client, factory = postgres_client
    session = factory()
    try:
        now = datetime.now(timezone.utc)
        tenant = Tenant(id=uuid4(), name="Q", slug=f"q-{uuid4().hex[:8]}", created_at=now, updated_at=now)
        session.add(tenant)
        session.commit()
        admin = _seed_user(session, tenant.id, role=ADMIN, email="admin@example.test")
        document = Document(
            id=uuid4(),
            filename="legacy.txt",
            content_type="text/plain",
            status="ready",
            tenant_id=tenant.id,
            owner_id=None,
            created_at=now,
            updated_at=now,
        )
        session.add(document)
        session.flush()
        session.add(
            DocumentChunk(
                id=uuid4(),
                document_id=document.id,
                chunk_index=0,
                content="QUARANTINE_FTS_TOKEN must never appear.",
                page_number=1,
                created_at=now,
                embedding=HashEmbedder().embed_texts(["QUARANTINE_FTS_TOKEN must never appear."])[0],
            )
        )
        session.commit()
        headers = _headers(admin)
    finally:
        session.close()

    for mode in ("semantic", "lexical", "hybrid"):
        response = client.post(
            "/search",
            json={"query": "QUARANTINE_FTS_TOKEN", "retrieval_mode": mode},
            headers=headers,
        )
        assert response.status_code == 200
        assert response.json() == []
    _ = DocumentACL
    verify = factory()
    try:
        verify.execute(text("SELECT to_tsvector('english', 'vacation days')"))
    finally:
        verify.close()
