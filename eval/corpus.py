"""Shared synthetic corpus loader for evaluation runners."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.core.security import hash_password
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.tenant import Tenant
from app.models.user import EMPLOYEE, User
from tests.auth_helpers import own_session_documents, seed_user
from tests.test_documents import FakeSession


def load_fake_corpus(dataset: dict, embedder) -> tuple[FakeSession, dict[str, uuid.UUID], dict[str, uuid.UUID], object]:
    document_ids: dict[str, uuid.UUID] = {}
    chunk_ids: dict[str, uuid.UUID] = {}
    documents: list[Document] = []
    chunks: list[DocumentChunk] = []
    now = datetime.now(timezone.utc)
    for doc in dataset["documents"]:
        document_id = uuid.uuid4()
        document_ids[doc["id"]] = document_id
        documents.append(
            Document(
                id=document_id,
                filename=doc["filename"],
                content_type="text/plain",
                status="ready",
                created_at=now,
                updated_at=now,
            )
        )
        for index, chunk in enumerate(doc["chunks"]):
            chunk_uuid = uuid.uuid4()
            chunk_ids[chunk["id"]] = chunk_uuid
            content = chunk["content"]
            chunks.append(
                DocumentChunk(
                    id=chunk_uuid,
                    document_id=document_id,
                    chunk_index=index,
                    content=content,
                    page_number=chunk.get("page_number", 1),
                    created_at=now,
                    embedding=embedder.embed_texts([content])[0],
                )
            )
    session = FakeSession(documents=documents)
    session.chunks = chunks
    owner = seed_user(session)
    own_session_documents(session, owner)
    return session, document_ids, chunk_ids, owner


def load_sql_corpus(session, dataset: dict, embedder, owner) -> tuple[dict[str, uuid.UUID], dict[str, uuid.UUID]]:
    document_ids: dict[str, uuid.UUID] = {}
    chunk_ids: dict[str, uuid.UUID] = {}
    now = datetime.now(timezone.utc)
    for doc in dataset["documents"]:
        document_id = uuid.uuid4()
        document_ids[doc["id"]] = document_id
        session.add(
            Document(
                id=document_id,
                filename=doc["filename"],
                content_type="text/plain",
                status="ready",
                tenant_id=owner.tenant_id,
                owner_id=owner.id,
                created_at=now,
                updated_at=now,
            )
        )
        for index, chunk in enumerate(doc["chunks"]):
            chunk_uuid = uuid.uuid4()
            chunk_ids[chunk["id"]] = chunk_uuid
            session.add(
                DocumentChunk(
                    id=chunk_uuid,
                    document_id=document_id,
                    chunk_index=index,
                    content=chunk["content"],
                    page_number=chunk.get("page_number", 1),
                    created_at=now,
                    embedding=embedder.embed_texts([chunk["content"]])[0],
                )
            )
    session.flush()
    return document_ids, chunk_ids


def seed_sql_owner(session) -> User:
    now = datetime.now(timezone.utc)
    tenant = Tenant(
        id=uuid.uuid4(),
        name="Eval",
        slug=f"eval-{uuid.uuid4().hex[:8]}",
        created_at=now,
        updated_at=now,
    )
    session.add(tenant)
    session.flush()
    owner = User(
        id=uuid.uuid4(),
        email=f"eval-{uuid.uuid4().hex[:8]}@example.test",
        tenant_id=tenant.id,
        password_hash=hash_password("eval-only-password-not-for-prod"),
        role=EMPLOYEE,
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    session.add(owner)
    session.flush()
    return owner
