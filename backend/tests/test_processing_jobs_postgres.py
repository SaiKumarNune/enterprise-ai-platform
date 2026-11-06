"""Isolated PostgreSQL tests for job claiming with FOR UPDATE SKIP LOCKED.

Requires TEST_DATABASE_URL pointing at a dedicated *_test database.
Never uses development PostgreSQL enterprise_ai.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.security import hash_password, normalize_email
from app.models.document import Document
from app.models.processing_job import JOB_QUEUED, ProcessingJob
from app.models.tenant import Tenant
from app.models.user import EMPLOYEE, User
from app.services.processing_jobs import claim_next_job
from tests.postgres_isolation import SKIP_ISOLATED_POSTGRES, postgres_client as postgres_client

pytestmark = SKIP_ISOLATED_POSTGRES


def _seed_queued_job(session: Session) -> ProcessingJob:
    now = datetime.now(timezone.utc)
    tenant = Tenant(
        id=uuid4(),
        name="Jobs",
        slug=f"jobs-{uuid4().hex[:8]}",
        created_at=now,
        updated_at=now,
    )
    session.add(tenant)
    session.flush()
    user = User(
        id=uuid4(),
        email=normalize_email(f"jobs-{uuid4().hex[:8]}@example.test"),
        tenant_id=tenant.id,
        password_hash=hash_password("pytest-only-job-password"),
        role=EMPLOYEE,
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    session.add(user)
    session.flush()
    document = Document(
        id=uuid4(),
        filename="notes.txt",
        content_type="text/plain",
        status="processing",
        tenant_id=tenant.id,
        owner_id=user.id,
        storage_path=f"{uuid4()}.txt",
        created_at=now,
        updated_at=now,
    )
    session.add(document)
    session.flush()
    job = ProcessingJob(
        id=uuid4(),
        document_id=document.id,
        tenant_id=tenant.id,
        requested_by=user.id,
        status=JOB_QUEUED,
        attempt_count=0,
        max_attempts=3,
        available_at=now,
        created_at=now,
        updated_at=now,
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def test_postgres_for_update_skip_locked_skips_held_row(
    postgres_client: tuple[object, sessionmaker[Session]],
) -> None:
    _client, factory = postgres_client
    setup = factory()
    try:
        job = _seed_queued_job(setup)
        job_id = job.id
    finally:
        setup.close()

    holder = factory()
    waiter = factory()
    try:
        held = holder.scalars(
            select(ProcessingJob)
            .where(ProcessingJob.id == job_id)
            .with_for_update(skip_locked=True)
        ).first()
        skipped = waiter.scalars(
            select(ProcessingJob)
            .where(ProcessingJob.id == job_id)
            .with_for_update(skip_locked=True)
        ).first()
        assert held is not None
        assert held.id == job_id
        assert skipped is None
    finally:
        holder.rollback()
        waiter.rollback()
        holder.close()
        waiter.close()


def test_postgres_concurrent_claim_next_job_is_exclusive(
    postgres_client: tuple[object, sessionmaker[Session]],
) -> None:
    _client, factory = postgres_client
    setup = factory()
    try:
        _seed_queued_job(setup)
    finally:
        setup.close()

    barrier = threading.Barrier(2)
    owners: list[str | None] = []
    lock = threading.Lock()

    def _worker(name: str) -> None:
        db = factory()
        try:
            barrier.wait(timeout=5)
            claimed = claim_next_job(db, name)
            with lock:
                owners.append(None if claimed is None else claimed.locked_by)
        finally:
            db.close()

    threads = [
        threading.Thread(target=_worker, args=("worker-a",)),
        threading.Thread(target=_worker, args=("worker-b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert owners.count(None) == 1
    assert len([name for name in owners if name is not None]) == 1
