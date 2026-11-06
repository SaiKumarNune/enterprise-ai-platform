from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document
from app.models.processing_job import (
    ACTIVE_JOB_STATUSES,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    ProcessingJob,
)
from app.services.authorization import Principal
from app.services.embeddings import EmbeddingBackend, EmbeddingError
from app.services.extraction import EmptyDocumentError, ExtractionError
from app.services.ingestion import (
    IngestionConflictError,
    LeaseLostError,
    process_stored_document,
)
from app.services.storage import LocalStorage, StoredFileMissingError

RETRYABLE_ERRORS = frozenset({"embedding_failed", "internal"})
SAFE_ERROR_CODES = {
    "embedding_failed": "embedding_failed",
    "empty_document": "empty_document",
    "extraction_failed": "extraction_failed",
    "missing_file": "missing_file",
    "conflict": "conflict",
    "internal": "internal",
}


class JobEnqueueError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def enqueue_processing_job(
    db: Session,
    document: Document,
    principal: Principal,
    *,
    now: datetime | None = None,
) -> ProcessingJob:
    if document.tenant_id != principal.tenant_id:
        raise JobEnqueueError("forbidden", "Document not found")
    if not document.storage_path:
        raise JobEnqueueError("conflict", "Document has no stored file")
    if document.status == "processing":
        existing = active_job_for_document(db, document.id)
        if existing is not None:
            return existing
        raise JobEnqueueError("conflict", "Document is already being processed")
    if document.status not in {"uploaded", "failed", "ready"}:
        raise JobEnqueueError("conflict", "Document cannot be processed in its current state")

    existing = active_job_for_document(db, document.id)
    if existing is not None:
        return existing

    stamp = now or datetime.now(timezone.utc)
    job = ProcessingJob(
        document_id=document.id,
        tenant_id=document.tenant_id,
        requested_by=principal.user_id,
        status=JOB_QUEUED,
        attempt_count=0,
        max_attempts=settings.processing_job_max_attempts,
        available_at=stamp,
        created_at=stamp,
        updated_at=stamp,
    )
    document.status = "processing"
    document.updated_at = stamp
    db.add(job)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        duplicate = active_job_for_document(db, document.id)
        if duplicate is not None:
            return duplicate
        raise JobEnqueueError("conflict", "Document is already being processed") from exc
    db.refresh(job)
    return job


def active_job_for_document(db: Session, document_id: uuid.UUID) -> ProcessingJob | None:
    jobs = _job_rows(db)
    if jobs is not None:
        matches = [
            item
            for item in jobs
            if item.document_id == document_id and item.status in ACTIVE_JOB_STATUSES
        ]
        matches.sort(key=lambda item: item.created_at)
        return matches[0] if matches else None
    statement = (
        select(ProcessingJob)
        .where(
            ProcessingJob.document_id == document_id,
            ProcessingJob.status.in_(ACTIVE_JOB_STATUSES),
        )
        .order_by(ProcessingJob.created_at.asc())
        .limit(1)
    )
    return db.scalars(statement).first()


def get_job(db: Session, job_id: uuid.UUID) -> ProcessingJob | None:
    return db.get(ProcessingJob, job_id)


def lease_heartbeat_interval(lease_seconds: int | None = None) -> float:
    lease = float(lease_seconds if lease_seconds is not None else settings.processing_job_lease_seconds)
    return max(1.0, lease / 3.0)


def claim_next_job(
    db: Session,
    worker_id: str,
    *,
    now: datetime | None = None,
    lease_seconds: int | None = None,
) -> ProcessingJob | None:
    stamp = now or datetime.now(timezone.utc)
    lease = lease_seconds if lease_seconds is not None else settings.processing_job_lease_seconds
    cutoff = stamp - timedelta(seconds=lease)
    jobs = _job_rows(db)
    if jobs is not None:
        if hasattr(db, "_row_lock") and not getattr(db, "_lock_held", False):
            db._row_lock.acquire()
            db._lock_held = True
        candidates = [
            item
            for item in jobs
            if _is_claimable(item, stamp, cutoff)
        ]
        candidates.sort(key=lambda item: item.created_at)
        job = candidates[0] if candidates else None
        if job is None:
            if hasattr(db, "rollback"):
                db.rollback()
            return None
        return _mark_claimed(db, job, worker_id, stamp)

    statement = (
        select(ProcessingJob)
        .where(
            ProcessingJob.available_at <= stamp,
            or_(
                ProcessingJob.status == JOB_QUEUED,
                and_(
                    ProcessingJob.status == JOB_RUNNING,
                    or_(ProcessingJob.locked_at.is_(None), ProcessingJob.locked_at <= cutoff),
                ),
            ),
        )
        .order_by(ProcessingJob.created_at.asc())
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    job = db.scalars(statement).first()
    if job is None:
        db.rollback()
        return None
    return _mark_claimed(db, job, worker_id, stamp)


def _is_claimable(job: ProcessingJob, stamp: datetime, cutoff: datetime) -> bool:
    if job.available_at > stamp:
        return False
    if job.status == JOB_QUEUED:
        return True
    if job.status != JOB_RUNNING:
        return False
    return job.locked_at is None or job.locked_at <= cutoff


def _mark_claimed(db: Session, job: ProcessingJob, worker_id: str, stamp: datetime) -> ProcessingJob:
    job.status = JOB_RUNNING
    job.attempt_count += 1
    job.locked_by = worker_id
    job.locked_at = stamp
    job.started_at = job.started_at or stamp
    job.updated_at = stamp
    job.last_error = None
    db.commit()
    db.refresh(job)
    return job


def renew_lease(
    db: Session,
    job_id: uuid.UUID,
    worker_id: str,
    *,
    now: datetime | None = None,
) -> bool:
    stamp = now or datetime.now(timezone.utc)
    current = _owned_running_job(db, job_id, worker_id)
    if current is None:
        return False
    current.locked_at = stamp
    current.updated_at = stamp
    db.commit()
    return True


def run_claimed_job(
    db: Session,
    job: ProcessingJob,
    storage: LocalStorage,
    embedder: EmbeddingBackend,
    *,
    now: datetime | None = None,
    worker_id: str | None = None,
    session_factory: Callable[[], Session] | None = None,
    heartbeat_interval: float | None = None,
) -> ProcessingJob:
    owner = worker_id or job.locked_by
    if not owner:
        return fail_job(db, job, "internal", worker_id="", stamp=now or datetime.now(timezone.utc), final=True)
    stamp = now or datetime.now(timezone.utc)
    if not renew_lease(db, job.id, owner, now=stamp):
        return get_job(db, job.id) or job

    stop = threading.Event()
    beater: threading.Thread | None = None
    if session_factory is not None:
        interval = heartbeat_interval if heartbeat_interval is not None else lease_heartbeat_interval()
        beater = _start_heartbeat(job.id, owner, stop, session_factory, interval)

    def hold_lease() -> bool:
        return renew_lease(db, job.id, owner)

    try:
        document = db.get(Document, job.document_id)
        if document is None:
            return fail_job(db, job, "internal", worker_id=owner, stamp=stamp, final=True)
        if document.tenant_id != job.tenant_id:
            return fail_job(db, job, "conflict", worker_id=owner, stamp=stamp, final=True)
        try:
            process_stored_document(
                db,
                document,
                storage,
                embedder,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
                assume_claimed=True,
                mark_failed_on_error=False,
                hold_lease=hold_lease,
            )
        except LeaseLostError:
            return get_job(db, job.id) or job
        except StoredFileMissingError:
            return fail_job(db, job, "missing_file", worker_id=owner, stamp=stamp, final=True)
        except EmptyDocumentError:
            return fail_job(db, job, "empty_document", worker_id=owner, stamp=stamp, final=True)
        except ExtractionError:
            return fail_job(db, job, "extraction_failed", worker_id=owner, stamp=stamp, final=True)
        except IngestionConflictError:
            return fail_job(db, job, "conflict", worker_id=owner, stamp=stamp, final=True)
        except EmbeddingError:
            return fail_job(db, job, "embedding_failed", worker_id=owner, stamp=stamp, final=False)
        except Exception:
            return fail_job(db, job, "internal", worker_id=owner, stamp=stamp, final=False)
        return succeed_job(db, job, worker_id=owner, stamp=stamp)
    finally:
        stop.set()
        if beater is not None:
            beater.join(timeout=1)


def succeed_job(
    db: Session,
    job: ProcessingJob,
    *,
    worker_id: str,
    stamp: datetime,
) -> ProcessingJob:
    current = _owned_running_job(db, job.id, worker_id)
    if current is None:
        return get_job(db, job.id) or job
    current.status = JOB_SUCCEEDED
    current.locked_by = None
    current.locked_at = None
    current.finished_at = stamp
    current.updated_at = stamp
    current.last_error = None
    current.error_detail = None
    db.commit()
    db.refresh(current)
    return current


def fail_job(
    db: Session,
    job: ProcessingJob,
    error_code: str,
    *,
    worker_id: str,
    stamp: datetime,
    final: bool,
) -> ProcessingJob:
    current = _owned_running_job(db, job.id, worker_id)
    if current is None:
        return get_job(db, job.id) or job
    code = SAFE_ERROR_CODES.get(error_code, "internal")
    retryable = (not final) and code in RETRYABLE_ERRORS and current.attempt_count < current.max_attempts
    current.last_error = code
    current.error_detail = None
    current.updated_at = stamp
    document = db.get(Document, current.document_id)
    if retryable:
        delay = settings.processing_job_backoff_seconds * current.attempt_count
        current.status = JOB_QUEUED
        current.locked_by = None
        current.locked_at = None
        current.available_at = stamp + timedelta(seconds=delay)
        current.finished_at = None
        db.commit()
        db.refresh(current)
        return current
    current.status = JOB_FAILED
    current.locked_by = None
    current.locked_at = None
    current.finished_at = stamp
    if document is not None:
        document.status = "failed"
        document.updated_at = stamp
    db.commit()
    db.refresh(current)
    return current


def _owned_running_job(db: Session, job_id: uuid.UUID, worker_id: str) -> ProcessingJob | None:
    if not worker_id:
        return None
    current = db.get(ProcessingJob, job_id, with_for_update=True)
    if current is None or current.status != JOB_RUNNING or current.locked_by != worker_id:
        if hasattr(db, "rollback"):
            db.rollback()
        return None
    return current


def _start_heartbeat(
    job_id: uuid.UUID,
    worker_id: str,
    stop: threading.Event,
    session_factory: Callable[[], Session],
    interval: float,
) -> threading.Thread:
    def _loop() -> None:
        while not stop.wait(interval):
            extra = session_factory()
            try:
                if not renew_lease(extra, job_id, worker_id):
                    return
            except Exception:
                return
            finally:
                extra.close()

    thread = threading.Thread(target=_loop, name="processing-lease-heartbeat", daemon=True)
    thread.start()
    return thread


def _job_rows(db: Session) -> list[ProcessingJob] | None:
    if hasattr(db, "jobs"):
        return list(db.jobs)
    return None
