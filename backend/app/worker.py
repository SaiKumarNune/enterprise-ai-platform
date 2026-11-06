"""PostgreSQL-backed document processing worker.

Run separately from the API process:

    PYTHONPATH=backend python -m app.worker

Use --once to claim and run a single job (tests and local debugging).
This is not Celery and does not use Redis.
"""

from __future__ import annotations

import argparse
import logging
import time
import uuid

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.logging import configure_logging, log_event
from app.services.embeddings import get_embedder
from app.services.processing_jobs import claim_next_job, run_claimed_job
from app.services.storage import LocalStorage

logger = logging.getLogger("app.worker")


def run_once(worker_id: str | None = None) -> bool:
    worker_id = worker_id or uuid.uuid4().hex[:16]
    db = SessionLocal()
    try:
        job = claim_next_job(db, worker_id)
        if job is None:
            return False
        storage = LocalStorage(root=settings.upload_dir, max_bytes=settings.max_upload_bytes)
        embedder = get_embedder()
        finished = run_claimed_job(
            db,
            job,
            storage,
            embedder,
            worker_id=worker_id,
            session_factory=SessionLocal,
        )
        log_event(
            logger,
            "processing_job.finished",
            job_status=finished.status,
            attempt_count=finished.attempt_count,
        )
        return True
    finally:
        db.close()


def run_forever() -> None:
    worker_id = uuid.uuid4().hex[:16]
    log_event(logger, "processing_worker.start")
    poll = max(0.2, settings.processing_job_poll_seconds)
    while True:
        try:
            claimed = run_once(worker_id)
        except Exception:
            log_event(logger, "processing_worker.error", level=logging.ERROR)
            claimed = False
        if not claimed:
            time.sleep(poll)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Document processing worker")
    parser.add_argument("--once", action="store_true", help="Claim and run at most one job, then exit")
    args = parser.parse_args()
    if args.once:
        run_once()
        return
    run_forever()


if __name__ == "__main__":
    main()
