import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.models.document import Document
from app.services.embeddings import EmbeddingError
from app.services.processing_jobs import (
    claim_next_job,
    fail_job,
    renew_lease,
    run_claimed_job,
    succeed_job,
)
from app.services.storage import LocalStorage, get_storage
from tests.auth_helpers import auth_headers, bearer_headers, seed_user
from tests.fake_embeddings import HashEmbedder
from tests.test_documents import FakeSession, _clear_overrides, _override_db, _sample_document

client = TestClient(app)


class _FailingEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("embedding backend unavailable")


def _override_storage(directory: Path) -> LocalStorage:
    storage = LocalStorage(root=directory, max_bytes=10 * 1024 * 1024)

    def override_get_storage() -> LocalStorage:
        return storage

    app.dependency_overrides[get_storage] = override_get_storage
    return storage


def _ready_document(tmp_path: Path, text: str = "Employees receive 15 vacation days.") -> tuple[Document, FakeSession]:
    stored_name = f"{uuid.uuid4()}.txt"
    (tmp_path / stored_name).write_text(text, encoding="utf-8")
    document = _sample_document("notes.txt")
    document.content_type = "text/plain"
    document.storage_path = stored_name
    session = FakeSession(documents=[document])
    return document, session


def test_process_without_wait_query_stays_synchronous(tmp_path: Path) -> None:
    document, session = _ready_document(tmp_path)
    _override_storage(tmp_path)
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        response = client.post(f"/documents/{document.id}/process", headers=headers)
    finally:
        _clear_overrides()

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["id"] == str(document.id)
    assert session.jobs == []


def test_async_process_enqueues_job_and_worker_completes(tmp_path: Path) -> None:
    document, session = _ready_document(tmp_path)
    storage = _override_storage(tmp_path)
    _override_db(session)
    user, headers = auth_headers(session)
    try:
        enqueued = client.post(
            f"/documents/{document.id}/process",
            params={"wait": "false"},
            headers=headers,
        )
        job_id = enqueued.json()["id"]
        claimed = claim_next_job(session, "worker-a")
        finished = run_claimed_job(session, claimed, storage, HashEmbedder())
        status = client.get(
            f"/documents/{document.id}/processing-jobs/{job_id}",
            headers=headers,
        )
    finally:
        _clear_overrides()

    assert enqueued.status_code == 202
    assert enqueued.json()["status"] == "queued"
    assert document.status == "ready"
    assert finished.status == "succeeded"
    assert status.status_code == 200
    assert status.json()["status"] == "succeeded"
    assert session.chunks
    assert user.id is not None


def test_duplicate_enqueue_returns_same_active_job(tmp_path: Path) -> None:
    document, session = _ready_document(tmp_path)
    _override_storage(tmp_path)
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        first = client.post(
            f"/documents/{document.id}/process",
            params={"wait": "false"},
            headers=headers,
        )
        second = client.post(
            f"/documents/{document.id}/process",
            params={"wait": "false"},
            headers=headers,
        )
        sync = client.post(f"/documents/{document.id}/process", headers=headers)
    finally:
        _clear_overrides()

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["id"] == second.json()["id"]
    assert sync.status_code == 409


def test_job_status_is_tenant_scoped(tmp_path: Path) -> None:
    document, session = _ready_document(tmp_path)
    _override_storage(tmp_path)
    _override_db(session)
    owner, owner_headers = auth_headers(session)
    outsider = seed_user(session, email="other@example.com")
    try:
        enqueued = client.post(
            f"/documents/{document.id}/process",
            params={"wait": "false"},
            headers=owner_headers,
        )
        job_id = enqueued.json()["id"]
        denied = client.get(
            f"/documents/{document.id}/processing-jobs/{job_id}",
            headers=bearer_headers(outsider),
        )
        mismatch = client.get(
            f"/documents/{uuid.uuid4()}/processing-jobs/{job_id}",
            headers=owner_headers,
        )
    finally:
        _clear_overrides()

    assert enqueued.status_code == 202
    assert denied.status_code == 404
    assert mismatch.status_code == 404
    assert owner.tenant_id != outsider.tenant_id


def test_stale_running_job_is_reclaimed(tmp_path: Path) -> None:
    document, session = _ready_document(tmp_path)
    _override_storage(tmp_path)
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        client.post(
            f"/documents/{document.id}/process",
            params={"wait": "false"},
            headers=headers,
        )
        first = claim_next_job(session, "crashed-worker")
        first.locked_at = datetime.now(timezone.utc) - timedelta(hours=1)
        session.commit()
        recovered = claim_next_job(session, "healthy-worker", lease_seconds=30)
    finally:
        _clear_overrides()

    assert first.id == recovered.id
    assert recovered.locked_by == "healthy-worker"
    assert recovered.attempt_count == 2
    assert recovered.status == "running"


def test_retryable_embedding_failure_then_success(tmp_path: Path) -> None:
    document, session = _ready_document(tmp_path)
    storage = _override_storage(tmp_path)
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        client.post(
            f"/documents/{document.id}/process",
            params={"wait": "false"},
            headers=headers,
        )
        first = claim_next_job(session, "w1")
        retried = run_claimed_job(session, first, storage, _FailingEmbedder())
        assert retried.status == "queued"
        assert retried.last_error == "embedding_failed"
        assert document.status == "processing"
        retried.available_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()
        second = claim_next_job(session, "w1")
        finished = run_claimed_job(session, second, storage, HashEmbedder())
    finally:
        _clear_overrides()

    assert finished.status == "succeeded"
    assert document.status == "ready"


def test_max_attempts_marks_document_failed(tmp_path: Path) -> None:
    document, session = _ready_document(tmp_path)
    storage = _override_storage(tmp_path)
    _override_db(session)
    _user, headers = auth_headers(session)

    try:
        client.post(
            f"/documents/{document.id}/process",
            params={"wait": "false"},
            headers=headers,
        )
        session.jobs[0].max_attempts = 1
        claimed = claim_next_job(session, "w1")
        finished = run_claimed_job(session, claimed, storage, _FailingEmbedder())
    finally:
        _clear_overrides()

    assert finished.status == "failed"
    assert finished.last_error == "embedding_failed"
    assert document.status == "failed"


def test_claim_is_exclusive(tmp_path: Path) -> None:
    document, session = _ready_document(tmp_path)
    _override_storage(tmp_path)
    _override_db(session)
    _user, headers = auth_headers(session)
    client.post(
        f"/documents/{document.id}/process",
        params={"wait": "false"},
        headers=headers,
    )
    results: list[str | None] = []

    def _claim(name: str) -> None:
        job = claim_next_job(session, name)
        results.append(None if job is None else job.locked_by)

    first = threading.Thread(target=_claim, args=("alpha",))
    second = threading.Thread(target=_claim, args=("beta",))
    first.start()
    second.start()
    first.join()
    second.join()
    _clear_overrides()
    winners = [item for item in results if item is not None]
    assert len(winners) == 1
    assert results.count(None) == 1


def test_lease_renewal_prevents_reclaim(tmp_path: Path) -> None:
    document, session = _ready_document(tmp_path)
    _override_storage(tmp_path)
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        client.post(
            f"/documents/{document.id}/process",
            params={"wait": "false"},
            headers=headers,
        )
        t0 = datetime.now(timezone.utc)
        claimed = claim_next_job(session, "worker-a", now=t0, lease_seconds=60)
        assert claimed is not None
        renewed = renew_lease(session, claimed.id, "worker-a", now=t0 + timedelta(seconds=50))
        stolen = claim_next_job(
            session, "worker-b", now=t0 + timedelta(seconds=70), lease_seconds=60
        )
    finally:
        _clear_overrides()

    assert renewed is True
    assert stolen is None
    assert session.jobs[0].locked_by == "worker-a"
    assert session.jobs[0].locked_at == t0 + timedelta(seconds=50)


def test_expired_lease_without_renewal_is_reclaimed(tmp_path: Path) -> None:
    document, session = _ready_document(tmp_path)
    _override_storage(tmp_path)
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        client.post(
            f"/documents/{document.id}/process",
            params={"wait": "false"},
            headers=headers,
        )
        t0 = datetime.now(timezone.utc)
        first = claim_next_job(session, "worker-a", now=t0, lease_seconds=60)
        stolen = claim_next_job(
            session, "worker-b", now=t0 + timedelta(seconds=70), lease_seconds=60
        )
    finally:
        _clear_overrides()

    assert first.id == stolen.id
    assert stolen.locked_by == "worker-b"
    assert stolen.attempt_count == 2


def test_stale_worker_cannot_succeed_after_reclaim(tmp_path: Path) -> None:
    document, session = _ready_document(tmp_path)
    _override_storage(tmp_path)
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        client.post(
            f"/documents/{document.id}/process",
            params={"wait": "false"},
            headers=headers,
        )
        t0 = datetime.now(timezone.utc)
        claim_next_job(session, "worker-a", now=t0, lease_seconds=60)
        recovered = claim_next_job(
            session, "worker-b", now=t0 + timedelta(seconds=70), lease_seconds=60
        )
        ignored = succeed_job(
            session,
            recovered,
            worker_id="worker-a",
            stamp=t0 + timedelta(seconds=80),
        )
    finally:
        _clear_overrides()

    assert ignored.status == "running"
    assert ignored.locked_by == "worker-b"
    assert document.status == "processing"


def test_stale_worker_cannot_fail_after_reclaim(tmp_path: Path) -> None:
    document, session = _ready_document(tmp_path)
    _override_storage(tmp_path)
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        client.post(
            f"/documents/{document.id}/process",
            params={"wait": "false"},
            headers=headers,
        )
        t0 = datetime.now(timezone.utc)
        claim_next_job(session, "worker-a", now=t0, lease_seconds=60)
        recovered = claim_next_job(
            session, "worker-b", now=t0 + timedelta(seconds=70), lease_seconds=60
        )
        ignored = fail_job(
            session,
            recovered,
            "embedding_failed",
            worker_id="worker-a",
            stamp=t0 + timedelta(seconds=80),
            final=True,
        )
    finally:
        _clear_overrides()

    assert ignored.status == "running"
    assert ignored.locked_by == "worker-b"
    assert ignored.last_error is None
    assert document.status == "processing"


def test_lost_lease_does_not_write_chunks(tmp_path: Path) -> None:
    document, session = _ready_document(tmp_path)
    storage = _override_storage(tmp_path)
    _override_db(session)
    _user, headers = auth_headers(session)

    class _ReclaimingEmbedder:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            vectors = HashEmbedder().embed_texts(texts)
            session.jobs[0].locked_at = datetime.now(timezone.utc) - timedelta(hours=1)
            session.commit()
            taken = claim_next_job(session, "worker-b", lease_seconds=30)
            assert taken is not None
            assert taken.locked_by == "worker-b"
            return vectors

    try:
        client.post(
            f"/documents/{document.id}/process",
            params={"wait": "false"},
            headers=headers,
        )
        claimed = claim_next_job(session, "worker-a")
        stale = run_claimed_job(session, claimed, storage, _ReclaimingEmbedder(), worker_id="worker-a")
    finally:
        _clear_overrides()

    assert stale.status == "running"
    assert stale.locked_by == "worker-b"
    assert document.status == "processing"
    assert session.chunks == []
