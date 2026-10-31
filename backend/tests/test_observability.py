import json
import logging

from fastapi.testclient import TestClient

from app.core.logging import JsonLogFormatter, log_event
from app.core.request_context import REQUEST_ID_HEADER, validate_request_id
from app.main import app

client = TestClient(app)


def test_health_live_does_not_need_database() -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert REQUEST_ID_HEADER in response.headers


def test_health_ready_uses_existing_db_contract() -> None:
    from unittest.mock import MagicMock

    from app.core.database import get_db

    mock_db = MagicMock()

    def override():
        yield mock_db

    app.dependency_overrides[get_db] = override
    try:
        ok = client.get("/health/ready")
        fail_db = MagicMock()
        fail_db.execute.side_effect = Exception("connection failed")

        def override_fail():
            yield fail_db

        app.dependency_overrides[get_db] = override_fail
        down = client.get("/health/ready")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert ok.status_code == 200
    assert ok.json() == {"status": "healthy", "database": "postgresql"}
    assert down.status_code == 503
    assert down.json() == {"detail": "Database unavailable"}
    assert "password" not in down.text.lower()


def test_request_id_generated_and_echoed() -> None:
    response = client.get("/health/live")
    assert response.headers[REQUEST_ID_HEADER]
    custom = client.get("/health/live", headers={REQUEST_ID_HEADER: "req-test.123"})
    assert custom.headers[REQUEST_ID_HEADER] == "req-test.123"


def test_invalid_request_id_is_replaced() -> None:
    assert validate_request_id("not a valid id because spaces") != "not a valid id because spaces"
    assert validate_request_id("ok-id_1") == "ok-id_1"


def test_json_logs_redact_authorization_and_omit_bodies(caplog) -> None:
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        name="app.http",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="http.request",
        args=(),
        exc_info=None,
    )
    record.event = "http.request"
    record.fields = {
        "method": "GET",
        "route": "/health/live",
        "status": 200,
        "duration_ms": 1,
        "authorization": "Bearer super-secret",
        "password": "nope",
    }
    record.request_id = "abc"
    line = formatter.format(record)
    payload = json.loads(line)
    assert payload["event"] == "http.request"
    assert payload["request_id"] == "abc"
    assert payload["method"] == "GET"
    assert "authorization" not in payload
    assert "password" not in payload
    assert "Bearer" not in line
    logger = logging.getLogger("app.test_redact")
    log_event(logger, "http.request", authorization="Bearer leak", route="/health")


def test_docs_routes_are_not_labeled_unmatched(monkeypatch) -> None:
    monkeypatch.setenv("METRICS_ENABLED", "true")
    monkeypatch.setenv("METRICS_TOKEN", "")
    client.get("/docs")
    client.get("/openapi.json")
    client.get("/no-such-phase12-route")
    body = client.get("/metrics").text
    assert 'route="/docs"' in body
    assert 'route="/openapi.json"' in body
    assert 'route="unmatched"' in body
    assert 'status="200"' in body


def test_metrics_endpoint_local_and_protected(monkeypatch) -> None:
    monkeypatch.setenv("METRICS_ENABLED", "true")
    monkeypatch.setenv("METRICS_TOKEN", "")
    client.get("/health/live")
    open_metrics = client.get("/metrics")
    assert open_metrics.status_code == 200
    body = open_metrics.text
    assert "http_requests_total" in body
    import re

    assert re.search(r'route="/health/live"', body)
    assert not re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", body)
    monkeypatch.setenv("METRICS_TOKEN", "metrics-test-token")
    denied = client.get("/metrics")
    allowed = client.get("/metrics", headers={"Authorization": "Bearer metrics-test-token"})
    assert denied.status_code == 401
    assert allowed.status_code == 200
    monkeypatch.setenv("METRICS_ENABLED", "false")
    hidden = client.get("/metrics")
    assert hidden.status_code == 404


def test_search_records_retrieval_metrics(monkeypatch) -> None:
    from tests.auth_helpers import auth_headers
    from tests.test_documents import FakeSession, _clear_overrides, _override_db

    monkeypatch.setenv("METRICS_ENABLED", "true")
    monkeypatch.setenv("METRICS_TOKEN", "")
    session = FakeSession()
    _override_db(session)
    _user, headers = auth_headers(session)
    try:
        client.post("/search", json={"query": "nothing-here", "retrieval_mode": "lexical"}, headers=headers)
        metrics = client.get("/metrics")
    finally:
        _clear_overrides()

    assert metrics.status_code == 200
    assert "retrieval_requests_total" in metrics.text


def test_unhandled_errors_do_not_leak_sql(monkeypatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient as InnerClient

    from app.main import unhandled_error_handler

    tiny = FastAPI()
    tiny.add_exception_handler(Exception, unhandled_error_handler)

    @tiny.get("/boom")
    def boom() -> None:
        raise RuntimeError("SELECT password FROM users")

    isolated = InnerClient(tiny, raise_server_exceptions=False)
    response = isolated.get("/boom")
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "SELECT" not in response.text
    assert "password" not in response.text
