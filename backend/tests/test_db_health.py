from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.core.database import get_db
from app.main import app

client = TestClient(app)


def test_db_health_returns_healthy_status() -> None:
    mock_db = MagicMock()

    def override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db
    try:
        response = client.get("/db-health")
    finally:
        app.dependency_overrides.clear()

    mock_db.execute.assert_called_once()
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "database": "postgresql"}


def test_db_health_returns_503_when_database_unavailable() -> None:
    mock_db = MagicMock()
    mock_db.execute.side_effect = Exception("connection failed")

    def override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db
    try:
        response = client.get("/db-health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable"}
    assert "password" not in response.text.lower()
    assert "postgresql+psycopg" not in response.text
