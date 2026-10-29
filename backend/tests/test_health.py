from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_healthy_status() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "enterprise-ai-platform",
    }


def test_docs_returns_http_200() -> None:
    response = client.get("/docs")

    assert response.status_code == 200
