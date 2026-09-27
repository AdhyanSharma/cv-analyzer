from fastapi.testclient import TestClient

from backend.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "cv-analyzer-api"
    assert body["version"] == "v15"


def test_openapi_is_available():
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"]["title"]


def test_docs_is_available():
    response = client.get("/docs")

    assert response.status_code == 200
