from fastapi.testclient import TestClient

from app.main import app


def test_health_ok_after_warmup():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}
