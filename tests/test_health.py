"""Tests for the /health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.core.config import APP_NAME


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["name"] == APP_NAME
    assert isinstance(body["version"], str)
    assert body["version"]
