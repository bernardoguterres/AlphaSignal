"""Tests for the health check endpoint."""

import pytest
from alphasignal.api.schemas import HealthResponse


def test_health_returns_200(client):
    """Test that health endpoint returns 200 status code."""
    response = client.get("/health/")
    assert response.status_code == 200


def test_health_response_schema(client):
    """Test that health response matches HealthResponse schema exactly."""
    response = client.get("/health/")
    data = response.json()

    # Validate against Pydantic model
    health_response = HealthResponse(**data)

    # Check all required fields are present
    assert "status" in data
    assert "version" in data
    assert "faiss_index_loaded" in data
    assert "sqlite_connected" in data
    assert "chunks_indexed" in data
    assert "uptime_seconds" in data


def test_health_status_healthy(client):
    """Test that health endpoint returns healthy status."""
    response = client.get("/health/")
    data = response.json()

    assert data["status"] == "healthy"


def test_health_uptime_positive(client):
    """Test that uptime_seconds is greater than zero."""
    response = client.get("/health/")
    data = response.json()

    assert data["uptime_seconds"] > 0


def test_health_reports_sqlite_state_from_app_state(client):
    """Test that health reflects the real app_state's metadata store, not hardcoded defaults."""
    response = client.get("/health/")
    data = response.json()

    # The conftest `client` fixture builds a fully wired app_state.
    assert data["sqlite_connected"] is True
    assert isinstance(data["chunks_indexed"], int)
    assert isinstance(data["faiss_index_loaded"], bool)


def test_health_handles_missing_app_state_gracefully():
    """Test that health_check falls back to defaults when app_state isn't set.

    This covers the branch where request.app.state has no app_state attribute
    (e.g. app not fully started yet), which should not raise.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from alphasignal.api.routes import health

    bare_app = FastAPI()
    bare_app.include_router(health.router, prefix="/health")

    with TestClient(bare_app) as bare_client:
        response = bare_client.get("/health/")

    assert response.status_code == 200
    data = response.json()
    assert data["faiss_index_loaded"] is False
    assert data["sqlite_connected"] is False
    assert data["chunks_indexed"] == 0
