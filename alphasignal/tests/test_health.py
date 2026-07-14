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


def test_health_status_no_data_on_empty_corpus(client):
    """Regression test for audit bug: status previously hardcoded "healthy"
    unconditionally, so a fresh deploy with zero ingested data reported
    itself as fully healthy - indistinguishable from a real, ready
    deployment. The conftest `client` fixture has an empty corpus
    (chunks_indexed=0), so status must now be "no_data", not "healthy"."""
    response = client.get("/health/")
    data = response.json()

    assert data["chunks_indexed"] == 0
    assert data["status"] == "no_data"
    # HTTP status code deliberately stays 200 even when status="no_data" -
    # Railway's healthcheck only looks at the HTTP status, and a fresh
    # deploy must still be allowed to come up before ingestion happens.
    assert response.status_code == 200


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


def _bare_client_with_app_state(app_state):
    """Build a minimal FastAPI test client with a fake app_state attached,
    for exercising health_check()'s status-derivation logic directly
    without needing a fully wired real app (real ingestion is off-limits
    this session, so this is the only way to test a non-empty corpus)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from alphasignal.api.routes import health

    bare_app = FastAPI()
    bare_app.include_router(health.router, prefix="/health")

    with TestClient(bare_app) as bare_client:
        bare_client.app.state.app_state = app_state
        yield bare_client


def test_health_status_healthy_with_chunks_indexed():
    """status must be "healthy" when the corpus actually has data -
    sanity check that the fix doesn't just always report "no_data"."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    fake_metadata_store = MagicMock()
    fake_metadata_store.count.return_value = 42
    fake_vector_store = MagicMock()
    fake_vector_store.index = MagicMock()  # non-None = loaded
    fake_retriever = SimpleNamespace(
        vector_store=fake_vector_store, metadata_store=fake_metadata_store
    )
    fake_app_state = SimpleNamespace(retriever=fake_retriever, start_time=0.0)

    for bare_client in _bare_client_with_app_state(fake_app_state):
        response = bare_client.get("/health/")
        data = response.json()

    assert response.status_code == 200
    assert data["chunks_indexed"] == 42
    assert data["status"] == "healthy"


def test_health_status_unhealthy_when_store_read_fails():
    """status must be "unhealthy" (not "healthy" or "no_data") when the
    metadata store itself can't be reached, distinct from a merely-empty
    corpus - a store failure is a real problem, not just "not ingested
    yet"."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    fake_metadata_store = MagicMock()
    fake_metadata_store.count.side_effect = RuntimeError("db connection lost")
    fake_retriever = SimpleNamespace(
        vector_store=None, metadata_store=fake_metadata_store
    )
    fake_app_state = SimpleNamespace(retriever=fake_retriever, start_time=0.0)

    for bare_client in _bare_client_with_app_state(fake_app_state):
        response = bare_client.get("/health/")
        data = response.json()

    assert response.status_code == 200
    assert data["status"] == "unhealthy"
    assert data["sqlite_connected"] is False


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
