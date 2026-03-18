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
