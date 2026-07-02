"""Tests for the FastAPI app-level exception handlers in alphasignal.api.app."""

import json

import pytest
from unittest.mock import MagicMock

from alphasignal.api.app import alphasignal_error_handler, general_exception_handler
from alphasignal.api.exceptions import DatabaseError


@pytest.mark.asyncio
async def test_alphasignal_error_handler_returns_500_with_error_body():
    """Test that AlphaSignalError subclasses are converted to a structured 500 response."""
    request = MagicMock()
    exc = DatabaseError("connection lost", detail="sqlite locked")

    response = await alphasignal_error_handler(request, exc)

    assert response.status_code == 500
    body = json.loads(response.body)
    assert body["error"] == "connection lost"
    assert body["code"] == "DB_ERROR"
    assert body["detail"] == "sqlite locked"


@pytest.mark.asyncio
async def test_general_exception_handler_returns_500_with_generic_body():
    """Test that unhandled exceptions are converted to a generic 500 response."""
    request = MagicMock()
    exc = ValueError("unexpected boom")

    response = await general_exception_handler(request, exc)

    assert response.status_code == 500
    body = json.loads(response.body)
    assert body["error"] == "Internal Server Error"
    assert body["code"] == "INTERNAL_ERROR"
    assert "unexpected boom" in body["detail"]
