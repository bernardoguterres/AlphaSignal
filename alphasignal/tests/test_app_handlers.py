"""Tests for the FastAPI app-level exception handlers in alphasignal.api.app."""

import json

import pytest
from unittest.mock import MagicMock

from alphasignal.api.app import general_exception_handler


@pytest.mark.asyncio
async def test_general_exception_handler_returns_500_with_generic_body():
    """Test that unhandled exceptions are converted to a generic 500 response.

    Audit finding (2026-07-14): detail used to be str(exc), leaking internal
    exception text (file paths, SQL fragments, credentials in connection
    strings, etc.) to every caller. The response body must now be generic;
    the real exception is still logged server-side via exc_info=True.
    """
    request = MagicMock()
    exc = ValueError("unexpected boom")

    response = await general_exception_handler(request, exc)

    assert response.status_code == 500
    body = json.loads(response.body)
    assert body["error"] == "Internal Server Error"
    assert body["code"] == "INTERNAL_ERROR"
    assert "unexpected boom" not in body["detail"]
