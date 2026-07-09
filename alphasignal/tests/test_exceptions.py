"""Tests for custom AlphaSignal exceptions."""

from alphasignal.api.exceptions import (
    AlphaSignalError,
    DatabaseError,
    GenerationError,
    IndexNotLoadedError,
    IngestionError,
)


def test_alphasignal_error_stores_message_code_and_detail():
    """Test that the base error exposes message, code, detail, and extra kwargs."""
    err = AlphaSignalError(
        message="Something broke", code="CUSTOM_CODE", detail="stack trace here", foo="bar"
    )

    assert err.message == "Something broke"
    assert err.code == "CUSTOM_CODE"
    assert err.detail == "stack trace here"
    assert err.extra == {"foo": "bar"}
    assert str(err) == "Something broke"


def test_openai_error_has_fixed_code():
    """Test that GenerationError always carries the OPENAI_ERROR code."""
    err = GenerationError("rate limited", detail="429 from OpenAI")

    assert err.code == "OPENAI_ERROR"
    assert err.message == "rate limited"
    assert err.detail == "429 from OpenAI"
    assert isinstance(err, AlphaSignalError)


def test_index_not_loaded_error_default_message():
    """Test that IndexNotLoadedError has a sensible default message."""
    err = IndexNotLoadedError()

    assert err.code == "INDEX_NOT_LOADED"
    assert err.message == "FAISS index not loaded"
    assert err.detail is None


def test_database_error_has_fixed_code():
    """Test that DatabaseError carries the DB_ERROR code."""
    err = DatabaseError("connection lost", detail="sqlite locked")

    assert err.code == "DB_ERROR"
    assert err.message == "connection lost"


def test_ingestion_error_includes_source_in_extra():
    """Test that IngestionError stores its source in the extra kwargs."""
    err = IngestionError("failed to fetch filing", source="EDGAR", detail="timeout")

    assert err.code == "INGESTION_ERROR"
    assert err.extra == {"source": "EDGAR"}
    assert err.detail == "timeout"
