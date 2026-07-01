"""Pytest configuration and fixtures for AlphaSignal tests."""

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """Create a test client for the FastAPI app."""
    # Set dummy API key for tests
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key-for-testing")

    from alphasignal.api.app import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def tmp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config():
    """Provide a mock configuration for testing."""
    return {
        "tickers": ["AAPL", "MSFT"],
        "ingestion": {
            "edgar": {
                "filing_types": ["10-K"],
                "years_back": 1,
                "rate_limit_delay": 0.1,
            },
            "news": {
                "sources": [],
                "max_articles_per_ticker": 10,
                "max_age_days": 30,
            },
        },
        "chunking": {
            "target_tokens": 100,
            "min_tokens": 50,
            "max_tokens": 150,
            "overlap_tokens": 10,
        },
        "embeddings": {
            "model": "text-embedding-ada-002",
            "batch_size": 10,
            "max_retries": 2,
            "retry_delay": 0.5,
        },
        "retrieval": {
            "dense_candidates": 10,
            "sparse_candidates": 10,
            "rerank_candidates": 5,
            "hybrid_weights": {"bm25": 0.5, "dense": 0.5},
        },
        "generation": {
            "model": "gpt-4o-mini",
            "max_tokens": 500,
            "temperature": 0.0,
            "sentiment_cache_hours": 1,
        },
        "storage": {
            "faiss_index_path": "test_data/faiss_index",
            "sqlite_db_path": "test_data/metadata.db",
            "embeddings_cache_path": "test_data/embeddings_cache",
        },
    }
