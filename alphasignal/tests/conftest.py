"""Pytest configuration and fixtures for AlphaSignal tests."""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create an isolated test client - empty tmp_path-backed stores, mocked
    OpenAI/CrossEncoder, no real network calls or real corpus data touched.

    Previously used TestClient(app) with the real production lifespan, which
    loaded the actual data/ corpus (FAISS index, BM25 index rebuilt from
    scratch, cross-encoder model) on every single test using this fixture.
    Harmless when the corpus was empty; once a real corpus existed (17k+,
    later 42k+ chunks) this made the full suite take over two hours instead
    of under a minute, on top of tests like
    test_health_status_no_data_on_empty_corpus silently depending on the
    real data/ directory happening to be empty. Rebuilt using the same
    isolated-construction pattern test_api.py's local client fixture already
    used correctly.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key-for-testing")

    from alphasignal.api.dependencies import require_api_key
    from alphasignal.api.routes import health, ingest, metrics, query, sentiment
    from alphasignal.api.state import AppState
    from alphasignal.embeddings.cache import EmbeddingCache
    from alphasignal.embeddings.embedder import Embedder
    from alphasignal.generation.generator import RAGGenerator
    from alphasignal.generation.sentiment import SentimentExtractor
    from alphasignal.ingestion.pipeline import IngestionPipeline
    from alphasignal.monitoring.metrics import MetricsCollector
    from alphasignal.retrieval.reranker import CrossEncoderReranker
    from alphasignal.retrieval.retriever import HybridRetriever
    from alphasignal.store.metadata_store import MetadataStore
    from alphasignal.store.vector_store import VectorStore

    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    with patch("alphasignal.embeddings.embedder.OpenAI"), patch(
        "alphasignal.generation.generator.OpenAI"
    ), patch("alphasignal.generation.sentiment.OpenAI"), patch(
        "alphasignal.retrieval.reranker.CrossEncoder"
    ):
        vector_store = VectorStore(str(tmp_path / "test_index"), dim=1536)
        vector_store.load()
        metadata_store = MetadataStore(str(tmp_path / "test.db"))
        embedding_cache = EmbeddingCache(str(tmp_path / "test_cache.pkl"))
        embedder = Embedder(config, embedding_cache)

        pipeline = IngestionPipeline(
            config,
            embedder=embedder,
            vector_store=vector_store,
            metadata_store=metadata_store,
        )
        retriever = HybridRetriever(config, embedder, vector_store, metadata_store)
        reranker = CrossEncoderReranker()
        generator = RAGGenerator(config)
        sentiment_extractor = SentimentExtractor(config)
        metrics_collector = MetricsCollector()

        app_state = AppState(
            config=config,
            pipeline=pipeline,
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            sentiment_extractor=sentiment_extractor,
            metrics_collector=metrics_collector,
            start_time=time.time(),
        )

        test_app = FastAPI()
        test_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        test_app.state.app_state = app_state

        _auth = [Depends(require_api_key)]
        test_app.include_router(health.router, prefix="/health", tags=["health"])
        test_app.include_router(
            query.router, prefix="/query", tags=["query"], dependencies=_auth
        )
        test_app.include_router(
            sentiment.router,
            prefix="/sentiment",
            tags=["sentiment"],
            dependencies=_auth,
        )
        test_app.include_router(
            ingest.router, prefix="/ingest", tags=["ingest"], dependencies=_auth
        )
        test_app.include_router(
            metrics.router, prefix="/metrics", tags=["metrics"], dependencies=_auth
        )

        with TestClient(test_app) as test_client:
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
