"""Integration tests for API endpoints."""

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from alphasignal.generation import GenerationResult
from alphasignal.ingestion.pipeline import IngestResult
from alphasignal.retrieval import RetrievedChunk


@pytest.fixture
def client(tmp_path):
    """Create test client with mocked dependencies."""
    import time
    import yaml
    from pathlib import Path
    from alphasignal.api.state import AppState
    from alphasignal.embeddings.cache import EmbeddingCache
    from alphasignal.embeddings.embedder import Embedder
    from alphasignal.generation.generator import RAGGenerator
    from alphasignal.generation.sentiment import SentimentExtractor
    from alphasignal.ingestion.pipeline import IngestionPipeline
    from alphasignal.monitoring.metrics import MetricsCollector
    from alphasignal.retrieval.retriever import HybridRetriever
    from alphasignal.retrieval.reranker import CrossEncoderReranker
    from alphasignal.store.metadata_store import MetadataStore
    from alphasignal.store.vector_store import VectorStore

    # Load config
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Mock all external dependencies
    with patch("alphasignal.embeddings.embedder.OpenAI"), patch(
        "alphasignal.generation.generator.OpenAI"
    ), patch("alphasignal.generation.sentiment.OpenAI"), patch(
        "alphasignal.retrieval.reranker.CrossEncoder"
    ):

        # Create test client without lifespan
        test_app = FastAPI()
        test_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Manually initialize app state
        from alphasignal.api.routes import health, query, sentiment, ingest, metrics

        # Create minimal stores using tmp_path
        vector_store = VectorStore(str(tmp_path / "test_index"), dim=1536)
        vector_store.load()

        metadata_store = MetadataStore(str(tmp_path / "test.db"))

        embedding_cache = EmbeddingCache(str(tmp_path / "test_cache.pkl"))
        embedder = Embedder(config, embedding_cache)

        pipeline = IngestionPipeline(config)
        retriever = HybridRetriever(config, embedder, vector_store, metadata_store)
        reranker = CrossEncoderReranker()
        generator = RAGGenerator(config)
        sentiment_extractor = SentimentExtractor(config)
        metrics_collector = MetricsCollector()

        # Create app state
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

        # Attach to app
        test_app.state.app_state = app_state

        # Register routes
        test_app.include_router(health.router, prefix="/health", tags=["health"])
        test_app.include_router(query.router, prefix="/query", tags=["query"])
        test_app.include_router(
            sentiment.router, prefix="/sentiment", tags=["sentiment"]
        )
        test_app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
        test_app.include_router(metrics.router, prefix="/metrics", tags=["metrics"])

        yield TestClient(test_app)


@pytest.fixture
def mock_retrieved_chunks():
    """Create mock retrieved chunks."""
    return [
        RetrievedChunk(
            chunk_id="aapl_10k_test_0001",
            ticker="AAPL",
            text="Apple Inc. reported revenue of $394.3 billion for fiscal year 2024.",
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_7",
            date=date(2024, 10, 31),
            url=None,
            dense_score=0.95,
            sparse_score=0.87,
            hybrid_score=0.92,
            final_score=0.94,
        ),
        RetrievedChunk(
            chunk_id="aapl_10k_test_0002",
            ticker="AAPL",
            text="iPhone revenue reached $200.6 billion, up 12% from the prior year.",
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_7",
            date=date(2024, 10, 31),
            url=None,
            dense_score=0.89,
            sparse_score=0.82,
            hybrid_score=0.86,
            final_score=0.88,
        ),
        RetrievedChunk(
            chunk_id="aapl_10k_test_0003",
            ticker="AAPL",
            text="Services revenue grew to $85.2 billion.",
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_7",
            date=date(2024, 10, 31),
            url=None,
            dense_score=0.78,
            sparse_score=0.75,
            hybrid_score=0.77,
            final_score=0.80,
        ),
    ]


def test_query_endpoint_returns_200(client, mock_retrieved_chunks):
    """Test that query endpoint returns 200 with valid response."""
    # Mock retriever
    with patch.object(
        client.app.state.app_state.retriever, "retrieve"
    ) as mock_retrieve:
        mock_retrieve.return_value = mock_retrieved_chunks

        # Mock reranker
        with patch.object(client.app.state.app_state.reranker, "rerank") as mock_rerank:
            mock_rerank.return_value = mock_retrieved_chunks[:2]  # Return top 2

            # Mock generator
            with patch.object(
                client.app.state.app_state.generator, "generate"
            ) as mock_generate:
                mock_generate.return_value = GenerationResult(
                    answer="Apple reported revenue of $394.3 billion [Source 1].",
                    cited_chunks=[mock_retrieved_chunks[0]],
                    prompt_tokens=100,
                    completion_tokens=50,
                    model="gpt-4o-mini",
                )

                # Make request
                response = client.post(
                    "/query/",
                    json={
                        "query": "What is Apple revenue?",
                        "ticker_filter": ["AAPL"],
                        "top_k": 5,
                    },
                )

                # Assertions
                assert response.status_code == 200
                data = response.json()
                assert "answer" in data
                assert "citations" in data
                assert "latency_ms" in data
                assert data["latency_ms"] >= 0


def test_query_endpoint_returns_citations(client, mock_retrieved_chunks):
    """Test that query endpoint returns citations."""
    # Mock retriever, reranker, and generator
    with patch.object(
        client.app.state.app_state.retriever, "retrieve"
    ) as mock_retrieve, patch.object(
        client.app.state.app_state.reranker, "rerank"
    ) as mock_rerank, patch.object(
        client.app.state.app_state.generator, "generate"
    ) as mock_generate:

        mock_retrieve.return_value = mock_retrieved_chunks
        mock_rerank.return_value = mock_retrieved_chunks[:2]
        mock_generate.return_value = GenerationResult(
            answer="Revenue grew [Source 1] and iPhone sales increased [Source 2].",
            cited_chunks=[mock_retrieved_chunks[0], mock_retrieved_chunks[1]],
            prompt_tokens=100,
            completion_tokens=50,
            model="gpt-4o-mini",
        )

        response = client.post(
            "/query/", json={"query": "What is Apple revenue?", "top_k": 5}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["citations"]) == 2
        assert data["citations"][0]["ticker"] == "AAPL"
        assert "excerpt" in data["citations"][0]


def test_query_endpoint_handles_empty_retrieval(client):
    """Test that query endpoint handles no results gracefully."""
    # Mock retriever to return empty list
    with patch.object(
        client.app.state.app_state.retriever, "retrieve"
    ) as mock_retrieve:
        mock_retrieve.return_value = []

        response = client.post(
            "/query/", json={"query": "What is Apple revenue?", "top_k": 5}
        )

        assert response.status_code == 200
        data = response.json()
        assert "No relevant information" in data["answer"]
        assert len(data["citations"]) == 0


def test_sentiment_endpoint_returns_200(client):
    """Test that sentiment endpoint returns 200."""
    from alphasignal.ingestion import Chunk
    from alphasignal.api.schemas import SentimentSignal

    # Mock metadata store
    mock_chunks = [
        Chunk(
            chunk_id=f"aapl_test_{i}",
            ticker="AAPL",
            text=f"Test chunk {i}",
            token_count=50,
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_7",
            date=date(2024, 10, 31),
            url=None,
            chunk_index=i,
            total_chunks=5,
        )
        for i in range(5)
    ]

    with patch.object(
        client.app.state.app_state.pipeline.metadata_store, "get_chunks_by_ticker"
    ) as mock_get_chunks:
        mock_get_chunks.return_value = mock_chunks

        # Mock sentiment extractor
        mock_signals = [
            SentimentSignal(
                ticker="AAPL",
                date=date(2024, 10, 31),
                score=0.75,
                confidence=0.85,
                source="SEC EDGAR",
                doc_type="10-K",
                key_positive=["growth", "strong"],
                key_negative=[],
                summary="Positive sentiment",
            )
        ]

        with patch.object(
            client.app.state.app_state.sentiment_extractor, "extract_ticker_sentiment"
        ) as mock_extract:
            mock_extract.return_value = mock_signals

            response = client.get("/sentiment/AAPL")

            assert response.status_code == 200
            data = response.json()
            assert data["ticker"] == "AAPL"
            assert len(data["signals"]) == 1
            assert data["latest_score"] == 0.75
            assert data["latency_ms"] >= 0


def test_sentiment_endpoint_invalid_ticker(client):
    """Test that sentiment endpoint rejects invalid ticker."""
    response = client.get("/sentiment/INVALID_TICKER_TOO_LONG")

    assert response.status_code == 422


def test_sentiment_endpoint_unknown_ticker(client):
    """Test that sentiment endpoint returns 404 for unknown ticker."""
    response = client.get("/sentiment/ZZZZ")

    assert response.status_code == 404


def test_ingest_endpoint_triggers_pipeline(client):
    """Test that ingest endpoint triggers pipeline."""
    # Mock pipeline
    with patch.object(
        client.app.state.app_state.pipeline, "full_ingest"
    ) as mock_ingest:
        mock_ingest.return_value = IngestResult(
            ticker="AAPL", chunks_created=10, chunks_embedded=10, chunks_stored=10
        )

        # Mock BM25 rebuild
        with patch.object(client.app.state.app_state.retriever, "build_bm25_index"):
            response = client.post("/ingest/AAPL")

            assert response.status_code == 200
            data = response.json()
            assert data["ticker"] == "AAPL"
            assert data["status"] == "completed"
            assert data["chunks_created"] == 10
            assert mock_ingest.called


def test_ingest_batch_processes_all_tickers(client):
    """Test that batch ingest processes all tickers."""
    # Mock pipeline
    with patch.object(
        client.app.state.app_state.pipeline, "full_ingest"
    ) as mock_ingest:
        mock_ingest.return_value = IngestResult(
            ticker="TEST", chunks_created=5, chunks_embedded=5, chunks_stored=5
        )

        # Mock BM25 rebuild
        with patch.object(client.app.state.app_state.retriever, "build_bm25_index"):
            response = client.post(
                "/ingest/batch", json={"tickers": ["AAPL", "MSFT", "GOOGL"]}
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) == 3
            assert data["total_latency_ms"] >= 0


def test_all_responses_include_latency_ms(client, mock_retrieved_chunks):
    """Test that all endpoints return latency_ms."""
    # Test query endpoint
    with patch.object(
        client.app.state.app_state.retriever, "retrieve"
    ) as mock_retrieve, patch.object(
        client.app.state.app_state.reranker, "rerank"
    ) as mock_rerank, patch.object(
        client.app.state.app_state.generator, "generate"
    ) as mock_generate:

        mock_retrieve.return_value = mock_retrieved_chunks
        mock_rerank.return_value = mock_retrieved_chunks[:2]
        mock_generate.return_value = GenerationResult(
            answer="Test answer",
            cited_chunks=[],
            prompt_tokens=10,
            completion_tokens=10,
            model="gpt-4o-mini",
        )

        query_response = client.post(
            "/query/", json={"query": "What is the revenue?", "top_k": 5}
        )
        assert "latency_ms" in query_response.json()
        assert query_response.json()["latency_ms"] >= 0

    # Test sentiment endpoint
    from alphasignal.ingestion import Chunk

    mock_chunks = [
        Chunk(
            chunk_id="test",
            ticker="AAPL",
            text="test",
            token_count=10,
            doc_type="10-K",
            source="SEC EDGAR",
            section=None,
            date=date.today(),
            url=None,
            chunk_index=0,
            total_chunks=1,
        )
    ]

    with patch.object(
        client.app.state.app_state.pipeline.metadata_store,
        "get_chunks_by_ticker",
        return_value=mock_chunks,
    ), patch.object(
        client.app.state.app_state.sentiment_extractor,
        "extract_ticker_sentiment",
        return_value=[],
    ):
        sentiment_response = client.get("/sentiment/AAPL")
        assert "latency_ms" in sentiment_response.json()
        assert sentiment_response.json()["latency_ms"] >= 0

    # Test ingest endpoint
    with patch.object(
        client.app.state.app_state.pipeline,
        "full_ingest",
        return_value=IngestResult("AAPL", 5, 5, 5),
    ), patch.object(client.app.state.app_state.retriever, "build_bm25_index"):
        ingest_response = client.post("/ingest/AAPL")
        assert "latency_ms" in ingest_response.json()
        assert ingest_response.json()["latency_ms"] >= 0
