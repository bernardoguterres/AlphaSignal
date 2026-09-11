"""Integration tests for API endpoints."""

import itertools
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

        # Share the tmp_path-backed stores with the pipeline - without this
        # it builds its own from config.yaml's real storage paths, loading
        # the actual production FAISS index/corpus on every test (harmless
        # when that corpus was empty; became a multi-hour full-suite run
        # once a real 17k+, later 42k+ chunk corpus existed on disk).
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

        # Register routes - mirror production wiring in app.py exactly,
        # including the auth dependency (all routers except /health), so
        # auth behavior tested here matches the real app.
        from fastapi import Depends
        from alphasignal.api.dependencies import require_api_key

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
                        "ticker_filter": "AAPL",
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


def test_query_endpoint_unknown_ticker_filter_is_not_404(client):
    """Deliberate policy: ticker_filter is a search filter, not a resource
    identifier, so an unrecognized value must behave like any other filter
    that excludes everything (200, empty citations) - not a 404, unlike
    /sentiment/{ticker} and /ingest/{ticker}."""
    with patch.object(
        client.app.state.app_state.retriever, "retrieve"
    ) as mock_retrieve:
        mock_retrieve.return_value = []

        response = client.post(
            "/query/",
            json={
                "query": "What is Zzzz revenue?",
                "ticker_filter": "ZZZZ",
                "top_k": 5,
            },
        )

        assert response.status_code == 200
        assert len(response.json()["citations"]) == 0
        # The unvalidated ticker_filter was still passed straight through
        # to the retriever, not silently dropped.
        assert mock_retrieve.call_args.kwargs["ticker"] == "ZZZZ"


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


def test_cors_does_not_combine_wildcard_origin_with_credentials():
    """Audit finding: allow_origins=["*"] + allow_credentials=True is an
    invalid combination per the CORS spec (browsers reject it outright) and
    was unnecessary here since auth is via X-API-Key headers, not cookies."""
    from starlette.middleware.cors import CORSMiddleware as CORSMiddlewareClass

    from alphasignal.api.app import app

    cors_middleware = next(
        m for m in app.user_middleware if m.cls is CORSMiddlewareClass
    )

    assert cors_middleware.kwargs["allow_origins"] == ["*"]
    assert cors_middleware.kwargs["allow_credentials"] is False


def test_ingest_endpoint_rejects_ticker_not_in_allowlist(client):
    """Audit finding: /ingest/{ticker} had no allowlist check at all, unlike
    /sentiment/{ticker} - an arbitrary ticker string could trigger real
    EDGAR/news fetches and OpenAI embedding spend."""
    with patch.object(
        client.app.state.app_state.pipeline, "full_ingest"
    ) as mock_ingest:
        response = client.post("/ingest/NOTREAL")

        assert response.status_code == 404
        assert not mock_ingest.called


def test_ingest_batch_rejects_ticker_not_in_allowlist_without_calling_pipeline(client):
    """A ticker outside the allowlist in a batch request must be marked
    failed and must never reach the ingestion pipeline (no EDGAR/news
    fetch, no OpenAI spend); other tickers in the batch still proceed."""
    with patch.object(
        client.app.state.app_state.pipeline, "full_ingest"
    ) as mock_ingest:
        mock_ingest.return_value = IngestResult(
            ticker="AAPL", chunks_created=5, chunks_embedded=5, chunks_stored=5
        )

        with patch.object(client.app.state.app_state.retriever, "build_bm25_index"):
            response = client.post(
                "/ingest/batch", json={"tickers": ["AAPL", "NOTREAL"]}
            )

        assert response.status_code == 200
        data = response.json()
        results_by_ticker = {r["ticker"]: r for r in data["results"]}
        assert results_by_ticker["NOTREAL"]["status"] == "failed"
        assert results_by_ticker["AAPL"]["status"] == "completed"
        # Pipeline must only have been invoked for the allowlisted ticker.
        assert mock_ingest.call_count == 1
        assert mock_ingest.call_args.args[0] == "AAPL"


def test_sentiment_endpoint_no_chunks_returns_empty_signals(client):
    """Test that sentiment endpoint returns empty signals when no chunks exist."""
    with patch.object(
        client.app.state.app_state.pipeline.metadata_store,
        "get_chunks_by_ticker",
        return_value=[],
    ):
        response = client.get("/sentiment/AAPL")

    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert data["signals"] == []
    assert data["latest_score"] is None
    # Regression test: data_available must be explicitly False when no
    # chunks exist for the ticker - distinct from a genuinely neutral
    # score, so a client can't accidentally coalesce "never ingested" into
    # 0.0 the way latest_score=None alone made easy to do (audit finding).
    assert data["data_available"] is False


def test_sentiment_endpoint_data_available_true_when_chunks_exist(client):
    """data_available must be True (the default) whenever real data backs
    the response - sanity check the flag isn't just always False."""
    from alphasignal.ingestion import Chunk

    mock_chunk = Chunk(
        chunk_id="aapl_10k_test_0001",
        ticker="AAPL",
        text="Apple reported strong quarterly revenue growth.",
        token_count=10,
        doc_type="10-K",
        source="SEC EDGAR",
        section="MD&A",
        date=date(2024, 1, 1),
        url=None,
        chunk_index=0,
        total_chunks=1,
    )

    with patch.object(
        client.app.state.app_state.pipeline.metadata_store,
        "get_chunks_by_ticker",
        return_value=[mock_chunk],
    ), patch.object(
        client.app.state.app_state.sentiment_extractor,
        "extract_ticker_sentiment",
        return_value=[],
    ):
        response = client.get("/sentiment/AAPL")

    assert response.status_code == 200
    data = response.json()
    assert data["data_available"] is True


def test_sentiment_endpoint_date_range_query(client):
    """Test that sentiment endpoint uses the date-range query path when dates given."""
    from alphasignal.ingestion import Chunk

    mock_chunks = [
        Chunk(
            chunk_id="aapl_range_0",
            ticker="AAPL",
            text="Range chunk",
            token_count=10,
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_1",
            date=date(2024, 5, 1),
            url=None,
            chunk_index=0,
            total_chunks=1,
        )
    ]

    with patch.object(
        client.app.state.app_state.pipeline.metadata_store,
        "get_chunks_by_date_range",
        return_value=mock_chunks,
    ) as mock_range, patch.object(
        client.app.state.app_state.sentiment_extractor,
        "extract_ticker_sentiment",
        return_value=[],
    ):
        response = client.get("/sentiment/AAPL?date_from=2024-01-01&date_to=2024-12-31")

    assert response.status_code == 200
    assert mock_range.called
    call_kwargs = mock_range.call_args.kwargs
    assert call_kwargs["start"] == date(2024, 1, 1)
    assert call_kwargs["end"] == date(2024, 12, 31)


def test_sentiment_endpoint_propagates_errors(client):
    """Test that sentiment endpoint records the error and re-raises the failure.

    The route's except-block calls metrics_collector.record_error() and then
    re-raises. This test app doesn't register alphasignal's exception
    handlers, so the underlying exception surfaces directly to the client.
    """
    with patch.object(
        client.app.state.app_state.pipeline.metadata_store,
        "get_chunks_by_ticker",
        side_effect=RuntimeError("db exploded"),
    ):
        with pytest.raises(RuntimeError, match="db exploded"):
            client.get("/sentiment/AAPL")

    assert (
        client.app.state.app_state.metrics_collector.get_summary()["errors"]["count"]
        == 1
    )


def _sentiment_signal(score, confidence, reliable, day=1):
    from alphasignal.api.schemas import SentimentSignal

    return SentimentSignal(
        ticker="AAPL",
        date=date(2024, 1, day),
        score=score,
        confidence=confidence,
        source="SEC EDGAR",
        doc_type="10-K",
        key_positive=[],
        key_negative=[],
        summary="test",
        reliable=reliable,
    )


def test_sentiment_endpoint_genuine_neutral_is_ok_not_degraded(client):
    """A real, reliable score of 0.0 is a valid successful result - not
    degraded, and distinct from 'no data' (data_available stays True)."""
    from alphasignal.ingestion import Chunk

    mock_chunk = Chunk(
        chunk_id="aapl_neutral_0001",
        ticker="AAPL",
        text="Results were in line with expectations.",
        token_count=10,
        doc_type="10-K",
        source="SEC EDGAR",
        section="item_7",
        date=date(2024, 1, 1),
        url=None,
        chunk_index=0,
        total_chunks=1,
    )
    neutral_signal = _sentiment_signal(score=0.0, confidence=0.6, reliable=True)

    with patch.object(
        client.app.state.app_state.pipeline.metadata_store,
        "get_chunks_by_ticker",
        return_value=[mock_chunk],
    ), patch.object(
        client.app.state.app_state.sentiment_extractor,
        "extract_ticker_sentiment",
        return_value=[neutral_signal],
    ):
        response = client.get("/sentiment/AAPL")

    assert response.status_code == 200
    data = response.json()
    assert data["data_available"] is True
    assert data["status"] == "ok"
    assert data["degraded"] is False
    assert data["degradation_reason"] is None
    assert data["latest_score"] == 0.0
    assert data["reliable_chunk_count"] == 1
    assert data["total_chunk_count"] == 1


def test_sentiment_endpoint_partial_degradation_visible(client):
    """One failed chunk plus reliable chunks must be explicitly flagged as
    degraded, while still surfacing the latest reliable prediction."""
    from alphasignal.ingestion import Chunk

    chunks = [
        Chunk(
            chunk_id=f"aapl_mixed_{i:04d}",
            ticker="AAPL",
            text=f"Chunk {i}",
            token_count=10,
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_7",
            date=date(2024, 1, 1 + i),
            url=None,
            chunk_index=i,
            total_chunks=2,
        )
        for i in range(2)
    ]
    signals = [
        _sentiment_signal(
            score=0.4, confidence=0.7, reliable=True, day=2
        ),  # most recent
        _sentiment_signal(score=0.0, confidence=0.0, reliable=False, day=1),
    ]

    with patch.object(
        client.app.state.app_state.pipeline.metadata_store,
        "get_chunks_by_ticker",
        return_value=chunks,
    ), patch.object(
        client.app.state.app_state.sentiment_extractor,
        "extract_ticker_sentiment",
        return_value=signals,
    ):
        response = client.get("/sentiment/AAPL")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["degraded"] is True
    assert data["degradation_reason"] == "partial_extraction_failure"
    # The reliable prediction is preserved, not discarded.
    assert data["latest_score"] == 0.4
    assert data["reliable_chunk_count"] == 1
    assert data["total_chunk_count"] == 2
    assert len(data["signals"]) == 2


def test_sentiment_endpoint_full_degradation_does_not_masquerade_as_neutral(client):
    """When every chunk-level extraction failed, latest_score must be None
    - never fabricated as 0.0 - and the response must say so explicitly."""
    from alphasignal.ingestion import Chunk

    mock_chunk = Chunk(
        chunk_id="aapl_alldegraded_0001",
        ticker="AAPL",
        text="Some filing text.",
        token_count=10,
        doc_type="10-K",
        source="SEC EDGAR",
        section="item_7",
        date=date(2024, 1, 1),
        url=None,
        chunk_index=0,
        total_chunks=1,
    )
    failed_signal = _sentiment_signal(score=0.0, confidence=0.0, reliable=False)

    with patch.object(
        client.app.state.app_state.pipeline.metadata_store,
        "get_chunks_by_ticker",
        return_value=[mock_chunk],
    ), patch.object(
        client.app.state.app_state.sentiment_extractor,
        "extract_ticker_sentiment",
        return_value=[failed_signal],
    ):
        response = client.get("/sentiment/AAPL")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["degraded"] is True
    assert data["degradation_reason"] == "full_extraction_failure"
    assert (
        data["latest_score"] is None
    ), "A fully-failed extraction must not fabricate a 0.0 neutral score"
    assert data["reliable_chunk_count"] == 0
    assert data["total_chunk_count"] == 1


def test_sentiment_endpoint_no_data_still_distinguishable_from_degraded(client):
    """No ingested chunks at all remains its own distinct state, separate
    from a degraded extraction over real data."""
    with patch.object(
        client.app.state.app_state.pipeline.metadata_store,
        "get_chunks_by_ticker",
        return_value=[],
    ):
        response = client.get("/sentiment/AAPL")

    assert response.status_code == 200
    data = response.json()
    assert data["data_available"] is False
    assert data["status"] == "no_data"
    assert data["degraded"] is False
    assert data["latest_score"] is None


def test_sentiment_endpoint_legacy_consumer_reading_only_old_fields_unaffected(client):
    """A consumer that only reads the pre-existing fields (score, sources,
    confidence via signals, latency, data_available) must keep working
    unmodified - the new fields are purely additive."""
    from alphasignal.ingestion import Chunk

    mock_chunk = Chunk(
        chunk_id="aapl_legacy_0001",
        ticker="AAPL",
        text="Legacy consumer chunk.",
        token_count=10,
        doc_type="10-K",
        source="SEC EDGAR",
        section="item_7",
        date=date(2024, 1, 1),
        url=None,
        chunk_index=0,
        total_chunks=1,
    )
    signal = _sentiment_signal(score=0.3, confidence=0.6, reliable=True)

    with patch.object(
        client.app.state.app_state.pipeline.metadata_store,
        "get_chunks_by_ticker",
        return_value=[mock_chunk],
    ), patch.object(
        client.app.state.app_state.sentiment_extractor,
        "extract_ticker_sentiment",
        return_value=[signal],
    ):
        response = client.get("/sentiment/AAPL")

    data = response.json()
    # Legacy field set, exactly as before this change.
    legacy_view = {
        "ticker": data["ticker"],
        "latest_score": data["latest_score"],
        "latency_ms": data["latency_ms"] >= 0,
        "data_available": data["data_available"],
        "signal_count": len(data["signals"]),
        "signal_score": data["signals"][0]["score"],
        "signal_confidence": data["signals"][0]["confidence"],
        "signal_source": data["signals"][0]["source"],
    }
    assert legacy_view == {
        "ticker": "AAPL",
        "latest_score": 0.3,
        "latency_ms": True,
        "data_available": True,
        "signal_count": 1,
        "signal_score": 0.3,
        "signal_confidence": 0.6,
        "signal_source": "SEC EDGAR",
    }


class TestSentimentContractInvariants:
    """Property-style verification of the sentiment response contract
    (objective 5 of the 2026-09-11 persistence/contract-verification pass).

    Drives GET /sentiment/{ticker} through every combination of reliable/
    unreliable per-chunk signals up to a small size, and asserts the
    response-level invariants hold for all of them - not just the
    hand-picked examples in the earlier degradation tests above.
    """

    def _signals_for_flags(self, reliable_flags):
        return [
            _sentiment_signal(
                score=0.4 if reliable else 0.0,
                confidence=0.7 if reliable else 0.0,
                reliable=reliable,
                day=i + 1,
            )
            for i, reliable in enumerate(reliable_flags)
        ]

    @pytest.mark.parametrize(
        "reliable_flags",
        [
            combo
            for n in range(1, 5)
            for combo in itertools.product([True, False], repeat=n)
        ],
    )
    def test_invariants_hold_for_every_reliability_combination(
        self, client, reliable_flags
    ):
        from alphasignal.ingestion import Chunk

        signals = self._signals_for_flags(list(reliable_flags))
        chunks = [
            Chunk(
                chunk_id=f"aapl_inv_{i:04d}",
                ticker="AAPL",
                text=f"chunk {i}",
                token_count=5,
                doc_type="10-K",
                source="SEC EDGAR",
                section="item_7",
                date=date(2024, 1, 1),
                url=None,
                chunk_index=i,
                total_chunks=len(signals),
            )
            for i in range(len(signals))
        ]

        with patch.object(
            client.app.state.app_state.pipeline.metadata_store,
            "get_chunks_by_ticker",
            return_value=chunks,
        ), patch.object(
            client.app.state.app_state.sentiment_extractor,
            "extract_ticker_sentiment",
            return_value=signals,
        ):
            response = client.get("/sentiment/AAPL")

        assert response.status_code == 200
        data = response.json()

        # status values are explicit and finite
        assert data["status"] in {"ok", "no_data", "degraded"}

        # data_available=false implies latest_score is null (not reachable
        # in this test - chunks always exist here - covered by the
        # dedicated no-data test below; asserted again for completeness)
        if not data["data_available"]:
            assert data["latest_score"] is None

        # reliable_chunk_count never exceeds total_chunk_count
        assert data["reliable_chunk_count"] <= data["total_chunk_count"]
        assert data["total_chunk_count"] == len(reliable_flags)
        assert data["reliable_chunk_count"] == sum(reliable_flags)

        # fully degraded output cannot report a positive reliable count,
        # and must never fabricate a score
        if data["reliable_chunk_count"] == 0:
            assert data["status"] == "degraded"
            assert data["degradation_reason"] == "full_extraction_failure"
            assert data["latest_score"] is None
            assert data["degraded"] is True
        elif data["reliable_chunk_count"] < data["total_chunk_count"]:
            assert data["status"] == "degraded"
            assert data["degradation_reason"] == "partial_extraction_failure"
            assert data["degraded"] is True
            assert data["latest_score"] is not None
        else:
            assert data["status"] == "ok"
            assert data["degraded"] is False
            assert data["degradation_reason"] is None
            assert data["latest_score"] is not None

        # degradation_reason is categorical only - never raw exception text
        if data["degradation_reason"] is not None:
            assert data["degradation_reason"] in {
                "partial_extraction_failure",
                "full_extraction_failure",
            }

        # legacy field types/meanings preserved
        assert isinstance(data["ticker"], str)
        assert isinstance(data["latency_ms"], int)
        assert isinstance(data["data_available"], bool)
        assert data["latest_score"] is None or isinstance(data["latest_score"], float)
        for sig in data["signals"]:
            assert isinstance(sig["score"], float)
            assert -1.0 <= sig["score"] <= 1.0
            assert isinstance(sig["confidence"], float)
            assert 0.0 <= sig["confidence"] <= 1.0
            assert isinstance(sig["reliable"], bool)

    def test_no_data_implies_latest_score_null_and_status_no_data(self, client):
        with patch.object(
            client.app.state.app_state.pipeline.metadata_store,
            "get_chunks_by_ticker",
            return_value=[],
        ):
            response = client.get("/sentiment/AAPL")

        data = response.json()
        assert data["data_available"] is False
        assert data["latest_score"] is None
        assert data["status"] == "no_data"

    def test_openapi_schema_generation_succeeds_and_status_is_enum(self, client):
        """Serialization sanity: the app must be able to generate its
        OpenAPI schema with the new Literal-typed fields, and status must
        show up as a real enum, not a bare string."""
        schema = client.app.openapi()
        sentiment_response_schema = schema["components"]["schemas"]["SentimentResponse"]
        status_schema = sentiment_response_schema["properties"]["status"]
        # Pydantic v2 emits enum constraints via allOf/$ref or inline enum
        # depending on version; either way "ok"/"no_data"/"degraded" must
        # appear somewhere in the schema for this field.
        schema_str = json.dumps(status_schema) + json.dumps(schema.get("$defs", {}))
        assert "no_data" in schema_str
        assert "degraded" in schema_str

    def test_json_serialization_represents_null_and_bool_correctly(self, client):
        """Raw JSON text sanity check: null must serialize as JSON `null`,
        not the string "None" or 0."""
        with patch.object(
            client.app.state.app_state.pipeline.metadata_store,
            "get_chunks_by_ticker",
            return_value=[],
        ):
            response = client.get("/sentiment/AAPL")

        assert '"latest_score":null' in response.text.replace(" ", "")
        assert '"data_available":false' in response.text.replace(" ", "")


def test_sentiment_summary_uses_only_reliable_signals_for_avg_and_trend(client):
    """The /summary endpoint's aggregate score/trend must exclude
    unreliable (fallback) signals - mixing them in would silently pull the
    average toward neutral for reasons unrelated to actual sentiment."""
    signals = [
        _sentiment_signal(score=0.8, confidence=0.9, reliable=True, day=3),
        _sentiment_signal(score=0.0, confidence=0.0, reliable=False, day=2),
        _sentiment_signal(score=0.6, confidence=0.85, reliable=True, day=1),
    ]
    from alphasignal.ingestion import Chunk

    chunks = [
        Chunk(
            chunk_id=f"aapl_summary_{i:04d}",
            ticker="AAPL",
            text=f"chunk {i}",
            token_count=5,
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_7",
            date=date(2024, 1, 1),
            url=None,
            chunk_index=i,
            total_chunks=3,
        )
        for i in range(3)
    ]

    with patch.object(
        client.app.state.app_state.pipeline.metadata_store,
        "get_chunks_by_ticker",
        return_value=chunks,
    ), patch.object(
        client.app.state.app_state.sentiment_extractor,
        "extract_ticker_sentiment",
        return_value=signals,
    ):
        response = client.get("/sentiment/AAPL/summary")

    data = response.json()
    # Average must be over [0.8, 0.6] only, not including the 0.0 fallback.
    assert data["avg_score"] == pytest.approx(0.7, abs=1e-6)
    assert data["signal_count"] == 3
    assert data["reliable_signal_count"] == 2
    assert data["degraded"] is True


def test_sentiment_summary_all_unreliable_reports_degraded_no_fabricated_average(
    client,
):
    signals = [
        _sentiment_signal(score=0.0, confidence=0.0, reliable=False, day=1),
        _sentiment_signal(score=0.0, confidence=0.0, reliable=False, day=2),
    ]
    from alphasignal.ingestion import Chunk

    chunks = [
        Chunk(
            chunk_id=f"aapl_summary2_{i:04d}",
            ticker="AAPL",
            text=f"chunk {i}",
            token_count=5,
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_7",
            date=date(2024, 1, 1),
            url=None,
            chunk_index=i,
            total_chunks=2,
        )
        for i in range(2)
    ]

    with patch.object(
        client.app.state.app_state.pipeline.metadata_store,
        "get_chunks_by_ticker",
        return_value=chunks,
    ), patch.object(
        client.app.state.app_state.sentiment_extractor,
        "extract_ticker_sentiment",
        return_value=signals,
    ):
        response = client.get("/sentiment/AAPL/summary")

    data = response.json()
    assert data["avg_score"] is None
    assert data["trend"] == "unknown"
    assert data["degraded"] is True
    assert data["reliable_signal_count"] == 0


def test_sentiment_summary_no_chunks(client):
    """Test that sentiment summary returns defaults when there are no chunks."""
    with patch.object(
        client.app.state.app_state.pipeline.metadata_store,
        "get_chunks_by_ticker",
        return_value=[],
    ):
        response = client.get("/sentiment/AAPL/summary")

    assert response.status_code == 200
    data = response.json()
    assert data["signal_count"] == 0
    assert data["avg_score"] is None
    assert data["trend"] == "unknown"


def test_sentiment_summary_no_signals_extracted(client):
    """Test that sentiment summary handles chunks present but no signals extracted."""
    from alphasignal.ingestion import Chunk

    mock_chunks = [
        Chunk(
            chunk_id="aapl_s_0",
            ticker="AAPL",
            text="chunk",
            token_count=10,
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_1",
            date=date(2024, 1, 1),
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
        response = client.get("/sentiment/AAPL/summary")

    assert response.status_code == 200
    data = response.json()
    assert data["signal_count"] == 0
    assert data["trend"] == "unknown"


def test_sentiment_summary_computes_trend_and_stats(client):
    """Test that sentiment summary computes avg_score, trend, and period_days."""
    from alphasignal.ingestion import Chunk
    from alphasignal.api.schemas import SentimentSignal

    mock_chunks = [
        Chunk(
            chunk_id=f"aapl_sum_{i}",
            ticker="AAPL",
            text=f"chunk {i}",
            token_count=10,
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_1",
            date=date(2024, 1, 1 + i),
            url=None,
            chunk_index=i,
            total_chunks=5,
        )
        for i in range(5)
    ]

    # 3 most recent signals score high (0.8), older 2 score low (0.0) -> improving trend
    signals = [
        SentimentSignal(
            ticker="AAPL",
            date=date(2024, 1, 10 - i),
            score=0.8 if i < 3 else 0.0,
            confidence=0.9,
            source="SEC EDGAR",
            doc_type="10-K",
            key_positive=[],
            key_negative=[],
            summary="s",
        )
        for i in range(5)
    ]

    with patch.object(
        client.app.state.app_state.pipeline.metadata_store,
        "get_chunks_by_ticker",
        return_value=mock_chunks,
    ), patch.object(
        client.app.state.app_state.sentiment_extractor,
        "extract_ticker_sentiment",
        return_value=signals,
    ):
        response = client.get("/sentiment/AAPL/summary")

    assert response.status_code == 200
    data = response.json()
    assert data["signal_count"] == 5
    assert data["trend"] == "improving"
    assert data["period_days"] == 4
    assert data["most_recent_date"] == "2024-01-10"


def test_sentiment_summary_unknown_ticker_404(client):
    """Test that sentiment summary rejects tickers not in the config."""
    response = client.get("/sentiment/ZZZZ/summary")
    assert response.status_code == 404


def test_query_endpoint_handles_openai_error(client):
    """Test that an OpenAIError from generation surfaces as a 503."""
    from openai import OpenAIError

    with patch.object(
        client.app.state.app_state.retriever, "retrieve"
    ) as mock_retrieve, patch.object(
        client.app.state.app_state.reranker, "rerank"
    ) as mock_rerank, patch.object(
        client.app.state.app_state.generator, "generate"
    ) as mock_generate:
        mock_retrieve.return_value = [
            RetrievedChunk(
                chunk_id="c1",
                ticker="AAPL",
                text="text",
                doc_type="10-K",
                source="SEC EDGAR",
                section="item_1",
                date=date(2024, 1, 1),
                url=None,
                dense_score=0.5,
                sparse_score=0.5,
                hybrid_score=0.5,
                final_score=None,
            )
        ]
        mock_rerank.return_value = mock_retrieve.return_value
        mock_generate.side_effect = OpenAIError("rate limited")

        response = client.post(
            "/query/", json={"query": "What is the revenue?", "top_k": 5}
        )

    assert response.status_code == 503


def test_ingest_endpoint_handles_pipeline_failure(client):
    """Test that a pipeline exception is caught and reported as a failed status."""
    with patch.object(
        client.app.state.app_state.pipeline,
        "full_ingest",
        side_effect=RuntimeError("EDGAR is down"),
    ):
        response = client.post("/ingest/AAPL")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "failed"
    assert data["chunks_created"] == 0


def test_ingest_batch_reports_failed_ticker(client):
    """Test that batch ingest marks individual failing tickers as failed."""
    from alphasignal.ingestion.pipeline import IngestResult

    def fake_ingest(ticker, filing_types=None, years_back=None):
        if ticker == "BADTICK":
            raise RuntimeError("boom")
        return IngestResult(
            ticker=ticker, chunks_created=3, chunks_embedded=3, chunks_stored=3
        )

    with patch.object(
        client.app.state.app_state.pipeline, "full_ingest", side_effect=fake_ingest
    ), patch.object(client.app.state.app_state.retriever, "build_bm25_index"):
        response = client.post("/ingest/batch", json={"tickers": ["AAPL", "BADTICK"]})

    assert response.status_code == 200
    data = response.json()
    statuses = {r["ticker"]: r["status"] for r in data["results"]}
    assert statuses["AAPL"] == "completed"
    assert statuses["BADTICK"] == "failed"


def test_metrics_endpoint_reports_recorded_latencies(client):
    """Regression test for the fixed metrics endpoint that never recorded real data.

    After making real query/sentiment/ingest requests, /metrics/ should
    reflect non-zero counts for those categories.
    """
    with patch.object(
        client.app.state.app_state.retriever, "retrieve"
    ) as mock_retrieve, patch.object(
        client.app.state.app_state.reranker, "rerank"
    ) as mock_rerank, patch.object(
        client.app.state.app_state.generator, "generate"
    ) as mock_generate:
        mock_retrieve.return_value = []
        mock_rerank.return_value = []
        mock_generate.return_value = GenerationResult(
            answer="",
            cited_chunks=[],
            prompt_tokens=0,
            completion_tokens=0,
            model="gpt-4o-mini",
        )
        client.post("/query/", json={"query": "What is the revenue?", "top_k": 5})

    response = client.get("/metrics/")

    assert response.status_code == 200
    data = response.json()
    assert data["query"]["count"] == 1
    assert "system" in data
    assert data["system"]["chunks_indexed"] == 0


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


# ---------------------------------------------------------------------------
# API-key auth (ALPHASIGNAL_API_KEY)
# ---------------------------------------------------------------------------


class TestApiKeyAuth:
    """Auth is off when ALPHASIGNAL_API_KEY is unset (all other tests rely on
    that); when set, every route except /health requires X-API-Key."""

    def test_routes_open_when_auth_unset(self, client, monkeypatch):
        monkeypatch.delenv("ALPHASIGNAL_API_KEY", raising=False)
        assert client.get("/metrics/").status_code == 200

    def test_protected_route_401_without_key(self, client, monkeypatch):
        monkeypatch.setenv("ALPHASIGNAL_API_KEY", "secret123")
        assert client.get("/metrics/").status_code == 401
        assert client.get("/sentiment/AAPL").status_code == 401
        assert client.post("/query/", json={"query": "test"}).status_code == 401

    def test_protected_route_401_with_wrong_key(self, client, monkeypatch):
        monkeypatch.setenv("ALPHASIGNAL_API_KEY", "secret123")
        resp = client.get("/metrics/", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_protected_route_ok_with_correct_key(self, client, monkeypatch):
        monkeypatch.setenv("ALPHASIGNAL_API_KEY", "secret123")
        resp = client.get("/metrics/", headers={"X-API-Key": "secret123"})
        assert resp.status_code == 200

    def test_health_stays_open_with_auth_enabled(self, client, monkeypatch):
        """Railway's healthcheck can't send headers - /health must stay open."""
        monkeypatch.setenv("ALPHASIGNAL_API_KEY", "secret123")
        assert client.get("/health/").status_code == 200
