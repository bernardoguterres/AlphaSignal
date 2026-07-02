"""Tests for the ingestion pipeline orchestration."""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from alphasignal.embeddings.cache import EmbeddingCache
from alphasignal.embeddings.embedder import Embedder
from alphasignal.ingestion import Chunk, RawArticle, RawDocument
from alphasignal.ingestion.pipeline import IngestionPipeline, IngestResult
from alphasignal.store.metadata_store import MetadataStore
from alphasignal.store.vector_store import VectorStore


@pytest.fixture
def pipeline_config():
    """Provide test configuration for the pipeline."""
    return {
        "ingestion": {
            "edgar": {
                "filing_types": ["10-K"],
                "years_back": 2,
                "rate_limit_delay": 0.1,
            },
            "news": {"sources": [], "max_articles_per_ticker": 10, "max_age_days": 30},
        },
        "chunking": {
            "target_tokens": 300,
            "min_tokens": 100,
            "max_tokens": 400,
            "overlap_tokens": 50,
        },
        "embeddings": {
            "model": "text-embedding-ada-002",
            "batch_size": 100,
            "max_retries": 3,
            "retry_delay": 1.0,
        },
        "storage": {
            "faiss_index_path": "test_data/faiss_index",
            "sqlite_db_path": "test_data/metadata.db",
            "embeddings_cache_path": "test_data/embeddings_cache",
        },
    }


@pytest.fixture
def raw_doc():
    """Create a sample RawDocument."""
    return RawDocument(
        ticker="AAPL",
        doc_type="10-K",
        filing_date=date(2024, 1, 15),
        period_of_report=date(2024, 1, 15),
        source="SEC EDGAR",
        sections={"item_1": "Apple designs and sells consumer electronics. " * 20},
        file_path="/tmp/filing.htm",
        accession_number="0001234567-24-000001",
    )


@pytest.fixture
def raw_article():
    """Create a sample RawArticle."""
    return RawArticle(
        ticker="AAPL",
        title="Apple posts strong quarter",
        content="Apple reported record revenue this quarter. " * 10,
        published_date=date(2024, 2, 1),
        url="https://example.com/apple-news",
        source="Reuters",
    )


@pytest.fixture
def pipeline(pipeline_config, tmp_path, monkeypatch):
    """Create an IngestionPipeline with shared, tmp-backed stores.

    Mirrors how the API wires the pipeline: shared embedder/vector_store/
    metadata_store instances so ingested data is immediately queryable.
    """
    monkeypatch.chdir(tmp_path)

    vector_store = VectorStore(str(tmp_path / "faiss_index"), dim=1536)
    vector_store.load()
    metadata_store = MetadataStore(str(tmp_path / "metadata.db"))

    with patch("alphasignal.embeddings.embedder.OpenAI"):
        cache = EmbeddingCache(str(tmp_path / "cache.pkl"))
        embedder = Embedder(pipeline_config, cache)

    pipeline = IngestionPipeline(
        pipeline_config,
        embedder=embedder,
        vector_store=vector_store,
        metadata_store=metadata_store,
    )
    return pipeline


def test_pipeline_init_creates_data_dir(pipeline_config, tmp_path, monkeypatch):
    """Test that pipeline initialization creates the data directory."""
    monkeypatch.chdir(tmp_path)

    with patch("alphasignal.embeddings.embedder.OpenAI"):
        p = IngestionPipeline(pipeline_config)

    assert (tmp_path / "data").exists()
    # Standalone construction builds its own stores rather than sharing None
    assert p.embedder is not None
    assert p.vector_store is not None
    assert p.metadata_store is not None


def test_pipeline_init_uses_shared_instances(pipeline_config, tmp_path, monkeypatch):
    """Test that shared embedder/vector_store/metadata_store are reused, not rebuilt."""
    monkeypatch.chdir(tmp_path)

    vector_store = VectorStore(str(tmp_path / "shared_index"), dim=1536)
    vector_store.load()
    metadata_store = MetadataStore(str(tmp_path / "shared.db"))

    with patch("alphasignal.embeddings.embedder.OpenAI"):
        cache = EmbeddingCache(str(tmp_path / "shared_cache.pkl"))
        embedder = Embedder(pipeline_config, cache)

    p = IngestionPipeline(
        pipeline_config,
        embedder=embedder,
        vector_store=vector_store,
        metadata_store=metadata_store,
    )

    # Pipeline should hold the exact same objects we passed in, not copies
    assert p.embedder is embedder
    assert p.vector_store is vector_store
    assert p.metadata_store is metadata_store


def test_ingest_ticker_edgar_uses_config_defaults(pipeline, raw_doc):
    """Test that ingest_ticker_edgar falls back to config values when args are None."""
    with patch.object(
        pipeline.edgar_ingester, "fetch_filings", return_value=[raw_doc]
    ) as mock_fetch:
        docs = pipeline.ingest_ticker_edgar("AAPL")

    assert docs == [raw_doc]
    mock_fetch.assert_called_once_with(
        ticker="AAPL", filing_types=["10-K"], years_back=2
    )


def test_ingest_ticker_edgar_uses_override_args(pipeline, raw_doc):
    """Test that explicit filing_types/years_back override config defaults."""
    with patch.object(
        pipeline.edgar_ingester, "fetch_filings", return_value=[raw_doc]
    ) as mock_fetch:
        pipeline.ingest_ticker_edgar("AAPL", filing_types=["10-Q"], years_back=5)

    mock_fetch.assert_called_once_with(
        ticker="AAPL", filing_types=["10-Q"], years_back=5
    )


def test_ingest_ticker_news(pipeline, raw_article):
    """Test that ingest_ticker_news delegates to the news ingester."""
    with patch.object(
        pipeline.news_ingester, "fetch_articles", return_value=[raw_article]
    ) as mock_fetch:
        articles = pipeline.ingest_ticker_news("AAPL")

    assert articles == [raw_article]
    mock_fetch.assert_called_once_with("AAPL")


def test_ingest_ticker_combines_filings_and_news(pipeline, raw_doc, raw_article):
    """Test that ingest_ticker returns both filings and articles."""
    with patch.object(
        pipeline.edgar_ingester, "fetch_filings", return_value=[raw_doc]
    ), patch.object(
        pipeline.news_ingester, "fetch_articles", return_value=[raw_article]
    ):
        docs, articles = pipeline.ingest_ticker("AAPL")

    assert docs == [raw_doc]
    assert articles == [raw_article]


def test_chunk_documents_aggregates_across_docs(pipeline, raw_doc):
    """Test that chunk_documents flattens chunks from multiple documents."""
    second_doc = RawDocument(
        ticker="AAPL",
        doc_type="10-Q",
        filing_date=date(2024, 4, 1),
        period_of_report=date(2024, 4, 1),
        source="SEC EDGAR",
        sections={"item_2": "Management discussion of quarterly results. " * 20},
        file_path="/tmp/filing2.htm",
        accession_number="0001234567-24-000002",
    )

    chunks = pipeline.chunk_documents([raw_doc, second_doc])

    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    doc_types = {c.doc_type for c in chunks}
    assert doc_types == {"10-K", "10-Q"}


def test_chunk_articles_aggregates(pipeline, raw_article):
    """Test that chunk_articles produces chunks tagged as news."""
    chunks = pipeline.chunk_articles([raw_article])

    assert len(chunks) > 0
    assert all(c.doc_type == "news" for c in chunks)


def test_ingest_and_chunk_ticker_combines_docs_and_articles(
    pipeline, raw_doc, raw_article
):
    """Test that ingest_and_chunk_ticker produces chunks from both sources."""
    with patch.object(
        pipeline.edgar_ingester, "fetch_filings", return_value=[raw_doc]
    ), patch.object(
        pipeline.news_ingester, "fetch_articles", return_value=[raw_article]
    ):
        chunks = pipeline.ingest_and_chunk_ticker("AAPL")

    doc_types = {c.doc_type for c in chunks}
    assert "10-K" in doc_types
    assert "news" in doc_types


def test_store_chunks_empty_list_is_noop(pipeline):
    """Test that store_chunks does nothing when given an empty chunk list."""
    # Should not raise and should not touch the stores
    pipeline.store_chunks([], {})
    assert pipeline.metadata_store.count() == 0
    assert len(pipeline.vector_store) == 0


def test_store_chunks_makes_data_immediately_queryable(pipeline):
    """Regression test for the fixed stale-vector-store bug.

    Stored chunks must be immediately visible via the *same* metadata_store
    and vector_store instances the pipeline was constructed with (this is
    what makes /query see freshly ingested data without a restart).
    """
    chunks = [
        Chunk(
            chunk_id=f"aapl_10k_test_{i:04d}",
            ticker="AAPL",
            text=f"Chunk text number {i}",
            token_count=10,
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_1",
            date=date(2024, 1, 1),
            url=None,
            chunk_index=i,
            total_chunks=3,
        )
        for i in range(3)
    ]
    embeddings = {c.chunk_id: np.random.rand(1536).astype(np.float32) for c in chunks}

    pipeline.store_chunks(chunks, embeddings)

    # Immediately queryable via the shared metadata store
    assert pipeline.metadata_store.count() == 3
    stored = pipeline.metadata_store.get_chunk("aapl_10k_test_0000")
    assert stored is not None
    assert stored.text == "Chunk text number 0"

    # Immediately queryable via the shared vector store (no reload needed)
    assert len(pipeline.vector_store) == 3
    results = pipeline.vector_store.search(embeddings["aapl_10k_test_0000"], k=1)
    assert results[0][0] == "aapl_10k_test_0000"


def test_full_ingest_runs_end_to_end_and_returns_counts(pipeline, raw_doc):
    """Test that full_ingest wires ingest -> chunk -> embed -> store correctly."""

    def fake_embed_chunks(chunks):
        return {c.chunk_id: np.random.rand(1536).astype(np.float32) for c in chunks}

    with patch.object(
        pipeline.edgar_ingester, "fetch_filings", return_value=[raw_doc]
    ), patch.object(
        pipeline.news_ingester, "fetch_articles", return_value=[]
    ), patch.object(
        pipeline.embedder, "embed_chunks", side_effect=fake_embed_chunks
    ):

        result = pipeline.full_ingest("AAPL")

    assert isinstance(result, IngestResult)
    assert result.ticker == "AAPL"
    assert result.chunks_created > 0
    assert result.chunks_embedded == result.chunks_created
    assert result.chunks_stored == result.chunks_created

    # Data should be immediately queryable after full_ingest completes
    assert pipeline.metadata_store.count() == result.chunks_created
    assert len(pipeline.vector_store) == result.chunks_created
