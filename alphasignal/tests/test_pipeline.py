"""Tests for the ingestion pipeline orchestration."""

from dataclasses import replace
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


def test_ingestion_writers_preserve_metadata_and_vectors(pipeline, raw_doc):
    from copy import copy

    second = copy(pipeline)
    path = pipeline.vector_store.index_path
    second.vector_store = VectorStore(str(path))
    second.vector_store.load()
    second.metadata_store = MetadataStore(str(pipeline.metadata_store.db_path))

    first_chunks = pipeline.chunk_documents([raw_doc])
    assert first_chunks
    import dataclasses

    second_chunks = [
        dataclasses.replace(
            chunk, chunk_id=f"second_{chunk.chunk_id}", source_id="second"
        )
        for chunk in first_chunks
    ]
    for owner, chunks in ((pipeline, first_chunks), (second, second_chunks)):
        owner.store_chunks(
            chunks, {c.chunk_id: np.ones(1536, dtype=np.float32) for c in chunks}
        )

    reloaded = VectorStore(str(path))
    reloaded.load()
    expected = {chunk.chunk_id for chunk in first_chunks + second_chunks}
    assert set(reloaded.chunk_ids) == expected
    assert set(reloaded.content_hashes) == expected
    assert all(second.metadata_store.get_chunk(chunk_id) for chunk_id in expected)


def test_failed_vector_publish_reports_error_and_retries_in_same_worker(
    pipeline, raw_doc
):
    chunks = pipeline.chunk_documents([raw_doc])
    assert chunks
    embeddings = {chunk.chunk_id: np.ones(1536, dtype=np.float32) for chunk in chunks}
    with patch(
        "alphasignal.store.vector_store.faiss.write_index",
        side_effect=OSError("disk full"),
    ):
        with pytest.raises(RuntimeError, match="publish"):
            pipeline.store_chunks(chunks, embeddings)
    assert pipeline.metadata_store.count() == len(chunks)
    assert not pipeline.vector_store.chunk_ids
    pipeline.store_chunks(chunks, embeddings)
    reloaded = VectorStore(str(pipeline.vector_store.index_path))
    reloaded.load()
    assert set(reloaded.chunk_ids) == set(embeddings)


def test_failed_orphan_cleanup_keeps_metadata_for_retry(pipeline, raw_doc):
    chunks = pipeline.chunk_documents([raw_doc])
    assert chunks and chunks[0].source_id
    embeddings = {chunk.chunk_id: np.ones(1536, dtype=np.float32) for chunk in chunks}
    pipeline.store_chunks(chunks, embeddings)
    sources = {chunks[0].source_id}

    with patch(
        "alphasignal.store.vector_store.faiss.write_index",
        side_effect=OSError("disk full"),
    ):
        with pytest.raises(RuntimeError, match="publish"):
            pipeline.store_chunks([], {}, source_prefixes=sources)
    assert pipeline.metadata_store.count() == len(chunks)
    assert set(pipeline.vector_store.chunk_ids) == set(embeddings)

    pipeline.store_chunks([], {}, source_prefixes=sources)
    reloaded = VectorStore(str(pipeline.vector_store.index_path))
    reloaded.load()
    assert pipeline.metadata_store.count() == 0
    assert not reloaded.chunk_ids


def test_publication_exception_does_not_poison_retry(pipeline, raw_doc):
    chunks = pipeline.chunk_documents([raw_doc])
    assert chunks
    embeddings = {chunk.chunk_id: np.ones(1536, dtype=np.float32) for chunk in chunks}
    with patch.object(
        pipeline.vector_store, "save", side_effect=OSError("manifest unavailable")
    ):
        with pytest.raises(OSError, match="manifest unavailable"):
            pipeline.store_chunks(chunks, embeddings)
    assert not pipeline.vector_store.chunk_ids
    pipeline.store_chunks(chunks, embeddings)
    assert set(pipeline.vector_store.chunk_ids) == set(embeddings)


def test_storage_startup_replays_interrupted_ingestion(pipeline, raw_doc, tmp_path):
    from alphasignal.scripts._common import build_storage_components

    chunks = pipeline.chunk_documents([raw_doc])
    embeddings = {chunk.chunk_id: np.ones(1536, dtype=np.float32) for chunk in chunks}
    with patch(
        "alphasignal.store.vector_store.faiss.write_index",
        side_effect=OSError("disk full"),
    ):
        with pytest.raises(RuntimeError, match="publish"):
            pipeline.store_chunks(chunks, embeddings)

    config = {
        **pipeline.config,
        "storage": {
            "faiss_index_path": str(pipeline.vector_store.index_path),
            "sqlite_db_path": str(pipeline.metadata_store.db_path),
            "embeddings_cache_path": str(tmp_path / "startup_cache"),
        },
    }
    with patch("alphasignal.embeddings.embedder.OpenAI") as provider:
        vectors, metadata, _ = build_storage_components(config, recover_pending=True)
    assert not provider.return_value.embeddings.create.called
    assert set(vectors.chunk_ids) == set(embeddings)
    assert vectors.content_hashes == metadata.get_content_hashes()


def test_storage_startup_ignores_batch_without_matching_commit(pipeline, raw_doc):
    from alphasignal.ingestion.recovery import pending_batch_path, recover_vector_batch

    chunks = pipeline.chunk_documents([raw_doc])
    embeddings = {chunk.chunk_id: np.ones(1536, dtype=np.float32) for chunk in chunks}
    with patch.object(
        pipeline.metadata_store, "add_chunks", side_effect=OSError("SQLite unavailable")
    ):
        with pytest.raises(OSError, match="SQLite unavailable"):
            pipeline.store_chunks(chunks, embeddings)

    assert pending_batch_path(pipeline.vector_store).exists()
    reloaded = VectorStore(str(pipeline.vector_store.index_path))
    reloaded.load()
    assert recover_vector_batch(reloaded, pipeline.metadata_store) == 0
    assert not reloaded.chunk_ids
    assert not pending_batch_path(reloaded).exists()


@pytest.mark.parametrize(
    ("interrupt_before", "replace_version"),
    [("vectors", False), ("metadata", False), ("metadata", True)],
)
def test_storage_startup_finishes_interrupted_empty_source_cleanup(
    request, tmp_path, interrupt_before, replace_version
):
    from alphasignal.retrieval.retriever import HybridRetriever
    from alphasignal.scripts._common import build_storage_components

    with patch("alphasignal.ingestion.pipeline.SemanticChunker"):
        pipeline = request.getfixturevalue("pipeline")
    chunks = [
        Chunk(
            chunk_id="synthetic_g1_0000",
            ticker="AAPL",
            text="consumer electronics",
            token_count=2,
            doc_type="10-K",
            source="synthetic",
            section="item_1",
            date=date(2024, 1, 15),
            url=None,
            chunk_index=0,
            total_chunks=1,
            source_id="synthetic",
        )
    ]
    embeddings = {chunk.chunk_id: np.ones(1536, dtype=np.float32) for chunk in chunks}
    pipeline.store_chunks(chunks, embeddings)
    # Sparse-only metadata can be legitimate and must survive retirement
    # of another source. Recovery only applies the journaled operation.
    unrelated = Chunk(
        chunk_id="unrelated_0000",
        ticker="MSFT",
        text="unrelated research",
        token_count=2,
        doc_type="10-K",
        source="unrelated",
        section="item_1",
        date=date(2024, 1, 15),
        url=None,
        chunk_index=0,
        total_chunks=1,
        source_id="unrelated",
    )
    pipeline.metadata_store.add_chunks([unrelated])

    # An authoritative re-chunk removes the whole source. The process stops
    # immediately before or after vector removal publishes.
    class ProcessStopped(BaseException):
        pass

    if interrupt_before == "vectors":
        store, method = pipeline.vector_store, "remove"
    else:
        store, method = pipeline.metadata_store, "delete_chunks"
    with patch.object(store, method, side_effect=ProcessStopped):
        with pytest.raises(ProcessStopped):
            pipeline.store_chunks([], {}, source_prefixes={chunks[0].source_id})
    assert bool(pipeline.vector_store.chunk_ids) == (interrupt_before == "vectors")
    assert pipeline.metadata_store.count() == 2

    expected_metadata = {unrelated.chunk_id: unrelated.content_hash}
    expected_vectors = {}
    if replace_version:
        # Simulate a store restored with newer content for the same ID while
        # the old cleanup journal remains in the index directory.
        replacement = replace(chunks[0], text="renewable energy", content_hash="")
        pipeline.metadata_store.add_chunks([replacement])
        expected_vectors = {replacement.chunk_id: replacement.content_hash}
        assert pipeline.vector_store.add(
            np.ones((1, 1536), dtype=np.float32),
            [replacement.chunk_id],
            expected_vectors,
        )
        expected_metadata.update(expected_vectors)

    config = {
        **pipeline.config,
        "storage": {
            "faiss_index_path": str(pipeline.vector_store.index_path),
            "sqlite_db_path": str(pipeline.metadata_store.db_path),
            "embeddings_cache_path": str(tmp_path / "startup_cache"),
        },
    }
    with patch("alphasignal.embeddings.embedder.OpenAI") as provider:
        vectors, metadata, embedder = build_storage_components(
            config, recover_pending=True
        )
    assert not provider.return_value.embeddings.create.called
    if not replace_version:
        with patch.object(
            embedder, "embed_texts", return_value=[np.ones(1536, dtype=np.float32)]
        ):
            hits = HybridRetriever(config, embedder, vectors, metadata).retrieve(
                "consumer electronics", ticker="AAPL"
            )
        assert not hits
    assert set(vectors.chunk_ids) == set(expected_vectors)
    assert vectors.content_hashes == expected_vectors
    assert metadata.get_content_hashes() == expected_metadata

    with patch("alphasignal.embeddings.embedder.OpenAI"):
        recovered_again, metadata_again, _ = build_storage_components(
            config, recover_pending=True
        )
    assert recovered_again.generation == vectors.generation
    assert metadata_again.get_content_hashes() == expected_metadata


@pytest.mark.parametrize(
    "boundary",
    (
        "before_commit_unchanged_hash",
        "after_commit",
        "after_publish",
        "after_removal",
        "after_delete",
        "restored_changed_replacement",
    ),
)
def test_storage_startup_cleans_replaced_chunks_if_interrupted_before_cleanup(
    request, tmp_path, boundary
):
    from alphasignal.ingestion.recovery import pending_batch_path
    from alphasignal.retrieval.retriever import HybridRetriever
    from alphasignal.scripts._common import build_storage_components

    with patch("alphasignal.ingestion.pipeline.SemanticChunker"):
        pipeline = request.getfixturevalue("pipeline")
    old = Chunk(
        chunk_id="synthetic_g1_0000",
        ticker="AAPL",
        text="consumer electronics",
        token_count=2,
        doc_type="10-K",
        source="synthetic",
        section="item_1",
        date=date(2024, 1, 15),
        url=None,
        chunk_index=0,
        total_chunks=1,
        source_id="synthetic",
    )
    new = replace(
        old, chunk_id="synthetic_g2_0000", text="renewable energy", content_hash=""
    )
    vec = np.ones(1536, dtype=np.float32)
    if boundary == "before_commit_unchanged_hash":
        pipeline.store_chunks([old, new], {old.chunk_id: vec, new.chunk_id: vec})
        new = replace(new, attribution="corrected attribution")
    else:
        pipeline.store_chunks([old], {old.chunk_id: vec})

    class ProcessStopped(BaseException):
        pass

    # Interrupt storage at each durable boundary of a replacement.
    original_add = pipeline.vector_store.add
    original_delete = pipeline.metadata_store.delete_chunks

    def publish_then_stop(*args, **kwargs):
        assert original_add(*args, **kwargs)
        raise ProcessStopped

    def delete_then_stop(*args, **kwargs):
        original_delete(*args, **kwargs)
        raise ProcessStopped

    if boundary == "before_commit_unchanged_hash":
        store, method, failure = pipeline.metadata_store, "add_chunks", ProcessStopped
    elif boundary in ("after_commit", "restored_changed_replacement"):
        store, method, failure = pipeline.vector_store, "add", ProcessStopped
    elif boundary == "after_publish":
        store, method, failure = pipeline.vector_store, "add", publish_then_stop
    else:
        store, method, failure = (
            pipeline.metadata_store,
            "delete_chunks",
            delete_then_stop if boundary == "after_delete" else ProcessStopped,
        )
    with patch.object(store, method, side_effect=failure):
        with pytest.raises(ProcessStopped):
            pipeline.store_chunks([new], {new.chunk_id: vec})
    if boundary == "restored_changed_replacement":
        # Restoration can change a replacement row even if its receipt
        # from the original transaction is still present.
        new = replace(new, text="restored newer text", content_hash="")
        pipeline.metadata_store.add_chunks([new])

    config = {
        **pipeline.config,
        "storage": {
            "faiss_index_path": str(pipeline.vector_store.index_path),
            "sqlite_db_path": str(pipeline.metadata_store.db_path),
            "embeddings_cache_path": str(tmp_path / "startup_cache"),
        },
    }
    with patch("alphasignal.embeddings.embedder.OpenAI") as provider:
        vectors, metadata, embedder = build_storage_components(
            config, recover_pending=True
        )
    assert not provider.return_value.embeddings.create.called
    with patch.object(embedder, "embed_texts", return_value=[vec]):
        hits = HybridRetriever(config, embedder, vectors, metadata).retrieve(
            "consumer electronics", ticker="AAPL"
        )
    if boundary == "before_commit_unchanged_hash":
        assert old.chunk_id in {hit.chunk_id for hit in hits}
        assert set(vectors.chunk_ids) == {old.chunk_id, new.chunk_id}
        assert metadata.get_chunk(new.chunk_id).attribution == ""
    elif boundary == "restored_changed_replacement":
        assert old.chunk_id in {hit.chunk_id for hit in hits}
        assert set(vectors.chunk_ids) == {old.chunk_id}
        assert metadata.get_content_hashes() == {
            old.chunk_id: old.content_hash,
            new.chunk_id: new.content_hash,
        }
    else:
        assert old.chunk_id not in {hit.chunk_id for hit in hits}
        assert set(vectors.chunk_ids) == {new.chunk_id}
        assert metadata.get_content_hashes() == {new.chunk_id: new.content_hash}
    assert not pending_batch_path(vectors).exists()
