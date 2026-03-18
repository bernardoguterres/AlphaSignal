"""Tests for embeddings and storage."""

import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from alphasignal.embeddings.cache import EmbeddingCache
from alphasignal.embeddings.embedder import Embedder
from alphasignal.ingestion import Chunk
from alphasignal.store.metadata_store import MetadataStore
from alphasignal.store.vector_store import VectorStore


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "embeddings": {
            "model": "text-embedding-ada-002",
            "batch_size": 100,
            "max_retries": 3,
            "retry_delay": 1.0,
        }
    }


def test_vector_store_add_and_search(tmp_path):
    """Test adding vectors and searching."""
    # Create vector store
    store = VectorStore(str(tmp_path / "index"), dim=1536)
    store.load()

    # Create 10 random embeddings
    embeddings = np.random.rand(10, 1536).astype(np.float32)
    chunk_ids = [f"chunk_{i:03d}" for i in range(10)]

    # Add to store
    store.add(embeddings, chunk_ids)

    # Search with first embedding as query
    query = embeddings[0]
    results = store.search(query, k=5)

    # Should find itself as top result
    assert len(results) > 0
    top_chunk_id, top_score = results[0]
    assert top_chunk_id == "chunk_000"
    assert top_score > 0.99  # Should be very similar to itself


def test_vector_store_persists_to_disk(tmp_path):
    """Test that vector store persists and loads from disk."""
    index_path = tmp_path / "index"

    # Create first store and add vectors
    store1 = VectorStore(str(index_path), dim=1536)
    store1.load()

    embeddings = np.random.rand(5, 1536).astype(np.float32)
    chunk_ids = [f"chunk_{i}" for i in range(5)]
    store1.add(embeddings, chunk_ids)

    assert len(store1) == 5

    # Create new store instance and load
    store2 = VectorStore(str(index_path), dim=1536)
    store2.load()

    # Should have loaded the persisted vectors
    assert len(store2) == 5
    assert store2.chunk_ids == chunk_ids


def test_vector_store_normalises_embeddings(tmp_path):
    """Test that vector store normalizes embeddings."""
    store = VectorStore(str(tmp_path / "index"), dim=1536)
    store.load()

    # Create un-normalized embeddings with large magnitudes
    embeddings = np.random.rand(5, 1536).astype(np.float32) * 100  # Large values
    chunk_ids = [f"chunk_{i}" for i in range(5)]

    store.add(embeddings, chunk_ids)

    # Search with un-normalized query
    query = np.random.rand(1536).astype(np.float32) * 50
    results = store.search(query, k=3)

    # Scores should still be in [0, 1] range due to normalization
    for chunk_id, score in results:
        assert 0.0 <= score <= 1.0, f"Score {score} out of range [0, 1]"


def test_metadata_store_add_and_retrieve(tmp_path):
    """Test adding and retrieving chunks from metadata store."""
    db_path = tmp_path / "test.db"
    store = MetadataStore(str(db_path))

    # Create test chunks
    chunks = [
        Chunk(
            chunk_id=f"aapl_10k_test_000{i}",
            ticker="AAPL",
            text=f"Test chunk {i}",
            token_count=100,
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_1",
            date=date(2024, 1, 1),
            url=None,
            chunk_index=i,
            total_chunks=5
        )
        for i in range(5)
    ]

    # Add chunks
    store.add_chunks(chunks)

    # Retrieve by ticker
    retrieved = store.get_chunks_by_ticker("AAPL")
    assert len(retrieved) == 5

    # Retrieve specific chunk
    chunk = store.get_chunk("aapl_10k_test_0000")
    assert chunk is not None
    assert chunk.ticker == "AAPL"
    assert chunk.text == "Test chunk 0"


def test_metadata_store_deduplicates(tmp_path):
    """Test that metadata store handles duplicate chunk_ids correctly."""
    db_path = tmp_path / "test.db"
    store = MetadataStore(str(db_path))

    # Create chunk
    chunk = Chunk(
        chunk_id="aapl_test_0001",
        ticker="AAPL",
        text="Original text",
        token_count=100,
        doc_type="10-K",
        source="SEC EDGAR",
        section="item_1",
        date=date(2024, 1, 1),
        url=None,
        chunk_index=0,
        total_chunks=1
    )

    # Add twice
    store.add_chunks([chunk])
    store.add_chunks([chunk])

    # Should only have one
    assert store.count() == 1


def test_embedding_cache_hit_and_miss(tmp_path):
    """Test embedding cache hits and misses."""
    cache_path = tmp_path / "cache.pkl"
    cache = EmbeddingCache(str(cache_path))

    # Set embedding for chunk_001
    embedding1 = np.random.rand(1536).astype(np.float32)
    cache.set("chunk_001", embedding1)

    # Get many with mix of cached and uncached
    cached, uncached = cache.get_many(["chunk_001", "chunk_002"])

    # Verify results
    assert "chunk_001" in cached
    assert np.array_equal(cached["chunk_001"], embedding1)
    assert uncached == ["chunk_002"]


def test_embedder_uses_cache(test_config, tmp_path):
    """Test that embedder uses cache to avoid redundant API calls."""
    cache_path = tmp_path / "cache.pkl"
    cache = EmbeddingCache(str(cache_path))

    # Pre-populate cache with 5 chunks
    for i in range(5):
        embedding = np.random.rand(1536).astype(np.float32)
        cache.set(f"chunk_{i:03d}", embedding)

    # Mock OpenAI client initialization
    with patch('alphasignal.embeddings.embedder.OpenAI') as MockOpenAI:
        # Create embedder with cache
        embedder = Embedder(test_config, cache)

        # Create 7 chunks (5 cached + 2 new)
        chunks = [
            Chunk(
                chunk_id=f"chunk_{i:03d}",
                ticker="TEST",
                text=f"Text {i}",
                token_count=50,
                doc_type="news",
                source="Test",
                section=None,
                date=date.today(),
                url=f"http://test.com/{i}",
                chunk_index=i,
                total_chunks=7
            )
            for i in range(7)
        ]

        # Mock OpenAI client to track calls
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=np.random.rand(1536).tolist()),
            MagicMock(embedding=np.random.rand(1536).tolist())
        ]

        with patch.object(embedder.client.embeddings, 'create', return_value=mock_response) as mock_create:
            # Embed chunks
            result = embedder.embed_chunks(chunks)

            # Should have embeddings for all 7 chunks
            assert len(result) == 7

            # OpenAI API should only be called once for the 2 new chunks
            assert mock_create.call_count == 1

            # Verify it was called with only the 2 uncached texts
            call_args = mock_create.call_args
            assert len(call_args.kwargs['input']) == 2
