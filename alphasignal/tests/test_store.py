"""Tests for embeddings and storage."""

import json
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


def test_vector_store_add_dedupes_against_existing_chunk_ids(tmp_path):
    """Regression test for the audit's confirmed FAISS/SQLite drift bug:
    add() previously had no dedup/upsert at all, so re-ingesting the same
    chunk (a retry, or a re-run of the same ticker) duplicated it in FAISS
    while MetadataStore.add_chunks() correctly deduped via session.merge() -
    len(vector_store) and metadata_store.count() would permanently drift
    apart. Re-adding an already-present chunk_id must now be a no-op on
    the vector side, matching the metadata side's upsert behavior."""
    store = VectorStore(str(tmp_path / "index"), dim=1536)
    store.load()

    embeddings = np.random.rand(5, 1536).astype(np.float32)
    chunk_ids = [f"chunk_{i:03d}" for i in range(5)]

    store.add(embeddings, chunk_ids)
    assert len(store) == 5

    # Re-ingest the exact same chunks (simulates a retry / re-run) - must
    # NOT duplicate them.
    store.add(embeddings, chunk_ids)
    assert len(store) == 5
    assert len(store.chunk_ids) == 5

    # A mix of already-present and genuinely new chunk_ids: only the new
    # ones should be added.
    more_embeddings = np.random.rand(3, 1536).astype(np.float32)
    mixed_chunk_ids = ["chunk_000", "chunk_new_001", "chunk_new_002"]
    store.add(more_embeddings, mixed_chunk_ids)

    assert len(store) == 7  # 5 original + 2 genuinely new
    assert store.chunk_ids.count("chunk_000") == 1  # not duplicated
    assert "chunk_new_001" in store.chunk_ids
    assert "chunk_new_002" in store.chunk_ids


def test_vector_store_add_all_duplicates_is_noop(tmp_path):
    """Adding a batch that's entirely already-present chunk_ids must not
    touch the index or trigger a save at all."""
    store = VectorStore(str(tmp_path / "index"), dim=1536)
    store.load()

    embeddings = np.random.rand(3, 1536).astype(np.float32)
    chunk_ids = ["a", "b", "c"]
    store.add(embeddings, chunk_ids)
    assert len(store) == 3

    with patch.object(store, "save") as mock_save:
        store.add(embeddings, chunk_ids)  # all duplicates
        mock_save.assert_not_called()

    assert len(store) == 3


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


def test_vector_store_save_does_not_clobber_existing_files_on_mid_write_failure(
    tmp_path,
):
    """Audit bug: save() wrote index.faiss and chunk_ids.json as two separate
    non-atomic steps, so a crash between them left the on-disk pair
    inconsistent. save() must write to temp files and only replace the real
    files once both writes have succeeded - if the second write fails, the
    original on-disk pair must be untouched."""
    index_path = tmp_path / "index"

    store = VectorStore(str(index_path), dim=1536)
    store.load()

    embeddings = np.random.rand(3, 1536).astype(np.float32)
    store.add(embeddings, ["a", "b", "c"])

    index_file = index_path / "index.faiss"
    ids_file = index_path / "chunk_ids.json"
    original_index_bytes = index_file.read_bytes()
    original_ids_bytes = ids_file.read_bytes()

    # Simulate the chunk_ids write failing after the FAISS index write
    # would otherwise have already landed.
    store.chunk_ids.append("d")
    with patch("builtins.open", side_effect=OSError("disk full")):
        store.save()

    # Original files must be untouched - no partial/inconsistent write.
    assert index_file.read_bytes() == original_index_bytes
    assert ids_file.read_bytes() == original_ids_bytes


def test_vector_store_load_recovers_from_corrupted_index(tmp_path):
    """Test that load() falls back to a fresh index if the persisted files are corrupt."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    # Write garbage where the FAISS index and chunk_ids files are expected
    (index_dir / "index.faiss").write_text("not a valid faiss index")
    (index_dir / "chunk_ids.json").write_text("also not valid json")

    store = VectorStore(str(index_dir), dim=1536)
    store.load()

    # Should have recovered with an empty, usable index rather than raising
    assert len(store) == 0
    assert store.chunk_ids == []

    # And it should still be usable afterwards
    embeddings = np.random.rand(2, 1536).astype(np.float32)
    store.add(embeddings, ["a", "b"])
    assert len(store) == 2


def test_vector_store_save_without_index_warns_and_noops(tmp_path):
    """Test that save() is a safe no-op when the index hasn't been created yet."""
    store = VectorStore(str(tmp_path / "index"), dim=1536)
    # Deliberately skip load()
    store.save()  # should not raise
    assert not (tmp_path / "index" / "index.faiss").exists()


def test_vector_store_add_raises_if_not_loaded(tmp_path):
    """Test that add() raises RuntimeError if load() was never called."""
    store = VectorStore(str(tmp_path / "index"), dim=1536)

    with pytest.raises(RuntimeError):
        store.add(np.random.rand(1, 1536).astype(np.float32), ["a"])


def test_vector_store_add_empty_embeddings_is_noop(tmp_path):
    """Test that add() with zero embeddings does nothing."""
    store = VectorStore(str(tmp_path / "index"), dim=1536)
    store.load()

    store.add(np.zeros((0, 1536), dtype=np.float32), [])

    assert len(store) == 0


def test_vector_store_add_mismatched_lengths_raises(tmp_path):
    """Test that mismatched embeddings/chunk_ids lengths raise ValueError."""
    store = VectorStore(str(tmp_path / "index"), dim=1536)
    store.load()

    embeddings = np.random.rand(2, 1536).astype(np.float32)
    with pytest.raises(ValueError):
        store.add(embeddings, ["only_one_id"])


def test_vector_store_search_raises_if_not_loaded(tmp_path):
    """Test that search() raises RuntimeError if load() was never called."""
    store = VectorStore(str(tmp_path / "index"), dim=1536)

    with pytest.raises(RuntimeError):
        store.search(np.random.rand(1536).astype(np.float32), k=5)


def test_vector_store_search_empty_index_returns_empty(tmp_path):
    """Test that searching an empty (but loaded) index returns no results."""
    store = VectorStore(str(tmp_path / "index"), dim=1536)
    store.load()

    results = store.search(np.random.rand(1536).astype(np.float32), k=5)

    assert results == []


def test_vector_store_len_returns_zero_before_load(tmp_path):
    """Test that len() is 0 before load() is called."""
    store = VectorStore(str(tmp_path / "index"), dim=1536)
    assert len(store) == 0


def test_metadata_store_add_chunks_empty_list_is_noop(tmp_path):
    """Test that add_chunks does nothing (and doesn't error) for an empty list."""
    store = MetadataStore(str(tmp_path / "test.db"))
    store.add_chunks([])
    assert store.count() == 0


def test_metadata_store_get_chunk_returns_none_when_missing(tmp_path):
    """Test that get_chunk returns None for an unknown chunk_id."""
    store = MetadataStore(str(tmp_path / "test.db"))
    assert store.get_chunk("does_not_exist") is None


def test_metadata_store_get_chunks_by_ticker_filters_by_doc_type(tmp_path):
    """Test that get_chunks_by_ticker respects the optional doc_type filter."""
    store = MetadataStore(str(tmp_path / "test.db"))

    chunks = [
        Chunk(
            chunk_id="aapl_10k_0",
            ticker="AAPL",
            text="10-K chunk",
            token_count=10,
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_1",
            date=date(2024, 1, 1),
            url=None,
            chunk_index=0,
            total_chunks=1,
        ),
        Chunk(
            chunk_id="aapl_news_0",
            ticker="AAPL",
            text="News chunk",
            token_count=10,
            doc_type="news",
            source="Reuters",
            section=None,
            date=date(2024, 1, 2),
            url="https://example.com",
            chunk_index=0,
            total_chunks=1,
        ),
    ]
    store.add_chunks(chunks)

    only_10k = store.get_chunks_by_ticker("AAPL", doc_type="10-K")
    assert len(only_10k) == 1
    assert only_10k[0].doc_type == "10-K"

    all_chunks = store.get_chunks_by_ticker("AAPL")
    assert len(all_chunks) == 2


def test_metadata_store_get_chunks_by_date_range_filters_by_ticker(tmp_path):
    """Test that get_chunks_by_date_range respects the optional ticker filter."""
    store = MetadataStore(str(tmp_path / "test.db"))

    chunks = [
        Chunk(
            chunk_id="aapl_dr_0",
            ticker="AAPL",
            text="AAPL chunk",
            token_count=10,
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_1",
            date=date(2024, 3, 1),
            url=None,
            chunk_index=0,
            total_chunks=1,
        ),
        Chunk(
            chunk_id="msft_dr_0",
            ticker="MSFT",
            text="MSFT chunk",
            token_count=10,
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_1",
            date=date(2024, 3, 2),
            url=None,
            chunk_index=0,
            total_chunks=1,
        ),
    ]
    store.add_chunks(chunks)

    aapl_only = store.get_chunks_by_date_range(
        start=date(2024, 1, 1), end=date(2024, 12, 31), ticker="AAPL"
    )
    assert len(aapl_only) == 1
    assert aapl_only[0].ticker == "AAPL"

    both = store.get_chunks_by_date_range(
        start=date(2024, 1, 1), end=date(2024, 12, 31)
    )
    assert len(both) == 2


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
            total_chunks=5,
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
        total_chunks=1,
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
    with patch("alphasignal.embeddings.embedder.OpenAI") as MockOpenAI:
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
                total_chunks=7,
            )
            for i in range(7)
        ]

        # Mock OpenAI client to track calls
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=np.random.rand(1536).tolist()),
            MagicMock(embedding=np.random.rand(1536).tolist()),
        ]

        with patch.object(
            embedder.client.embeddings, "create", return_value=mock_response
        ) as mock_create:
            # Embed chunks
            result = embedder.embed_chunks(chunks)

            # Should have embeddings for all 7 chunks
            assert len(result) == 7

            # OpenAI API should only be called once for the 2 new chunks
            assert mock_create.call_count == 1

            # Verify it was called with only the 2 uncached texts
            call_args = mock_create.call_args
            assert len(call_args.kwargs["input"]) == 2


def test_embedding_cache_persists_across_instances(tmp_path):
    """Test that a saved cache can be reloaded by a fresh EmbeddingCache instance."""
    cache_path = tmp_path / "persist_cache.pkl"

    cache1 = EmbeddingCache(str(cache_path))
    embedding = np.random.rand(1536).astype(np.float32)
    cache1.set("chunk_persist", embedding)
    cache1.save()

    # New instance should load the persisted cache from disk
    cache2 = EmbeddingCache(str(cache_path))
    assert len(cache2) == 1
    assert np.array_equal(cache2.get("chunk_persist"), embedding)


def test_embedding_cache_get_returns_none_for_missing_key(tmp_path):
    """Test that get() returns None (not KeyError) for an uncached chunk_id."""
    cache = EmbeddingCache(str(tmp_path / "cache.pkl"))
    assert cache.get("nonexistent") is None


def test_embedding_cache_handles_corrupted_file_gracefully(tmp_path):
    """Test that a corrupted cache file falls back to an empty cache instead of raising."""
    cache_path = tmp_path / "corrupt_cache.npy"
    # Write garbage directly to the derived vector/id paths the cache
    # actually reads from (cache_path itself is never read).
    cache_path.with_suffix(".npy").write_bytes(b"not a valid npy stream")
    cache_path.with_suffix(".json").write_text("not valid json")

    cache = EmbeddingCache(str(cache_path))

    assert len(cache) == 0


def test_embedding_cache_rejects_pickled_payload(tmp_path):
    """A .npy file must be loaded with allow_pickle=False - a pickled object
    array (the old on-disk format) must not silently deserialize."""
    import pickle

    cache_path = tmp_path / "legacy_cache.npy"
    # Simulate a stale pre-migration pickle file sitting at the new .npy path.
    with open(cache_path.with_suffix(".npy"), "wb") as f:
        pickle.dump({"chunk_001": np.random.rand(4).astype(np.float32)}, f)
    cache_path.with_suffix(".json").write_text(json.dumps(["chunk_001"]))

    cache = EmbeddingCache(str(cache_path))

    # Must fail safe (empty cache), not execute/deserialize the pickle payload.
    assert len(cache) == 0


def test_embedder_embed_texts_empty_list_returns_empty_array(test_config):
    """Test that embed_texts([]) short-circuits without calling the API."""
    with patch("alphasignal.embeddings.embedder.OpenAI"):
        cache = MagicMock()
        embedder = Embedder(test_config, cache)

        result = embedder.embed_texts([])

    assert len(result) == 0


def test_embedder_embed_chunks_empty_list_returns_empty_dict(test_config):
    """Test that embed_chunks([]) returns {} without touching the cache."""
    with patch("alphasignal.embeddings.embedder.OpenAI"):
        cache = MagicMock()
        embedder = Embedder(test_config, cache)

        result = embedder.embed_chunks([])

    assert result == {}
    cache.get_many.assert_not_called()


def test_embedder_retries_then_succeeds_after_transient_error(test_config, tmp_path):
    """Test that embed_texts retries on failure and succeeds on a later attempt."""
    cache = EmbeddingCache(str(tmp_path / "retry_cache.pkl"))

    with patch("alphasignal.embeddings.embedder.OpenAI"):
        embedder = Embedder(test_config, cache)
        embedder.retry_delay = 0.01  # keep the test fast

        success_response = MagicMock()
        success_response.data = [MagicMock(embedding=np.random.rand(1536).tolist())]

        with patch.object(
            embedder.client.embeddings,
            "create",
            side_effect=[Exception("transient error"), success_response],
        ) as mock_create, patch("alphasignal.embeddings.embedder.time.sleep"):
            result = embedder.embed_texts(["hello world"])

    assert mock_create.call_count == 2
    assert result.shape == (1, 1536)


def test_embedder_raises_after_exhausting_retries(test_config, tmp_path):
    """Test that embed_texts raises once max_retries is exhausted."""
    cache = EmbeddingCache(str(tmp_path / "fail_cache.pkl"))

    with patch("alphasignal.embeddings.embedder.OpenAI"):
        embedder = Embedder(test_config, cache)
        embedder.max_retries = 2
        embedder.retry_delay = 0.01

        with patch.object(
            embedder.client.embeddings,
            "create",
            side_effect=Exception("permanent failure"),
        ), patch("alphasignal.embeddings.embedder.time.sleep"):
            with pytest.raises(Exception, match="permanent failure"):
                embedder.embed_texts(["hello world"])
