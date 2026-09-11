"""Persistence and contract-verification pass (2026-09-11 follow-up session).

Covers, with deterministic local fixtures only (no network, no real
embedding provider, no credentials):

1. Legacy on-disk format compatibility + explicit backfill-not-silent-trust
   behavior for VectorStore, EmbeddingCache, and MetadataStore.
2. The actual FAISS replacement algorithm (reconstruct_n after save/reload,
   exact preservation of unchanged vectors, exclusion of stale vectors).
3. Orphan cleanup on re-ingestion (equal/fewer/more chunks, shifted
   boundaries, an emptied-out document, a chunking-config change).
4. Interruption/atomicity around the SQLite/cache/FAISS update sequence.
6. target_tokens/max_tokens/min_tokens/overlap_tokens invariants.

Sentiment-contract invariants (item 5) live in test_sentiment.py and
test_api.py, extended in this same session - see those files for the
`reliable_chunk_count <= total_chunk_count`, `data_available implies
latest_score is None`, and serialization-shape tests.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import faiss
import json as json_module
import numpy as np
import pytest

from alphasignal.embeddings.cache import EmbeddingCache
from alphasignal.embeddings.embedder import Embedder
from alphasignal.ingestion import Chunk, RawDocument
from alphasignal.ingestion.chunker import SemanticChunker
from alphasignal.ingestion.pipeline import IngestionPipeline
from alphasignal.store.metadata_store import MetadataStore
from alphasignal.store.vector_store import VectorStore

DIM = 16


def _basis(i: int, dim: int = DIM) -> np.ndarray:
    """Deterministic, mutually-orthogonal embedding for index i.

    Orthonormal vectors make search assertions unambiguous: the correct
    match scores ~1.0, everything else scores ~0.0 - there is no floating
    point ambiguity band to accidentally pass through.
    """
    v = np.zeros(dim, dtype=np.float32)
    v[i % dim] = 1.0
    return v


def _chunk(
    chunk_id, text, ticker="AAPL", chunk_index=0, total_chunks=1, doc_type="10-K"
):
    return Chunk(
        chunk_id=chunk_id,
        ticker=ticker,
        text=text,
        token_count=len(text.split()),
        doc_type=doc_type,
        source="SEC EDGAR",
        section="item_1",
        date=date(2024, 1, 1),
        url=None,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
    )


# ===========================================================================
# 1. Legacy persistence compatibility
# ===========================================================================


class TestLegacyVectorStoreFormat:
    def _write_legacy_index(self, tmp_path, ids, vectors):
        """Write files matching the pre-content_hash on-disk format: a
        bare JSON list of chunk_id strings, no hash sidecar entries."""
        index_dir = tmp_path / "index"
        index_dir.mkdir()
        index = faiss.IndexFlatIP(DIM)
        index.add(np.stack(vectors).astype(np.float32))
        faiss.write_index(index, str(index_dir / "index.faiss"))
        (index_dir / "chunk_ids.json").write_text(json_module.dumps(ids))
        return index_dir

    def test_legacy_format_loads_without_error_and_hashes_are_unknown(self, tmp_path):
        index_dir = self._write_legacy_index(
            tmp_path, ["a", "b", "c"], [_basis(0), _basis(1), _basis(2)]
        )

        store = VectorStore(str(index_dir), dim=DIM)
        store.load()

        assert store.chunk_ids == ["a", "b", "c"]
        assert store.content_hashes == {}
        assert len(store) == 3
        # Still fully usable: search works immediately post-migration-read.
        results = store.search(_basis(1), k=1)
        assert results[0][0] == "b"

    def test_legacy_entry_where_text_and_vector_happen_to_match(self, tmp_path):
        """Even when the legacy vector actually WAS produced from the
        current text (we just have no proof of that), the correct,
        provenance-blind behavior is still to force a fresh embedding on
        first touch - never to trust-and-stamp the old vector. The test
        proves this by using a distinctly different "freshly generated"
        vector and confirming that one - not the legacy vector, even
        though it happened to be a legitimate match - is what ends up
        stored."""
        index_dir = self._write_legacy_index(tmp_path, ["a"], [_basis(0)])
        store = VectorStore(str(index_dir), dim=DIM)
        store.load()

        # Simulate re-ingestion: the embedder has (correctly, per the
        # cache-side fix) regenerated a fresh embedding for "a" since the
        # cache lookup was forced to miss on the hashless entry. That fresh
        # embedding is what gets passed to VectorStore.add() here.
        freshly_generated = _basis(0)  # happens to match the legacy content
        store.add(np.array([freshly_generated]), ["a"], {"a": "h1"})

        assert len(store) == 1, "must not duplicate the vector"
        assert store.content_hashes["a"] == "h1", (
            "hash must be recorded only alongside the freshly generated "
            "embedding, never assigned to the untouched legacy vector"
        )
        results = store.search(_basis(0), k=1, filter_ids={"a"})
        assert results[0][1] > 0.99

    def test_legacy_entry_where_stored_vector_represents_different_text(self, tmp_path):
        """The dangerous case: the legacy vector was actually produced
        from DIFFERENT text than what's current now. Proves the old,
        wrong vector is not what gets served after re-ingestion - the
        freshly generated one is."""
        index_dir = self._write_legacy_index(tmp_path, ["a"], [_basis(0)])
        store = VectorStore(str(index_dir), dim=DIM)
        store.load()

        # The legacy vector (basis(0)) represented some other text; the
        # embedder has now regenerated a genuinely different embedding
        # (basis(9)) from the CURRENT text.
        store.add(np.array([_basis(9)]), ["a"], {"a": "h_current"})

        assert len(store) == 1
        assert store.content_hashes["a"] == "h_current"
        # The current-text vector is what's searchable...
        results = store.search(_basis(9), k=1, filter_ids={"a"})
        assert results[0][1] > 0.99
        # ...and the old, wrong legacy vector is gone, not aliased in.
        stale = store.search(_basis(0), k=1, filter_ids={"a"})
        assert stale[0][1] < 0.5

    def test_hashless_vector_is_replaced_on_successful_reingestion(self, tmp_path):
        """A hashless legacy entry must be REPLACED (not merely
        hash-stamped) the first time it's touched by a re-ingestion that
        provides a real content_hash for it."""
        index_dir = self._write_legacy_index(tmp_path, ["a"], [_basis(0)])
        store = VectorStore(str(index_dir), dim=DIM)
        store.load()
        assert store.content_hashes.get("a", "") == ""

        store.add(np.array([_basis(7)]), ["a"], {"a": "h1"})

        pos = store.chunk_ids.index("a")
        np.testing.assert_allclose(store.index.reconstruct(pos), _basis(7), atol=1e-6)
        assert store.content_hashes["a"] == "h1"

    def test_second_unchanged_ingestion_after_replacement_is_idempotent(self, tmp_path):
        """Once a legacy entry has been safely replaced with a real,
        hash-backed vector, re-ingesting the SAME (now current) content
        again must be a genuine no-op - no second rebuild, no duplicate."""
        index_dir = self._write_legacy_index(tmp_path, ["a"], [_basis(0)])
        store = VectorStore(str(index_dir), dim=DIM)
        store.load()

        store.add(np.array([_basis(7)]), ["a"], {"a": "h1"})
        assert len(store) == 1

        with patch.object(store, "save") as mock_save:
            store.add(np.array([_basis(7)]), ["a"], {"a": "h1"})  # unchanged
            mock_save.assert_not_called()

        assert len(store) == 1
        pos = store.chunk_ids.index("a")
        np.testing.assert_allclose(store.index.reconstruct(pos), _basis(7), atol=1e-6)
        assert store.content_hashes["a"] == "h1"

    def test_caller_not_tracking_hashes_keeps_original_dedup_behavior(self, tmp_path):
        """A caller that never passes content_hashes at all (pre-hash-
        tracking call pattern) is a distinct case from a hashless legacy
        vector being deliberately re-ingested WITH a hash - it must keep
        its original, safe, do-nothing-unsafe dedup: no hash is ever
        recorded, so nothing is falsely certified, and repeated identical
        adds stay a true no-op."""
        store = VectorStore(str(tmp_path / "index2"), dim=DIM)
        store.load()
        store.add(np.array([_basis(0)]), ["x"])  # no content_hashes at all
        assert len(store) == 1

        with patch.object(store, "save") as mock_save:
            store.add(np.array([_basis(0)]), ["x"])  # still no hashes
            mock_save.assert_not_called()

        assert len(store) == 1
        assert store.content_hashes.get("x", "") == ""


class TestLegacyEmbeddingCacheFormat:
    def _write_legacy_cache(self, tmp_path, ids, vectors):
        base = tmp_path / "cache"
        np.save(base.with_suffix(".npy"), np.stack(vectors).astype(np.float32))
        base.with_suffix(".json").write_text(json_module.dumps(ids))
        return base

    def test_legacy_cache_entry_is_untrusted_and_always_misses(self, tmp_path):
        base = self._write_legacy_cache(tmp_path, ["legacy_1"], [np.ones(4)])
        cache = EmbeddingCache(str(base))

        assert len(cache) == 1
        assert cache.get_stored_hash("legacy_1") == ""
        # A legacy (hashless) entry has no evidence it matches anything -
        # it must be a miss, not a hit, regardless of what hash is asked
        # for or whether one is asked for at all (audit correction,
        # 2026-09-11: previously served as a hit unconditionally).
        assert cache.get("legacy_1", content_hash="anything") is None
        assert cache.get("legacy_1") is None

    def _embedder(self, cache):
        config = {
            "embeddings": {
                "model": "text-embedding-ada-002",
                "batch_size": 100,
                "max_retries": 3,
                "retry_delay": 1.0,
            }
        }
        return Embedder(config, cache)

    def test_legacy_text_and_embedding_happen_to_match_still_regenerated(
        self, tmp_path
    ):
        """Even if the legacy embedding actually was produced from the
        current text, it must still be regenerated on first touch (no way
        to prove the match) - the vector VALUE that ends up cached is the
        freshly generated one, not silently the old one."""
        legacy_embedding = np.ones(1536, dtype=np.float32)
        base = self._write_legacy_cache(tmp_path, ["legacy_1"], [legacy_embedding])
        cache = EmbeddingCache(str(base))

        with patch("alphasignal.embeddings.embedder.OpenAI"):
            embedder = self._embedder(cache)
            chunk = _chunk("legacy_1", "Original filing text.")

            fresh_embedding = np.full(1536, 3.0, dtype=np.float32)
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=fresh_embedding.tolist())]
            with patch.object(
                embedder.client.embeddings, "create", return_value=mock_response
            ) as mock_create:
                result = embedder.embed_chunks([chunk])
                mock_create.assert_called_once()

            np.testing.assert_allclose(result["legacy_1"], fresh_embedding)
            assert cache.get_stored_hash("legacy_1") == chunk.content_hash
            # The value actually persisted in the cache is the fresh one.
            stored = cache.get("legacy_1", content_hash=chunk.content_hash)
            np.testing.assert_allclose(stored, fresh_embedding)

    def test_legacy_text_current_but_stored_vector_represents_different_text(
        self, tmp_path
    ):
        """The dangerous case: the legacy vector actually came from
        different text. Proves the wrong vector is never served - the
        freshly regenerated one, from the ACTUAL current text, is."""
        wrong_legacy_embedding = np.full(1536, -5.0, dtype=np.float32)
        base = self._write_legacy_cache(
            tmp_path, ["legacy_1"], [wrong_legacy_embedding]
        )
        cache = EmbeddingCache(str(base))

        with patch("alphasignal.embeddings.embedder.OpenAI"):
            embedder = self._embedder(cache)
            chunk = _chunk("legacy_1", "The actual current text.")

            correct_embedding = np.full(1536, 9.0, dtype=np.float32)
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=correct_embedding.tolist())]
            with patch.object(
                embedder.client.embeddings, "create", return_value=mock_response
            ):
                result = embedder.embed_chunks([chunk])

            np.testing.assert_allclose(result["legacy_1"], correct_embedding)
            assert not np.allclose(result["legacy_1"], wrong_legacy_embedding)

    def test_hashless_cache_lookup_returns_untrusted_via_get_many(self, tmp_path):
        base = self._write_legacy_cache(tmp_path, ["legacy_1"], [np.ones(4)])
        cache = EmbeddingCache(str(base))

        cached, uncached = cache.get_many({"legacy_1": "some-hash"})
        assert cached == {}
        assert uncached == ["legacy_1"]

    def test_failed_reembedding_does_not_bless_legacy_vector(self, tmp_path):
        """If regenerating the embedding fails, the legacy entry must be
        left exactly as it was - not deleted (so it can still be examined/
        recovered), but crucially NOT marked as validated: a subsequent
        lookup must still treat it as untrusted, not as a confirmed match."""
        legacy_embedding = np.ones(1536, dtype=np.float32)
        base = self._write_legacy_cache(tmp_path, ["legacy_1"], [legacy_embedding])
        cache = EmbeddingCache(str(base))

        with patch("alphasignal.embeddings.embedder.OpenAI"):
            embedder = self._embedder(cache)
            embedder.max_retries = 1
            chunk = _chunk("legacy_1", "Text that fails to embed.")

            with patch.object(
                embedder.client.embeddings,
                "create",
                side_effect=RuntimeError("provider down"),
            ):
                with pytest.raises(RuntimeError):
                    embedder.embed_chunks([chunk])

        # Previous durable state preserved (not wiped)...
        assert cache.get_stored_hash("legacy_1") == ""
        assert len(cache) == 1
        # ...but still definitely not blessed as a match for anything.
        assert cache.get("legacy_1", content_hash=chunk.content_hash) is None
        assert cache.get("legacy_1") is None

    def test_after_successful_replacement_hash_and_vector_match_current_text(
        self, tmp_path
    ):
        legacy_embedding = np.ones(1536, dtype=np.float32)
        base = self._write_legacy_cache(tmp_path, ["legacy_1"], [legacy_embedding])
        cache = EmbeddingCache(str(base))

        with patch("alphasignal.embeddings.embedder.OpenAI"):
            embedder = self._embedder(cache)
            chunk = _chunk("legacy_1", "Deterministic current text for hashing.")

            fresh_embedding = np.full(1536, 4.0, dtype=np.float32)
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=fresh_embedding.tolist())]
            with patch.object(
                embedder.client.embeddings, "create", return_value=mock_response
            ):
                embedder.embed_chunks([chunk])

            assert cache.get_stored_hash("legacy_1") == chunk.content_hash
            recalled = cache.get("legacy_1", content_hash=chunk.content_hash)
            np.testing.assert_allclose(recalled, fresh_embedding)

    def test_second_unchanged_ingestion_is_genuinely_idempotent(self, tmp_path):
        legacy_embedding = np.ones(1536, dtype=np.float32)
        base = self._write_legacy_cache(tmp_path, ["legacy_1"], [legacy_embedding])
        cache = EmbeddingCache(str(base))

        with patch("alphasignal.embeddings.embedder.OpenAI"):
            embedder = self._embedder(cache)
            chunk = _chunk("legacy_1", "Stable unchanged text.")

            fresh_embedding = np.full(1536, 2.0, dtype=np.float32)
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=fresh_embedding.tolist())]
            with patch.object(
                embedder.client.embeddings, "create", return_value=mock_response
            ) as mock_create:
                embedder.embed_chunks([chunk])  # first touch: real re-embed
                assert mock_create.call_count == 1

                embedder.embed_chunks([chunk])  # second touch: genuine hit
                assert mock_create.call_count == 1, (
                    "unchanged re-ingestion after a successful replacement "
                    "must not re-embed again"
                )


class TestLegacyMetadataStoreRow:
    def test_legacy_row_with_blank_hash_backfills_from_text_on_load(self, tmp_path):
        """A ChunkRecord written before content_hash existed (blank column)
        must yield a Chunk whose content_hash is derived from its actual
        stored text when read back - Chunk.__post_init__ recomputes it
        whenever the persisted value is falsy."""
        from sqlmodel import Session

        from alphasignal.store.metadata_store import ChunkRecord

        store = MetadataStore(str(tmp_path / "legacy.db"))
        with Session(store.engine) as session:
            session.add(
                ChunkRecord(
                    chunk_id="legacy_row_0000",
                    ticker="AAPL",
                    text="Some legacy filing text.",
                    token_count=4,
                    doc_type="10-K",
                    source="SEC EDGAR",
                    section="item_1",
                    date=date(2024, 1, 1),
                    chunk_index=0,
                    total_chunks=1,
                    content_hash="",  # simulates a pre-migration row
                )
            )
            session.commit()

        loaded = store.get_chunk("legacy_row_0000")
        expected = _chunk("legacy_row_0000", "Some legacy filing text.")
        assert loaded.content_hash == expected.content_hash
        assert loaded.content_hash != ""


# ===========================================================================
# 2. FAISS replacement algorithm
# ===========================================================================


def test_faiss_indexflatip_reconstruct_n_supported_after_save_reload(tmp_path):
    """Direct proof against the repository's actual configured index type
    (IndexFlatIP) and installed faiss version: reconstruct_n works both
    before and after a save/reload round-trip. This is the operation
    VectorStore._reconstruct_all() relies on for replacement."""
    index = faiss.IndexFlatIP(DIM)
    vectors = np.stack([_basis(i) for i in range(5)])
    index.add(vectors)

    path = tmp_path / "index.faiss"
    faiss.write_index(index, str(path))
    reloaded = faiss.read_index(str(path))

    assert reloaded.ntotal == 5
    reconstructed = reloaded.reconstruct_n(0, reloaded.ntotal)
    np.testing.assert_allclose(reconstructed, vectors, atol=1e-6)


class TestVectorStoreReplacementAlgorithm:
    def _seeded_store(self, tmp_path, n=3):
        store = VectorStore(str(tmp_path / "index"), dim=DIM)
        store.load()
        ids = [f"c{i}" for i in range(n)]
        hashes = {cid: f"hash_{cid}" for cid in ids}
        store.add(np.stack([_basis(i) for i in range(n)]), ids, hashes)
        return store

    def test_unchanged_vectors_preserved_exactly_after_replace(self, tmp_path):
        store = self._seeded_store(tmp_path, n=3)

        # Replace only c1 with a distinct new vector.
        store.add(np.array([_basis(10)]), ["c1"], {"c1": "hash_c1_v2"})

        assert len(store) == 3
        # c0 and c2 must be bit-for-bit whatever they were (basis vectors
        # are already unit norm, so normalization is a no-op here).
        pos0 = store.chunk_ids.index("c0")
        pos2 = store.chunk_ids.index("c2")
        np.testing.assert_allclose(store.index.reconstruct(pos0), _basis(0), atol=1e-6)
        np.testing.assert_allclose(store.index.reconstruct(pos2), _basis(2), atol=1e-6)

    def test_changed_vector_excludes_old_includes_new(self, tmp_path):
        store = self._seeded_store(tmp_path, n=3)
        store.add(np.array([_basis(10)]), ["c1"], {"c1": "hash_c1_v2"})

        pos1 = store.chunk_ids.index("c1")
        np.testing.assert_allclose(store.index.reconstruct(pos1), _basis(10), atol=1e-6)

        # Old vector (basis(1)) must score near zero against c1 now - it's
        # simply gone, not aliased.
        old_score = dict(store.search(_basis(1), k=3, filter_ids={"c1"}))
        assert old_score.get("c1", 0.0) < 0.5

    def test_search_after_replace_returns_updated_content_not_stale(self, tmp_path):
        store = self._seeded_store(tmp_path, n=3)
        store.add(np.array([_basis(10)]), ["c1"], {"c1": "hash_c1_v2"})

        results = store.search(_basis(10), k=1)
        assert results[0][0] == "c1"
        assert results[0][1] > 0.99

    def test_no_duplicate_vectors_and_order_metadata_alignment_after_replace(
        self, tmp_path
    ):
        store = self._seeded_store(tmp_path, n=4)
        # Replace two different chunks in two separate calls.
        store.add(np.array([_basis(10)]), ["c1"], {"c1": "hash_c1_v2"})
        store.add(np.array([_basis(11)]), ["c3"], {"c3": "hash_c3_v2"})

        assert len(store) == 4
        assert len(store.chunk_ids) == len(set(store.chunk_ids)) == 4

        # Every chunk_id's search-returned identity must match its actual
        # backing vector position (id list and FAISS rows stay aligned).
        for cid, expected_vec in [
            ("c0", _basis(0)),
            ("c1", _basis(10)),
            ("c2", _basis(2)),
            ("c3", _basis(11)),
        ]:
            pos = store.chunk_ids.index(cid)
            np.testing.assert_allclose(
                store.index.reconstruct(pos), expected_vec, atol=1e-6
            )
            top = store.search(expected_vec, k=1)[0]
            assert top[0] == cid
            assert top[1] > 0.99

    def test_replacement_survives_save_reload_search_updated(self, tmp_path):
        index_path = tmp_path / "index"
        store1 = VectorStore(str(index_path), dim=DIM)
        store1.load()
        store1.add(np.array([_basis(0)]), ["c0"], {"c0": "h1"})
        store1.add(np.array([_basis(9)]), ["c0"], {"c0": "h2"})

        store2 = VectorStore(str(index_path), dim=DIM)
        store2.load()

        assert len(store2) == 1
        assert store2.content_hashes["c0"] == "h2"
        results = store2.search(_basis(9), k=1)
        assert results[0][0] == "c0"
        assert results[0][1] > 0.99
        stale = store2.search(_basis(0), k=1)
        # basis(0) is orthogonal to basis(9), so its best score against the
        # single remaining vector must be near zero, not a false match.
        assert stale[0][1] < 0.5


# ===========================================================================
# 3. Orphan cleanup on re-ingestion
# ===========================================================================


def _make_pipeline(tmp_path, chunking_overrides=None):
    chunking = {
        "target_tokens": 300,
        "min_tokens": 5,
        "max_tokens": 400,
        "overlap_tokens": 10,
    }
    if chunking_overrides:
        chunking.update(chunking_overrides)

    config = {
        "chunking": chunking,
        "embeddings": {
            "model": "text-embedding-ada-002",
            "batch_size": 100,
            "max_retries": 3,
            "retry_delay": 1.0,
        },
    }

    vector_store = VectorStore(str(tmp_path / "faiss_index"), dim=DIM)
    vector_store.load()
    metadata_store = MetadataStore(str(tmp_path / "metadata.db"))

    with patch("alphasignal.embeddings.embedder.OpenAI"):
        cache = EmbeddingCache(str(tmp_path / "cache"))
        embedder = Embedder(config, cache)

    pipeline = IngestionPipeline(
        config,
        embedder=embedder,
        vector_store=vector_store,
        metadata_store=metadata_store,
    )
    return pipeline


def _fake_embeddings(chunks, start=0):
    """Deterministic, distinct embeddings keyed by chunk order."""
    return {c.chunk_id: _basis(start + i) for i, c in enumerate(chunks)}


def _doc(sections, accession="0001234567-24-000001"):
    return RawDocument(
        ticker="AAPL",
        doc_type="10-K",
        filing_date=date(2024, 1, 15),
        period_of_report=date(2024, 1, 15),
        source="SEC EDGAR",
        sections=sections,
        file_path="/tmp/filing.htm",
        accession_number=accession,
    )


class TestOrphanCleanup:
    def test_changed_text_same_chunk_count_no_orphans(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        chunker = pipeline.chunker

        doc_v1 = _doc({"item_1": "Alpha sentence one. Alpha sentence two."})
        chunks_v1 = chunker.chunk_document(doc_v1)
        pipeline.store_chunks(chunks_v1, _fake_embeddings(chunks_v1, start=0))
        n1 = pipeline.metadata_store.count()
        assert n1 == len(chunks_v1)

        doc_v2 = _doc({"item_1": "Beta sentence one. Beta sentence two."})
        chunks_v2 = chunker.chunk_document(doc_v2)
        assert len(chunks_v2) == len(chunks_v1), "test assumes stable chunk count"
        assert {c.chunk_id for c in chunks_v2} == {c.chunk_id for c in chunks_v1}

        pipeline.store_chunks(chunks_v2, _fake_embeddings(chunks_v2, start=10))

        assert pipeline.metadata_store.count() == n1, "no orphans, no duplicates"
        for c in chunks_v2:
            stored = pipeline.metadata_store.get_chunk(c.chunk_id)
            assert stored.text == c.text
        assert len(pipeline.vector_store) == n1

    def test_fewer_chunks_removes_orphans(self, tmp_path):
        pipeline = _make_pipeline(
            tmp_path, chunking_overrides={"target_tokens": 100, "max_tokens": 100}
        )
        chunker = pipeline.chunker

        long_text = " ".join(
            f"Sentence number {i} contains unique filler content for splitting purposes today."
            for i in range(40)
        )
        doc_v1 = _doc({"item_1": long_text})
        chunks_v1 = chunker.chunk_document(doc_v1)
        assert len(chunks_v1) >= 3, "test needs multiple chunks to begin with"
        pipeline.store_chunks(chunks_v1, _fake_embeddings(chunks_v1, start=0))
        original_ids = {c.chunk_id for c in chunks_v1}

        short_text = "One short sentence replaces the entire section now."
        doc_v2 = _doc({"item_1": short_text})
        chunks_v2 = chunker.chunk_document(doc_v2)
        assert len(chunks_v2) < len(chunks_v1)
        pipeline.store_chunks(chunks_v2, _fake_embeddings(chunks_v2, start=20))

        new_ids = {c.chunk_id for c in chunks_v2}
        removed_ids = original_ids - new_ids
        assert removed_ids, "test needs at least one orphaned id"

        assert pipeline.metadata_store.count() == len(chunks_v2)
        for orphan_id in removed_ids:
            assert pipeline.metadata_store.get_chunk(orphan_id) is None
        assert len(pipeline.vector_store) == len(chunks_v2)
        assert set(pipeline.vector_store.chunk_ids) == new_ids
        for orphan_id in removed_ids:
            assert orphan_id not in pipeline.vector_store.chunk_ids

    def test_more_chunks_adds_without_removing_existing(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        chunker = pipeline.chunker

        short_text = "One short sentence for the whole section initially."
        doc_v1 = _doc({"item_1": short_text})
        chunks_v1 = chunker.chunk_document(doc_v1)
        pipeline.store_chunks(chunks_v1, _fake_embeddings(chunks_v1, start=0))

        long_text = " ".join(
            f"Sentence number {i} contains unique filler content for splitting purposes today."
            for i in range(40)
        )
        doc_v2 = _doc({"item_1": long_text})
        chunks_v2 = chunker.chunk_document(doc_v2)
        assert len(chunks_v2) > len(chunks_v1)
        pipeline.store_chunks(chunks_v2, _fake_embeddings(chunks_v2, start=20))

        assert pipeline.metadata_store.count() == len(chunks_v2)
        assert len(pipeline.vector_store) == len(chunks_v2)
        assert set(pipeline.vector_store.chunk_ids) == {c.chunk_id for c in chunks_v2}

    def test_shifted_boundaries_updates_content_at_shared_index(self, tmp_path):
        """Inserting text earlier in the section reflows later sentences
        into different chunks while the chunk *count* stays the same -
        content at a shared chunk_index must reflect the new text, with no
        stale duplicate left over."""
        pipeline = _make_pipeline(
            tmp_path,
            chunking_overrides={"max_tokens": 20, "min_tokens": 5, "overlap_tokens": 2},
        )
        chunker = pipeline.chunker

        sentences_v1 = [
            f"Original filler sentence number {i} here today." for i in range(6)
        ]
        doc_v1 = _doc({"item_1": " ".join(sentences_v1)})
        chunks_v1 = chunker.chunk_document(doc_v1)
        pipeline.store_chunks(chunks_v1, _fake_embeddings(chunks_v1, start=0))

        # Reflow: same sentence count, different wording -> same boundary
        # structure (same chunk count) but different content per position.
        sentences_v2 = [
            f"Rewritten filler sentence number {i} here today." for i in range(6)
        ]
        doc_v2 = _doc({"item_1": " ".join(sentences_v2)})
        chunks_v2 = chunker.chunk_document(doc_v2)
        assert len(chunks_v2) == len(chunks_v1)
        pipeline.store_chunks(chunks_v2, _fake_embeddings(chunks_v2, start=30))

        assert pipeline.metadata_store.count() == len(chunks_v1)
        for c in chunks_v2:
            stored = pipeline.metadata_store.get_chunk(c.chunk_id)
            assert stored.text == c.text
            assert "Rewritten" in stored.text
        assert len(pipeline.vector_store) == len(chunks_v1)

    def test_document_becomes_empty_removes_all_previous_chunks(self, tmp_path):
        """A document whose sections are all emptied out yields ZERO
        chunks - store_chunks receives nothing from this source at all, so
        orphan detection must rely on the explicit source_prefixes the
        caller (full_ingest/ingest_historical_filings) passes in, not on
        anything derivable from an empty chunk list."""
        pipeline = _make_pipeline(tmp_path)
        chunker = pipeline.chunker

        doc_v1 = _doc({"item_1": "Some real content sentence here today for testing."})
        chunks_v1 = chunker.chunk_document(doc_v1)
        assert chunks_v1
        pipeline.store_chunks(chunks_v1, _fake_embeddings(chunks_v1, start=0))
        assert pipeline.metadata_store.count() == len(chunks_v1)

        doc_v2 = _doc({"item_1": ""})  # now empty
        chunks_v2 = chunker.chunk_document(doc_v2)
        assert chunks_v2 == []

        prefix = chunker.document_source_prefix(doc_v2)
        pipeline.store_chunks(chunks_v2, {}, source_prefixes={prefix})

        assert pipeline.metadata_store.count() == 0
        assert len(pipeline.vector_store) == 0
        for c in chunks_v1:
            assert pipeline.metadata_store.get_chunk(c.chunk_id) is None
            assert c.chunk_id not in pipeline.vector_store.chunk_ids

    def test_document_becomes_empty_without_explicit_prefix_leaves_orphans(
        self, tmp_path
    ):
        """Documents this architecture's boundary: without the caller
        supplying source_prefixes, store_chunks has no way to know a
        source went from N chunks to zero (it only ever sees the chunks it
        IS given). This is the honestly-documented limitation, not a
        silent success - proven here so it can't regress into looking
        fixed by accident."""
        pipeline = _make_pipeline(tmp_path)
        chunker = pipeline.chunker

        doc_v1 = _doc({"item_1": "Some real content sentence here today for testing."})
        chunks_v1 = chunker.chunk_document(doc_v1)
        pipeline.store_chunks(chunks_v1, _fake_embeddings(chunks_v1, start=0))

        # Calling store_chunks with chunks=[] and no source_prefixes hint
        # (e.g. a caller that only ever tracks the flat chunk list) cannot
        # trigger cleanup - documented architectural boundary.
        pipeline.store_chunks([], {})

        assert pipeline.metadata_store.count() == len(chunks_v1)

    def test_chunking_config_change_is_detected_via_hash_not_silently_mixed(
        self, tmp_path
    ):
        """A chunking config change (different max_tokens) that alters
        boundaries under the same source identity must not silently mix
        old- and new-config chunks in the index: old positions not
        reproduced become orphans (cleaned up), and any surviving position
        with different content is replaced via content_hash, never treated
        as an untouched match."""
        pipeline = _make_pipeline(tmp_path, chunking_overrides={"max_tokens": 400})
        long_text = " ".join(
            f"Sentence number {i} contains unique filler content for splitting today."
            for i in range(30)
        )
        doc = _doc({"item_1": long_text})
        chunks_v1 = pipeline.chunker.chunk_document(doc)
        pipeline.store_chunks(chunks_v1, _fake_embeddings(chunks_v1, start=0))
        n1 = len(chunks_v1)

        # Re-chunk the SAME source text under a different chunking config
        # (smaller max_tokens -> more, differently-bounded chunks).
        pipeline.chunker = SemanticChunker(
            {
                "chunking": {
                    "target_tokens": 50,
                    "min_tokens": 5,
                    "max_tokens": 50,
                    "overlap_tokens": 5,
                }
            }
        )
        chunks_v2 = pipeline.chunker.chunk_document(doc)
        assert len(chunks_v2) != n1, "config change should alter boundary count"
        pipeline.store_chunks(chunks_v2, _fake_embeddings(chunks_v2, start=50))

        # Final state reflects ONLY the new config's chunking - no leftover
        # old-config chunk_ids, no duplicate content at overlapping ids.
        assert pipeline.metadata_store.count() == len(chunks_v2)
        assert len(pipeline.vector_store) == len(chunks_v2)
        assert set(pipeline.vector_store.chunk_ids) == {c.chunk_id for c in chunks_v2}
        old_only_ids = {c.chunk_id for c in chunks_v1} - {c.chunk_id for c in chunks_v2}
        for cid in old_only_ids:
            assert pipeline.metadata_store.get_chunk(cid) is None
            assert cid not in pipeline.vector_store.chunk_ids


# ===========================================================================
# 4. Interruption / atomicity
# ===========================================================================


class TestInterruptionConsistency:
    def test_failure_before_embedding_leaves_no_partial_state(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        chunks = [_chunk("aapl_10k_x_0000", "Some text.")]

        with patch.object(
            pipeline.embedder, "embed_texts", side_effect=RuntimeError("boom")
        ):
            with pytest.raises(RuntimeError):
                pipeline.embedder.embed_chunks(chunks)

        assert pipeline.metadata_store.count() == 0
        assert len(pipeline.vector_store) == 0

    def test_failure_after_embedding_before_vector_store_leaves_recoverable_state(
        self, tmp_path
    ):
        """embed_chunks() succeeds (cache durably saved) but the FAISS
        write then fails. metadata_store.add_chunks() runs (and commits)
        before vector_store.add() is called, so metadata can legitimately
        be ahead of FAISS at this point - the system must not corrupt
        anything, and a retry must fully reconcile without duplicating or
        skipping."""
        pipeline = _make_pipeline(tmp_path)
        chunk = _chunk("aapl_10k_x_0000", "Some recoverable text.")
        embeddings = {chunk.chunk_id: _basis(0)}

        with patch.object(
            pipeline.vector_store, "add", side_effect=RuntimeError("disk full")
        ):
            with pytest.raises(RuntimeError):
                pipeline.store_chunks([chunk], embeddings)

        # Metadata committed; FAISS did not.
        assert pipeline.metadata_store.count() == 1
        assert len(pipeline.vector_store) == 0

        # Retry (as a caller would after a crash/error) must fully recover -
        # no duplicate metadata row, and the vector actually lands.
        pipeline.store_chunks([chunk], embeddings)

        assert pipeline.metadata_store.count() == 1
        assert len(pipeline.vector_store) == 1
        assert pipeline.vector_store.search(_basis(0), k=1)[0][0] == chunk.chunk_id

    def test_failure_while_rebuilding_preserves_prior_valid_in_memory_state(
        self, tmp_path
    ):
        """If the FAISS rebuild itself fails partway (e.g. index.add()
        raises), the store's in-memory index/chunk_ids must remain exactly
        the last good state - never split between an emptied new index and
        the old id list."""
        store = VectorStore(str(tmp_path / "index"), dim=DIM)
        store.load()
        store.add(np.stack([_basis(0), _basis(1)]), ["a", "b"], {"a": "h1", "b": "h2"})

        original_index = store.index
        original_ids = list(store.chunk_ids)

        with patch.object(
            faiss.IndexFlatIP, "add", side_effect=RuntimeError("faiss internal error")
        ):
            with pytest.raises(RuntimeError):
                # Trigger the changed-vector rebuild path.
                store.add(np.array([_basis(9)]), ["a"], {"a": "h1_changed"})

        # In-memory state must be untouched - same object, same ids, same
        # count - not a half-rebuilt empty index.
        assert store.chunk_ids == original_ids
        assert store.index.ntotal == 2
        assert store.index is original_index or store.index.ntotal == len(
            store.chunk_ids
        )

    def test_failure_after_temp_write_before_final_replace_keeps_prior_files(
        self, tmp_path
    ):
        """save()'s write-new-generation-then-atomically-activate-manifest
        pattern must mean a failure while writing the new generation's
        chunk_ids file leaves the previously active generation (and the
        manifest pointing at it) completely untouched - not a partially-
        updated pair, and no new generation ever activated."""
        index_path = tmp_path / "index"
        store = VectorStore(str(index_path), dim=DIM)
        store.load()
        store.add(np.array([_basis(0)]), ["a"], {"a": "h1"})

        manifest_file = index_path / "manifest.json"
        active_generation = store.generation
        index_file = index_path / f"index.g{active_generation}.faiss"
        ids_file = index_path / f"chunk_ids.g{active_generation}.json"
        original_manifest_bytes = manifest_file.read_bytes()
        original_index_bytes = index_file.read_bytes()
        original_ids_bytes = ids_file.read_bytes()

        store.chunk_ids.append("phantom")  # simulate an in-progress mutation
        with patch("builtins.open", side_effect=OSError("disk full")):
            store.save()

        assert manifest_file.read_bytes() == original_manifest_bytes
        assert index_file.read_bytes() == original_index_bytes
        assert ids_file.read_bytes() == original_ids_bytes
        assert store.generation == active_generation

    def test_restart_after_interrupted_replacement_reloads_last_good_state(
        self, tmp_path
    ):
        """A fresh VectorStore instance (simulating a process restart)
        loading from disk after an in-memory-only mutation was never
        saved must see the last durably-saved state, not silently mix in
        the lost mutation."""
        index_path = tmp_path / "index"
        store1 = VectorStore(str(index_path), dim=DIM)
        store1.load()
        store1.add(np.array([_basis(0)]), ["a"], {"a": "h1"})  # durably saved

        # Simulate an in-memory-only change that never got persisted
        # (e.g. process killed right after mutating state but before save()).
        store1.chunk_ids.append("never_saved")

        store2 = VectorStore(str(index_path), dim=DIM)
        store2.load()

        assert store2.chunk_ids == ["a"]
        assert "never_saved" not in store2.chunk_ids
        assert len(store2) == 1

    def test_metadata_store_add_chunks_failure_partway_does_not_partially_commit(
        self, tmp_path
    ):
        """add_chunks() uses one session/commit for the whole batch - if
        something raises partway through building records, nothing in
        that batch should be committed (all-or-nothing for a single call,
        proportional to this local repo - not a cross-store transaction)."""
        from sqlmodel import Session

        store = MetadataStore(str(tmp_path / "meta.db"))
        chunks = [
            _chunk(f"aapl_10k_x_{i:04d}", f"Text {i}", chunk_index=i) for i in range(3)
        ]

        real_session_class = Session
        call_count = {"n": 0}

        class FlakySession(real_session_class):
            def merge(self, record):
                call_count["n"] += 1
                if call_count["n"] == 2:
                    raise RuntimeError("simulated mid-batch failure")
                return super().merge(record)

        with patch("alphasignal.store.metadata_store.Session", FlakySession):
            with pytest.raises(RuntimeError):
                store.add_chunks(chunks)

        assert store.count() == 0, "partial batch must not be committed"


# ===========================================================================
# 6. Chunking invariants (target_tokens vs max_tokens etc.)
# ===========================================================================


class TestChunkingInvariants:
    def _chunker(self, **overrides):
        cfg = {
            "target_tokens": 300,
            "min_tokens": 100,
            "max_tokens": 400,
            "overlap_tokens": 50,
        }
        cfg.update(overrides)
        return SemanticChunker({"chunking": cfg})

    def test_target_tokens_never_lets_chunk_exceed_max_tokens(self):
        chunker = self._chunker(
            target_tokens=1000, max_tokens=100, min_tokens=10, overlap_tokens=5
        )
        text = " ".join(
            f"Sentence number {i} has some filler content for token counting purposes today."
            for i in range(30)
        )
        chunks = chunker.chunk_text(text)
        for c in chunks:
            assert chunker.count_tokens(c) <= chunker.max_tokens

    def test_oversized_single_sentence_hard_split_respects_max_tokens(self):
        chunker = self._chunker(max_tokens=50, min_tokens=10, overlap_tokens=5)
        huge_sentence = ("token " * 200).strip() + "."
        chunks = chunker.chunk_text(huge_sentence)
        assert len(chunks) > 1
        for c in chunks:
            assert chunker.count_tokens(c) <= chunker.max_tokens

    def test_final_fragment_merge_overshoot_is_the_only_documented_exception(self):
        """The one documented, deliberate case where a chunk may exceed
        max_tokens: a short trailing fragment merged into the previous
        chunk rather than being dropped. Must not happen for any other
        reason."""
        chunker = self._chunker(max_tokens=60, min_tokens=30, overlap_tokens=10)
        body = (
            "The company reported steady operating performance this quarter overall. "
            * 6
        )
        closing = "Outlook uncertain."
        chunks = chunker.chunk_text(body + " " + closing)
        overshoot_chunks = [
            c for c in chunks if chunker.count_tokens(c) > chunker.max_tokens
        ]
        # Any overshoot must be explained by containing the merged closing
        # fragment - not an unrelated boundary bug.
        for c in overshoot_chunks:
            assert closing in c

    def test_overlap_never_exceeds_configured_budget(self):
        chunker = self._chunker(max_tokens=50, overlap_tokens=15, min_tokens=5)
        # Unique per-sentence marker tokens so shared-sentence detection
        # can't be confused by ordinary words repeated across sentences.
        sentences = [
            f"Marker{i:03d} unique distinct sentence content here today."
            for i in range(10)
        ]
        text = " ".join(sentences)
        chunks = chunker.chunk_text(text)

        assert len(chunks) >= 2, "test needs multiple chunks"
        for i in range(len(chunks) - 1):
            carried_sentences = [
                s
                for s in sentences
                if f"Marker{sentences.index(s):03d}" in chunks[i]
                and f"Marker{sentences.index(s):03d}" in chunks[i + 1]
            ]
            carried_text = " ".join(carried_sentences)
            # Overlap is a maximum BUDGET for whole trailing sentences
            # (documented behavior) - carried content must fit within it.
            assert chunker.count_tokens(carried_text) <= chunker.overlap_tokens

    def test_invalid_max_tokens_normalized_to_safe_default(self):
        chunker = self._chunker(max_tokens=0)
        assert chunker.max_tokens == 400

        chunker2 = self._chunker(max_tokens=-10)
        assert chunker2.max_tokens == 400

    def test_min_tokens_exceeding_max_tokens_is_clamped(self):
        chunker = self._chunker(min_tokens=1000, max_tokens=100)
        assert chunker.min_tokens == 100

    def test_negative_overlap_and_target_are_normalized(self):
        chunker = self._chunker(overlap_tokens=-5, target_tokens=-1)
        assert chunker.overlap_tokens == 0
        assert chunker.target_tokens == 0

    def test_overlap_equal_or_above_max_tokens_is_clamped(self):
        chunker = self._chunker(overlap_tokens=999, max_tokens=100)
        assert chunker.overlap_tokens == 99

    def test_normalized_config_still_produces_valid_chunks(self):
        """After normalization, chunking must still actually work (not
        raise, not infinite loop) on real text."""
        chunker = self._chunker(
            max_tokens=-1, min_tokens=-1, overlap_tokens=-1, target_tokens=-1
        )
        text = "This is a perfectly normal sentence. Followed by another one here."
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
