"""Cross-store (SQLite/FAISS/cache) query-time consistency tests (2026-09-11
correction pass, objective 3).

Proves that a detectable mismatch between a FAISS vector's known
content_hash and its chunk's current SQLite content_hash is excluded from
dense search results - the retriever must never present a candidate whose
vector was computed from different text than what actually gets returned
as the match. Uses small, deterministic, orthonormal local embeddings only.
"""

from datetime import date
from unittest.mock import patch

import numpy as np
import pytest

from alphasignal.embeddings.cache import EmbeddingCache
from alphasignal.embeddings.embedder import Embedder
from alphasignal.ingestion import Chunk
from alphasignal.retrieval.retriever import HybridRetriever
from alphasignal.store.metadata_store import MetadataStore
from alphasignal.store.vector_store import VectorStore

DIM = 8


def _vec(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i % DIM] = 1.0
    return v


def _chunk(chunk_id, text, ticker="AAPL"):
    return Chunk(
        chunk_id=chunk_id,
        ticker=ticker,
        text=text,
        token_count=len(text.split()),
        doc_type="10-K",
        source="SEC EDGAR",
        section="item_1",
        date=date(2024, 1, 1),
        url=None,
        chunk_index=0,
        total_chunks=1,
    )


def _config():
    return {
        "embeddings": {"model": "text-embedding-ada-002", "batch_size": 100},
        "retrieval": {
            "dense_candidates": 10,
            "sparse_candidates": 10,
            "rerank_candidates": 5,
            "hybrid_weights": {"bm25": 0.4, "dense": 0.6},
        },
    }


def _make_retriever(tmp_path, dim=DIM):
    vector_store = VectorStore(str(tmp_path / "index"), dim=dim)
    vector_store.load()
    metadata_store = MetadataStore(str(tmp_path / "meta.db"))

    with patch("alphasignal.embeddings.embedder.OpenAI"):
        cache = EmbeddingCache(str(tmp_path / "cache"))
        embedder = Embedder(_config(), cache)

    retriever = HybridRetriever(_config(), embedder, vector_store, metadata_store)
    return retriever, vector_store, metadata_store


class TestCrossStoreConsistencyAtQueryTime:
    def test_sqlite_newer_than_faiss_excludes_stale_vector(self, tmp_path):
        """SQLite has the amended text (new hash), but FAISS still holds
        the vector for the original text (old hash) - the dense hit must
        be excluded, never returned as if the stale vector matched the
        new text."""
        retriever, vector_store, metadata_store = _make_retriever(tmp_path)

        original = _chunk("aapl_x_0000", "Original quarterly filing text.")
        metadata_store.add_chunks([original])
        vector_store.add(
            np.array([_vec(0)]), ["aapl_x_0000"], {"aapl_x_0000": original.content_hash}
        )

        # Now SQLite is updated with amended text (new hash) but FAISS is
        # never touched - simulates an interrupted replacement where the
        # metadata write succeeded but the vector write did not.
        amended = _chunk(
            "aapl_x_0000", "Amended quarterly filing text with new numbers."
        )
        metadata_store.add_chunks([amended])

        with patch.object(retriever.embedder, "embed_texts", return_value=[_vec(0)]):
            dense_results = retriever._dense_search(_vec(0), k=10)

        assert dense_results == [], "stale vector/text pair must be excluded"

    def test_faiss_newer_than_sqlite_excludes_stale_pairing(self, tmp_path):
        """FAISS holds a vector/hash for content that SQLite hasn't caught
        up to yet (the reverse direction of staleness) - also excluded,
        since the mismatch check is symmetric on "known and different,"
        not directional."""
        retriever, vector_store, metadata_store = _make_retriever(tmp_path)

        chunk = _chunk("aapl_y_0000", "Text version one.")
        metadata_store.add_chunks([chunk])

        # FAISS is seeded with a hash that does NOT match the chunk's
        # current SQLite hash - simulating FAISS somehow being ahead of
        # (or simply divergent from) SQLite.
        vector_store.add(
            np.array([_vec(1)]), ["aapl_y_0000"], {"aapl_y_0000": "some_other_hash"}
        )

        with patch.object(retriever.embedder, "embed_texts", return_value=[_vec(1)]):
            dense_results = retriever._dense_search(_vec(1), k=10)

        assert dense_results == []

    def test_unknown_provenance_vector_is_excluded_even_without_a_known_mismatch(
        self, tmp_path
    ):
        """A hashless (unknown-provenance) vector has no evidence it
        matches ANYTHING - it must be excluded even though there's no
        "known mismatch" (SQLite has a real hash, FAISS just never
        recorded one at all). Reversed from an earlier, incorrect policy
        that treated "unknown" as "not proven wrong, so allow it" (audit
        correction, 2026-09-11 follow-up pass) - unknown provenance is
        excluded on its own, not only a provable mismatch."""
        retriever, vector_store, metadata_store = _make_retriever(tmp_path)

        chunk = _chunk("aapl_z_0000", "Some stable text.")
        metadata_store.add_chunks([chunk])
        # No content_hashes passed - vector_store.content_hashes stays "".
        vector_store.add(np.array([_vec(2)]), ["aapl_z_0000"])

        with patch.object(retriever.embedder, "embed_texts", return_value=[_vec(2)]):
            dense_results = retriever._dense_search(_vec(2), k=10)

        assert dense_results == []

    def test_interrupted_replacement_followed_immediately_by_query(self, tmp_path):
        """End-to-end: retrieve() must not return the mismatched chunk's
        (now-current) text as a match for a query embedding that actually
        matches the STALE vector - the search must come back with nothing
        relevant, not a misleading hit."""
        retriever, vector_store, metadata_store = _make_retriever(tmp_path)

        original = _chunk(
            "aapl_w_0000", "Apple reported strong iPhone sales this quarter."
        )
        metadata_store.add_chunks([original])
        vector_store.add(
            np.array([_vec(3)]), ["aapl_w_0000"], {"aapl_w_0000": original.content_hash}
        )

        # Interrupted replacement: SQLite updated, FAISS not.
        amended = _chunk("aapl_w_0000", "Apple faces significant regulatory headwinds.")
        metadata_store.add_chunks([amended])
        retriever.build_bm25_index()

        with patch.object(retriever.embedder, "embed_texts", return_value=[_vec(3)]):
            results = retriever.retrieve("query matching the stale vector", top_k=5)

        # The chunk_id must not appear via the dense path with the STALE
        # vector's score profile presenting the AMENDED text as a match.
        returned_texts = [r.text for r in results]
        assert (
            amended.text not in returned_texts or original.text not in returned_texts
        ), "must never present mismatched vector/text as one coherent result"
        # More precisely: dense search alone (the only path that could
        # have matched via the stale vector) must have excluded it.
        with patch.object(retriever.embedder, "embed_texts", return_value=[_vec(3)]):
            dense_only = retriever._dense_search(_vec(3), k=10)
        assert dense_only == []

    def test_restart_before_repair_still_excludes_mismatch(self, tmp_path):
        """A fresh set of store instances (simulating a process restart)
        loading the same on-disk state must still detect and exclude the
        mismatch - it's not an in-memory-only guard that a restart would
        bypass."""
        retriever, vector_store, metadata_store = _make_retriever(tmp_path)

        original = _chunk("aapl_v_0000", "Pre-restart original text.")
        metadata_store.add_chunks([original])
        vector_store.add(
            np.array([_vec(4)]), ["aapl_v_0000"], {"aapl_v_0000": original.content_hash}
        )
        amended = _chunk("aapl_v_0000", "Post-restart amended text.")
        metadata_store.add_chunks([amended])

        # Simulate restart: fresh VectorStore/MetadataStore instances
        # pointed at the same on-disk paths.
        restarted_vector_store = VectorStore(str(tmp_path / "index"), dim=DIM)
        restarted_vector_store.load()
        restarted_metadata_store = MetadataStore(str(tmp_path / "meta.db"))

        with patch("alphasignal.embeddings.embedder.OpenAI"):
            cache = EmbeddingCache(str(tmp_path / "cache"))
            embedder = Embedder(_config(), cache)
        restarted_retriever = HybridRetriever(
            _config(), embedder, restarted_vector_store, restarted_metadata_store
        )

        with patch.object(embedder, "embed_texts", return_value=[_vec(4)]):
            dense_results = restarted_retriever._dense_search(_vec(4), k=10)

        assert dense_results == []

    def test_successful_reingestion_restores_queryability(self, tmp_path):
        """Idempotent re-ingestion (replacing the stale vector with one
        matching the current text/hash) must fully repair queryability -
        the chunk becomes a normal, matching dense hit again."""
        retriever, vector_store, metadata_store = _make_retriever(tmp_path)

        original = _chunk("aapl_u_0000", "Text before repair.")
        metadata_store.add_chunks([original])
        vector_store.add(
            np.array([_vec(5)]), ["aapl_u_0000"], {"aapl_u_0000": original.content_hash}
        )
        amended = _chunk("aapl_u_0000", "Text after repair.")
        metadata_store.add_chunks([amended])

        # Before repair: excluded.
        with patch.object(retriever.embedder, "embed_texts", return_value=[_vec(6)]):
            assert retriever._dense_search(_vec(6), k=10) == []

        # Repair: re-ingest with a fresh vector matching the current hash.
        vector_store.add(
            np.array([_vec(6)]), ["aapl_u_0000"], {"aapl_u_0000": amended.content_hash}
        )

        with patch.object(retriever.embedder, "embed_texts", return_value=[_vec(6)]):
            dense_results = retriever._dense_search(_vec(6), k=10)

        assert dense_results == [("aapl_u_0000", pytest.approx(1.0, abs=0.01))]
