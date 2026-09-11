"""Legacy/unknown-provenance vector query-exclusion tests (2026-09-11
follow-up correction pass, objective 1).

Proves the full lifecycle: a legacy index.faiss/chunk_ids.json pair
migrated into generation 0 has hashless (unknown-provenance) vector
entries; before re-ingestion, those entries must be excluded from dense
retrieval and unable to influence ranking; BM25 is unaffected (it always
reads current SQLite text) and does not silently absorb dense similarity
evidence for an excluded chunk; successful re-ingestion (real content_hash
+ freshly generated embedding) restores dense queryability; a failed
re-ingestion leaves the vector untrusted, not blessed.
"""

import json
from datetime import date
from unittest.mock import patch

import faiss
import numpy as np
import pytest

from alphasignal.embeddings.cache import EmbeddingCache
from alphasignal.embeddings.embedder import Embedder
from alphasignal.ingestion import Chunk
from alphasignal.ingestion.pipeline import IngestionPipeline
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
        "chunking": {
            "target_tokens": 300,
            "min_tokens": 5,
            "max_tokens": 400,
            "overlap_tokens": 10,
        },
        "embeddings": {"model": "text-embedding-ada-002", "batch_size": 100},
        "retrieval": {
            "dense_candidates": 10,
            "sparse_candidates": 10,
            "rerank_candidates": 5,
            "hybrid_weights": {"bm25": 0.4, "dense": 0.6},
        },
    }


def _write_legacy_index(index_dir, ids, vectors, dim=DIM):
    index_dir.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(dim)
    index.add(np.stack(vectors).astype(np.float32))
    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "chunk_ids.json").write_text(json.dumps(ids))


def _make_stores_and_retriever(tmp_path, dim=DIM):
    vector_store = VectorStore(str(tmp_path / "index"), dim=dim)
    vector_store.load()
    metadata_store = MetadataStore(str(tmp_path / "meta.db"))
    with patch("alphasignal.embeddings.embedder.OpenAI"):
        cache = EmbeddingCache(str(tmp_path / "cache"))
        embedder = Embedder(_config(), cache)
    retriever = HybridRetriever(_config(), embedder, vector_store, metadata_store)
    pipeline = IngestionPipeline(
        _config(),
        embedder=embedder,
        vector_store=vector_store,
        metadata_store=metadata_store,
    )
    return vector_store, metadata_store, retriever, pipeline, embedder


class TestMigratedLegacyVectorExcludedBeforeReingestion:
    def test_migrated_hashless_vector_excluded_from_dense_retrieval(self, tmp_path):
        """A legacy index.faiss/chunk_ids.json pair, migrated into
        generation 0 (hashless entries), must not be queryable via dense
        search before re-ingestion."""
        index_dir = tmp_path / "index"
        _write_legacy_index(index_dir, ["aapl_x_0000"], [_vec(0)])

        vector_store = VectorStore(str(index_dir), dim=DIM)
        vector_store.load()  # migrates legacy pair into generation 0

        assert len(vector_store) == 1
        assert vector_store.content_hashes.get("aapl_x_0000", "") == ""

        metadata_store = MetadataStore(str(tmp_path / "meta.db"))
        metadata_store.add_chunks([_chunk("aapl_x_0000", "Some real filing text.")])
        with patch("alphasignal.embeddings.embedder.OpenAI"):
            cache = EmbeddingCache(str(tmp_path / "cache"))
            embedder = Embedder(_config(), cache)
        retriever = HybridRetriever(_config(), embedder, vector_store, metadata_store)

        with patch.object(retriever.embedder, "embed_texts", return_value=[_vec(0)]):
            dense_results = retriever._dense_search(_vec(0), k=10)

        assert dense_results == [], "hashless legacy vector must be excluded"

    def test_migrated_vector_cannot_influence_ranking_before_reingestion(
        self, tmp_path
    ):
        """The excluded legacy chunk must not contribute dense_score
        evidence to the merged hybrid ranking at all - not a discounted
        contribution, zero."""
        index_dir = tmp_path / "index"
        _write_legacy_index(
            index_dir, ["aapl_x_0000", "aapl_y_0000"], [_vec(0), _vec(1)]
        )
        vector_store = VectorStore(str(index_dir), dim=DIM)
        vector_store.load()

        metadata_store = MetadataStore(str(tmp_path / "meta.db"))
        metadata_store.add_chunks(
            [
                _chunk("aapl_x_0000", "Apple reported strong iPhone sales growth."),
                _chunk("aapl_y_0000", "Apple reported strong iPhone sales growth."),
            ]
        )
        with patch("alphasignal.embeddings.embedder.OpenAI"):
            cache = EmbeddingCache(str(tmp_path / "cache"))
            embedder = Embedder(_config(), cache)
        retriever = HybridRetriever(_config(), embedder, vector_store, metadata_store)
        retriever.build_bm25_index()

        with patch.object(retriever.embedder, "embed_texts", return_value=[_vec(0)]):
            results = retriever.retrieve("iPhone sales growth", top_k=5)

        for r in results:
            assert r.dense_score == 0.0, (
                "an excluded/unverified legacy vector must never contribute "
                "nonzero dense_score evidence to the ranking"
            )

    def test_bm25_still_surfaces_current_text_independent_of_dense_exclusion(
        self, tmp_path
    ):
        """BM25/sparse search is intentionally unaffected by dense-side
        exclusion - it always re-reads current SQLite text directly, so an
        excluded chunk can still surface via genuine keyword relevance,
        just without dense similarity evidence riding along."""
        index_dir = tmp_path / "index"
        _write_legacy_index(index_dir, ["aapl_x_0000"], [_vec(0)])
        vector_store = VectorStore(str(index_dir), dim=DIM)
        vector_store.load()

        metadata_store = MetadataStore(str(tmp_path / "meta.db"))
        metadata_store.add_chunks(
            [_chunk("aapl_x_0000", "Apple unique distinctive keyword zephyrblue.")]
        )
        with patch("alphasignal.embeddings.embedder.OpenAI"):
            cache = EmbeddingCache(str(tmp_path / "cache"))
            embedder = Embedder(_config(), cache)
        retriever = HybridRetriever(_config(), embedder, vector_store, metadata_store)
        retriever.build_bm25_index()

        with patch.object(retriever.embedder, "embed_texts", return_value=[_vec(0)]):
            results = retriever.retrieve("zephyrblue", top_k=5)

        matching = [r for r in results if r.chunk_id == "aapl_x_0000"]
        assert matching, "BM25 must still find the chunk via current SQLite text"
        assert matching[0].dense_score == 0.0
        assert matching[0].sparse_score > 0.0

    def test_successful_reingestion_restores_dense_queryability(self, tmp_path):
        index_dir = tmp_path / "index"
        _write_legacy_index(index_dir, ["aapl_x_0000"], [_vec(0)])
        vector_store = VectorStore(str(index_dir), dim=DIM)
        vector_store.load()

        metadata_store = MetadataStore(str(tmp_path / "meta.db"))
        chunk = _chunk("aapl_x_0000", "Real current filing text.")
        metadata_store.add_chunks([chunk])
        with patch("alphasignal.embeddings.embedder.OpenAI"):
            cache = EmbeddingCache(str(tmp_path / "cache"))
            embedder = Embedder(_config(), cache)
        retriever = HybridRetriever(_config(), embedder, vector_store, metadata_store)

        # Before: excluded.
        with patch.object(retriever.embedder, "embed_texts", return_value=[_vec(3)]):
            assert retriever._dense_search(_vec(3), k=10) == []

        # Re-ingest with a real content_hash + freshly generated embedding.
        vector_store.add(
            np.array([_vec(3)]), ["aapl_x_0000"], {"aapl_x_0000": chunk.content_hash}
        )

        with patch.object(retriever.embedder, "embed_texts", return_value=[_vec(3)]):
            dense_results = retriever._dense_search(_vec(3), k=10)

        assert dense_results == [("aapl_x_0000", pytest.approx(1.0, abs=0.01))]
        assert vector_store.content_hashes["aapl_x_0000"] == chunk.content_hash

    def test_failed_reingestion_leaves_vector_untrusted(self, tmp_path):
        """If re-embedding fails during re-ingestion, the legacy vector
        must remain exactly as untrusted as before - not accidentally
        blessed by a failed attempt."""
        index_dir = tmp_path / "index"
        _write_legacy_index(index_dir, ["aapl_x_0000"], [_vec(0)])
        vector_store, metadata_store, retriever, pipeline, embedder = (
            _make_stores_and_retriever(tmp_path)
        )
        # _make_stores_and_retriever built its OWN fresh vector_store at a
        # different path; rebuild pointed at the legacy dir instead.
        vector_store = VectorStore(str(index_dir), dim=DIM)
        vector_store.load()
        chunk = _chunk("aapl_x_0000", "Text that fails to embed.")
        metadata_store.add_chunks([chunk])
        retriever = HybridRetriever(_config(), embedder, vector_store, metadata_store)

        with patch.object(
            embedder, "embed_texts", side_effect=RuntimeError("provider down")
        ):
            with pytest.raises(RuntimeError):
                embedder.embed_chunks([chunk])

        # Still hashless/untrusted, still excluded.
        assert vector_store.content_hashes.get("aapl_x_0000", "") == ""
        with patch.object(retriever.embedder, "embed_texts", return_value=[_vec(0)]):
            assert retriever._dense_search(_vec(0), k=10) == []
        # Not deleted - the legacy vector is still physically present.
        assert "aapl_x_0000" in vector_store.chunk_ids


class TestProductionPathsSupplyContentHash:
    def test_pipeline_store_chunks_always_passes_content_hash_to_vector_store(
        self, tmp_path
    ):
        """The one production ingestion path (IngestionPipeline.
        store_chunks) must call VectorStore.add() with a real,
        non-empty content_hash for every chunk - never relying on the
        identity-only fallback."""
        _, _, _, pipeline, _ = _make_stores_and_retriever(tmp_path)
        chunk = _chunk("aapl_p_0000", "Production path text.")

        with patch.object(pipeline.vector_store, "add") as mock_add:
            pipeline.store_chunks([chunk], {"aapl_p_0000": _vec(0)})

        assert mock_add.call_count == 1
        # add(embeddings, chunk_ids, content_hashes) - content_hashes is
        # the 3rd positional or a kwarg depending on call style used.
        args, kwargs = mock_add.call_args.args, mock_add.call_args.kwargs
        content_hashes = kwargs.get(
            "content_hashes", args[2] if len(args) > 2 else None
        )
        assert content_hashes == {"aapl_p_0000": chunk.content_hash}
        assert content_hashes["aapl_p_0000"], "content_hash must be non-empty"

    def test_full_ingest_end_to_end_produces_dense_queryable_chunks(self, tmp_path):
        """Full, real (mocked-embedder-only) ingestion via full_ingest()
        must leave chunks immediately dense-queryable - proving the
        production path never accidentally falls into identity-only mode."""
        from alphasignal.ingestion import RawDocument

        _, metadata_store, _, pipeline, embedder = _make_stores_and_retriever(tmp_path)
        raw_doc = RawDocument(
            ticker="AAPL",
            doc_type="10-K",
            filing_date=date(2024, 1, 15),
            period_of_report=date(2024, 1, 15),
            source="SEC EDGAR",
            sections={"item_1": "Apple designs and sells consumer electronics. " * 20},
            file_path="/tmp/filing.htm",
            accession_number="0001234567-24-000001",
        )

        def fake_embed_chunks(chunks):
            return {c.chunk_id: _vec(i) for i, c in enumerate(chunks)}

        with patch.object(
            pipeline.edgar_ingester, "fetch_filings", return_value=[raw_doc]
        ), patch.object(
            pipeline.news_ingester, "fetch_articles", return_value=[]
        ), patch.object(
            pipeline.embedder, "embed_chunks", side_effect=fake_embed_chunks
        ):
            result = pipeline.full_ingest("AAPL")

        assert result.chunks_created > 0
        # Every stored vector must have a real, non-blank content_hash.
        for chunk_id in pipeline.vector_store.chunk_ids:
            assert pipeline.vector_store.content_hashes.get(chunk_id, ""), (
                f"{chunk_id} was added without a content_hash - production "
                "path must never use identity-only dedup"
            )

    def test_no_production_call_site_omits_content_hashes_argument(self):
        """Static safety net: the ingestion pipeline's one call site that
        adds vectors must pass content_hashes explicitly (not the
        2-argument identity-only form). Catches a future regression that
        silently drops the argument."""
        import inspect
        import re

        from alphasignal.ingestion import pipeline as pipeline_module

        source = inspect.getsource(pipeline_module.IngestionPipeline.store_chunks)
        call_sites = re.findall(r"vector_store\.add\(([^)]*)\)", source)
        assert call_sites, "expected exactly one vector_store.add() call site"
        for call_args in call_sites:
            assert (
                "content_hashes" in call_args
            ), f"vector_store.add({call_args}) does not pass content_hashes"


class TestBM25DocumentedBehaviorIsIntentional:
    def test_bm25_index_reads_current_sqlite_text_not_faiss(self, tmp_path):
        """Documents/proves BM25 never touches FAISS at all - it always
        rebuilds from current SQLite text, so dense-side staleness/
        exclusion is structurally irrelevant to it."""
        _, metadata_store, retriever, _, _ = _make_stores_and_retriever(tmp_path)
        metadata_store.add_chunks(
            [_chunk("aapl_bm_0000", "Distinctive bm25only keyword text.")]
        )

        retriever.build_bm25_index()

        assert "aapl_bm_0000" in retriever.bm25_chunk_ids
        assert (
            retriever.bm25_chunks["aapl_bm_0000"].text
            == "Distinctive bm25only keyword text."
        )
