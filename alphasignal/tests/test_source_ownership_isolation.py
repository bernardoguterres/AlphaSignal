"""Source-ownership cleanup isolation tests (2026-09-11 correction pass,
objective 4).

Proves that orphan cleanup for one source document/article cannot delete
or touch another source's rows/vectors/cache entries, even when the two
sources have deliberately adversarial, overlapping, or wildcard-laden
chunk_ids. Covers both the exact source_id equality path (preferred) and
the legacy escaped-LIKE prefix path (fallback for source_id-less rows).
"""

from datetime import date

import numpy as np

from alphasignal.ingestion import Chunk
from alphasignal.store.metadata_store import MetadataStore
from alphasignal.store.vector_store import VectorStore

DIM = 8


def _vec(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i % DIM] = 1.0
    return v


def _chunk(chunk_id, source_id, text="text", chunk_index=0, total_chunks=1):
    return Chunk(
        chunk_id=chunk_id,
        ticker="AAPL",
        text=text,
        token_count=2,
        doc_type="10-K",
        source="SEC EDGAR",
        section="item_1",
        date=date(2024, 1, 1),
        url=None,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        source_id=source_id,
    )


class TestExactSourceIdIsolation:
    """The preferred path: get_chunk_ids_by_source_id (exact equality)."""

    def test_shared_textual_prefix_does_not_cross_delete(self, tmp_path):
        store = MetadataStore(str(tmp_path / "meta.db"))
        store.add_chunks(
            [
                _chunk("doc_0000", source_id="document"),
                _chunk("docextra_0000", source_id="documentextra"),
            ]
        )

        found = store.get_chunk_ids_by_source_id("document")
        assert found == ["doc_0000"]

    def test_numeric_suffix_ids_are_not_confused(self, tmp_path):
        """document:1 vs document:10 - textually one is a prefix of the
        other, but exact-equality ownership can never confuse them."""
        store = MetadataStore(str(tmp_path / "meta.db"))
        store.add_chunks(
            [
                _chunk("a_0000", source_id="document:1"),
                _chunk("b_0000", source_id="document:10"),
            ]
        )

        assert store.get_chunk_ids_by_source_id("document:1") == ["a_0000"]
        assert store.get_chunk_ids_by_source_id("document:10") == ["b_0000"]

    def test_wildcard_characters_in_source_id_are_literal(self, tmp_path):
        store = MetadataStore(str(tmp_path / "meta.db"))
        store.add_chunks(
            [
                _chunk("a_0000", source_id="doc_%_weird"),
                _chunk(
                    "b_0000", source_id="doc_x_weird"
                ),  # would match doc_%_weird as a SQL LIKE pattern
            ]
        )

        assert store.get_chunk_ids_by_source_id("doc_%_weird") == ["a_0000"]

    def test_blank_source_id_matches_nothing(self, tmp_path):
        store = MetadataStore(str(tmp_path / "meta.db"))
        store.add_chunks([_chunk("a_0000", source_id="")])
        assert store.get_chunk_ids_by_source_id("") == []

    def test_different_source_types_never_cross(self, tmp_path):
        store = MetadataStore(str(tmp_path / "meta.db"))
        store.add_chunks(
            [
                _chunk("aapl_10k_abc12345_0000", source_id="aapl_10k_abc12345"),
                _chunk("aapl_news_abc12345_0000", source_id="aapl_news_abc12345"),
            ]
        )

        assert store.get_chunk_ids_by_source_id("aapl_10k_abc12345") == [
            "aapl_10k_abc12345_0000"
        ]
        assert store.get_chunk_ids_by_source_id("aapl_news_abc12345") == [
            "aapl_news_abc12345_0000"
        ]

    def test_delete_chunks_for_one_source_leaves_others_untouched(self, tmp_path):
        meta_store = MetadataStore(str(tmp_path / "meta.db"))
        vector_store = VectorStore(str(tmp_path / "index"), dim=DIM)
        vector_store.load()

        chunk_a = _chunk("a_0000", source_id="source_a")
        chunk_b = _chunk("b_0000", source_id="source_b")
        meta_store.add_chunks([chunk_a, chunk_b])
        vector_store.add(
            np.array([_vec(0), _vec(1)]),
            ["a_0000", "b_0000"],
            {"a_0000": chunk_a.content_hash, "b_0000": chunk_b.content_hash},
        )

        # Simulate orphan cleanup for source_a only (as store_chunks would
        # do when source_a's re-chunking no longer produces "a_0000").
        orphans = meta_store.get_chunk_ids_by_source_id("source_a")
        meta_store.delete_chunks(orphans)
        vector_store.remove(orphans)

        assert meta_store.get_chunk("a_0000") is None
        assert meta_store.get_chunk("b_0000") is not None
        assert "a_0000" not in vector_store.chunk_ids
        assert "b_0000" in vector_store.chunk_ids
        assert len(vector_store) == 1


class TestLegacyPrefixIsolation:
    """Fallback path: get_chunk_ids_with_prefix (escaped LIKE) for rows
    with no source_id at all."""

    def test_document_1_vs_document_10_via_like_prefix(self, tmp_path):
        store = MetadataStore(str(tmp_path / "meta.db"))
        # No source_id set - forces the legacy chunk_id-prefix path.
        store.add_chunks(
            [
                _chunk("document:1_0000", source_id=""),
                _chunk("document:10_0000", source_id=""),
            ]
        )

        found = store.get_chunk_ids_with_prefix("document:1")
        assert found == ["document:1_0000"]

    def test_sql_wildcard_percent_in_prefix_is_literal(self, tmp_path):
        store = MetadataStore(str(tmp_path / "meta.db"))
        store.add_chunks(
            [
                _chunk("weird%prefix_0000", source_id=""),
                _chunk("weirdXprefix_0000", source_id=""),
            ]
        )

        found = store.get_chunk_ids_with_prefix("weird%prefix")
        assert found == ["weird%prefix_0000"]

    def test_sql_wildcard_underscore_in_prefix_is_literal(self, tmp_path):
        store = MetadataStore(str(tmp_path / "meta.db"))
        store.add_chunks(
            [
                _chunk("aapl_10k_ab12cd34_0000", source_id=""),
                _chunk(
                    "aaplx10kxab12cd34_0000", source_id=""
                ),  # would match if _ were a wildcard
            ]
        )

        found = store.get_chunk_ids_with_prefix("aapl_10k_ab12cd34")
        assert found == ["aapl_10k_ab12cd34_0000"]

    def test_similar_ticker_names_do_not_cross(self, tmp_path):
        store = MetadataStore(str(tmp_path / "meta.db"))
        store.add_chunks(
            [
                _chunk("aapl_10k_abc12345_0000", source_id=""),
                _chunk("aaplx_10k_def67890_0000", source_id=""),
            ]
        )

        found = store.get_chunk_ids_with_prefix("aapl_10k_abc12345")
        assert found == ["aapl_10k_abc12345_0000"]


class TestPipelineOrphanCleanupIsolation:
    """End-to-end: IngestionPipeline.store_chunks must only ever clean up
    the source(s) actually being re-ingested this call, never an unrelated
    one that happens to share a lexical prefix."""

    def test_reingesting_one_source_never_touches_another_with_shared_prefix(
        self, tmp_path
    ):
        from unittest.mock import patch

        from alphasignal.embeddings.cache import EmbeddingCache
        from alphasignal.embeddings.embedder import Embedder
        from alphasignal.ingestion.pipeline import IngestionPipeline

        config = {
            "chunking": {
                "target_tokens": 300,
                "min_tokens": 5,
                "max_tokens": 400,
                "overlap_tokens": 10,
            },
            "embeddings": {
                "model": "text-embedding-ada-002",
                "batch_size": 100,
                "max_retries": 3,
                "retry_delay": 1.0,
            },
        }
        vector_store = VectorStore(str(tmp_path / "index"), dim=DIM)
        vector_store.load()
        metadata_store = MetadataStore(str(tmp_path / "meta.db"))
        with patch("alphasignal.embeddings.embedder.OpenAI"):
            cache = EmbeddingCache(str(tmp_path / "cache"))
            embedder = Embedder(config, cache)
        pipeline = IngestionPipeline(
            config,
            embedder=embedder,
            vector_store=vector_store,
            metadata_store=metadata_store,
        )

        # Two chunks from deliberately adversarial, textually-overlapping
        # source_ids ("document:1" is a string-prefix of "document:10").
        chunk_1 = _chunk("x_0000", source_id="document:1", text="Source one text.")
        chunk_10 = _chunk("y_0000", source_id="document:10", text="Source ten text.")
        pipeline.store_chunks(
            [chunk_1, chunk_10],
            {"x_0000": _vec(0), "y_0000": _vec(1)},
        )
        assert metadata_store.count() == 2

        # Re-ingest ONLY "document:1" with zero chunks now (source emptied) -
        # "document:10"'s chunk must survive untouched.
        pipeline.store_chunks([], {}, source_prefixes={"document:1"})

        assert metadata_store.get_chunk("x_0000") is None
        assert metadata_store.get_chunk("y_0000") is not None
        assert "x_0000" not in vector_store.chunk_ids
        assert "y_0000" in vector_store.chunk_ids
