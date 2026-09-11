"""MetadataStore schema-compatibility tests (2026-09-11 follow-up
correction pass, objective 2).

Proves that an existing SQLite database created before the ChunkRecord.
source_id column existed can start successfully against the current
MetadataStore, gets its schema upgraded exactly once and idempotently, and
never loses or guesses at existing data. Uses temporary, synthetic
databases only.
"""

import datetime
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import inspect

from alphasignal.ingestion import Chunk
from alphasignal.store.metadata_store import MetadataStore


def _create_pre_source_id_db(db_path: Path, rows: list[dict]):
    """Build a database matching the schema from before source_id existed,
    via raw sqlite3 - never via a competing SQLModel class (which would
    collide with MetadataStore's own ChunkRecord in the same process's
    shared SQLModel metadata registry)."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE chunks (
            chunk_id VARCHAR NOT NULL PRIMARY KEY,
            ticker VARCHAR NOT NULL,
            text VARCHAR NOT NULL,
            token_count INTEGER NOT NULL,
            doc_type VARCHAR NOT NULL,
            source VARCHAR NOT NULL,
            section VARCHAR,
            date DATE NOT NULL,
            url VARCHAR,
            chunk_index INTEGER NOT NULL,
            total_chunks INTEGER NOT NULL,
            content_hash VARCHAR NOT NULL DEFAULT '',
            created_at DATETIME NOT NULL
        )
        """)
    conn.execute("CREATE INDEX ix_chunks_ticker ON chunks (ticker)")
    conn.execute("CREATE INDEX ix_chunks_doc_type ON chunks (doc_type)")
    conn.execute("CREATE INDEX ix_chunks_date ON chunks (date)")
    for row in rows:
        conn.execute(
            "INSERT INTO chunks (chunk_id, ticker, text, token_count, doc_type, "
            "source, section, date, url, chunk_index, total_chunks, "
            "content_hash, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                row["chunk_id"],
                row["ticker"],
                row["text"],
                row.get("token_count", 2),
                row.get("doc_type", "10-K"),
                row.get("source", "SEC EDGAR"),
                row.get("section", "item_1"),
                row.get("date", "2024-01-01"),
                row.get("url"),
                row.get("chunk_index", 0),
                row.get("total_chunks", 1),
                row.get("content_hash", ""),
                "2024-01-01 00:00:00",
            ),
        )
    conn.commit()
    conn.close()


def _new_chunk(chunk_id, source_id="", text="new text"):
    return Chunk(
        chunk_id=chunk_id,
        ticker="AAPL",
        text=text,
        token_count=2,
        doc_type="10-K",
        source="SEC EDGAR",
        section="item_1",
        date=datetime.date(2024, 2, 1),
        url=None,
        chunk_index=0,
        total_chunks=1,
        source_id=source_id,
    )


class TestFreshDatabaseCreation:
    def test_fresh_database_has_source_id_column_from_the_start(self, tmp_path):
        store = MetadataStore(str(tmp_path / "fresh.db"))
        inspector = inspect(store.engine)
        columns = {col["name"] for col in inspector.get_columns("chunks")}
        assert "source_id" in columns

    def test_fresh_database_new_rows_persist_source_id(self, tmp_path):
        store = MetadataStore(str(tmp_path / "fresh.db"))
        store.add_chunks([_new_chunk("a_0000", source_id="aapl_10k_abc123")])
        assert store.get_chunk("a_0000").source_id == "aapl_10k_abc123"


class TestUpgradeFromPreviousSchema:
    def test_old_database_starts_successfully(self, tmp_path):
        db_path = tmp_path / "old.db"
        _create_pre_source_id_db(
            db_path, [{"chunk_id": "old_0000", "ticker": "AAPL", "text": "old text"}]
        )

        store = MetadataStore(str(db_path))  # must not raise

        inspector = inspect(store.engine)
        columns = {col["name"] for col in inspector.get_columns("chunks")}
        assert "source_id" in columns

    def test_old_rows_remain_readable_after_upgrade(self, tmp_path):
        db_path = tmp_path / "old.db"
        _create_pre_source_id_db(
            db_path,
            [
                {
                    "chunk_id": "old_0000",
                    "ticker": "AAPL",
                    "text": "Original pre-migration text.",
                    "content_hash": "somehash",
                }
            ],
        )

        store = MetadataStore(str(db_path))
        chunk = store.get_chunk("old_0000")

        assert chunk is not None
        assert chunk.text == "Original pre-migration text."
        assert chunk.content_hash == "somehash"
        # Unknown ownership - never guessed from chunk_id or anything else.
        assert chunk.source_id == ""

    def test_upgrade_does_not_change_row_count(self, tmp_path):
        db_path = tmp_path / "old.db"
        _create_pre_source_id_db(
            db_path,
            [
                {"chunk_id": f"old_{i:04d}", "ticker": "AAPL", "text": f"text {i}"}
                for i in range(5)
            ],
        )

        store = MetadataStore(str(db_path))
        assert store.count() == 5

    def test_repeated_startup_is_idempotent(self, tmp_path):
        db_path = tmp_path / "old.db"
        _create_pre_source_id_db(
            db_path, [{"chunk_id": "old_0000", "ticker": "AAPL", "text": "text"}]
        )

        store1 = MetadataStore(str(db_path))
        assert store1.count() == 1

        # Second (and third) startup against the now-upgraded database must
        # not raise, duplicate the column, or lose data.
        store2 = MetadataStore(str(db_path))
        store3 = MetadataStore(str(db_path))

        assert store2.count() == 1
        assert store3.count() == 1
        inspector = inspect(store3.engine)
        columns = [col["name"] for col in inspector.get_columns("chunks")]
        assert columns.count("source_id") == 1

    def test_new_rows_after_upgrade_persist_source_id_alongside_old_rows(
        self, tmp_path
    ):
        db_path = tmp_path / "old.db"
        _create_pre_source_id_db(
            db_path, [{"chunk_id": "old_0000", "ticker": "AAPL", "text": "old"}]
        )

        store = MetadataStore(str(db_path))
        store.add_chunks([_new_chunk("new_0000", source_id="aapl_10k_xyz789")])

        old_chunk = store.get_chunk("old_0000")
        new_chunk = store.get_chunk("new_0000")
        assert old_chunk.source_id == ""
        assert new_chunk.source_id == "aapl_10k_xyz789"
        assert store.count() == 2

    def test_mixed_legacy_and_new_records_cleanup_isolation(self, tmp_path):
        """A cleanup targeting a new record's exact source_id must not
        touch legacy (source_id-less) rows, and vice versa via the
        LIKE-prefix fallback."""
        db_path = tmp_path / "old.db"
        _create_pre_source_id_db(
            db_path,
            [{"chunk_id": "legacy_prefix_0000", "ticker": "AAPL", "text": "legacy"}],
        )
        store = MetadataStore(str(db_path))
        store.add_chunks([_new_chunk("new_0000", source_id="new_prefix")])

        # Exact-match cleanup for the new record's source finds only it.
        assert store.get_chunk_ids_by_source_id("new_prefix") == ["new_0000"]
        # The legacy row (blank source_id) is invisible to exact-match
        # lookups for any real source_id - it must be found via the
        # chunk_id-prefix fallback instead.
        assert store.get_chunk_ids_by_source_id("legacy_prefix") == []
        assert store.get_chunk_ids_with_prefix("legacy_prefix") == [
            "legacy_prefix_0000"
        ]

        store.delete_chunks(store.get_chunk_ids_by_source_id("new_prefix"))
        assert store.get_chunk("new_0000") is None
        assert store.get_chunk("legacy_prefix_0000") is not None

    def test_document_1_vs_document_10_legacy_rows_not_confused(self, tmp_path):
        db_path = tmp_path / "old.db"
        _create_pre_source_id_db(
            db_path,
            [
                {"chunk_id": "document:1_0000", "ticker": "AAPL", "text": "one"},
                {"chunk_id": "document:10_0000", "ticker": "AAPL", "text": "ten"},
            ],
        )
        store = MetadataStore(str(db_path))

        assert store.get_chunk_ids_with_prefix("document:1") == ["document:1_0000"]
        assert store.get_chunk_ids_with_prefix("document:10") == ["document:10_0000"]

    def test_literal_percent_and_underscore_in_legacy_chunk_ids(self, tmp_path):
        db_path = tmp_path / "old.db"
        _create_pre_source_id_db(
            db_path,
            [
                {"chunk_id": "weird%prefix_0000", "ticker": "AAPL", "text": "a"},
                {"chunk_id": "aapl_10k_ab12cd34_0000", "ticker": "AAPL", "text": "b"},
            ],
        )
        store = MetadataStore(str(db_path))

        assert store.get_chunk_ids_with_prefix("weird%prefix") == ["weird%prefix_0000"]
        assert store.get_chunk_ids_with_prefix("aapl_10k_ab12cd34") == [
            "aapl_10k_ab12cd34_0000"
        ]


class TestUpgradeFailureIsActionable:
    def test_failed_alter_table_raises_actionable_error_not_silent_corruption(
        self, tmp_path
    ):
        db_path = tmp_path / "old.db"
        _create_pre_source_id_db(
            db_path, [{"chunk_id": "old_0000", "ticker": "AAPL", "text": "text"}]
        )

        with patch(
            "alphasignal.store.metadata_store.text",
            side_effect=RuntimeError("simulated ALTER TABLE failure"),
        ):
            with pytest.raises(RuntimeError, match="Failed to upgrade"):
                MetadataStore(str(db_path))

        # The database itself must remain exactly as it was - readable via
        # a direct connection, no partial/corrupt state.
        conn = sqlite3.connect(str(db_path))
        cols = [row[1] for row in conn.execute("PRAGMA table_info(chunks)")]
        conn.close()
        assert "source_id" not in cols
        assert "chunk_id" in cols  # table itself intact
