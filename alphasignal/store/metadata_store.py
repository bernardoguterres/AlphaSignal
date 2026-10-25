"""SQLite metadata store module."""

import logging
from datetime import date as date_type
from datetime import datetime
from pathlib import Path

from sqlalchemy import inspect, text
from sqlmodel import Field, Session, SQLModel, create_engine, func, select

from alphasignal.ingestion import Chunk

logger = logging.getLogger(__name__)


class ChunkRecord(SQLModel, table=True):
    """SQLModel representing a chunk in the database."""

    __tablename__ = "chunks"  # type: ignore

    chunk_id: str = Field(primary_key=True)
    ticker: str = Field(index=True)
    text: str
    token_count: int
    doc_type: str = Field(index=True)
    source: str
    section: str | None = None
    date: date_type = Field(index=True)
    url: str | None = None
    chunk_index: int
    total_chunks: int
    content_hash: str = ""
    # Exact source-document/article ownership identity (see Chunk.source_id).
    # Indexed so orphan-cleanup lookups (get_chunk_ids_by_source_id) are an
    # exact-equality query, not a string-prefix/LIKE match against chunk_id -
    # eliminating wildcard-escaping/collision concerns as an attack surface
    # entirely rather than merely mitigating them (audit correction,
    # 2026-09-11: "prefer exact persisted source ownership fields over raw
    # string-prefix deletion where practical").
    source_id: str = Field(default="", index=True)
    accession_number: str | None = None
    attribution: str = ""
    period_end: date_type | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MetadataStore:
    """Manages SQLite database for chunk metadata."""

    def __init__(self, db_path: str):
        """Initialize metadata store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create engine
        db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(db_url, echo=False)

        # Idempotent schema upgrade FIRST: create_all() below only creates
        # tables that don't exist yet - it never adds a missing column to
        # an existing table (audit finding, 2026-09-11 compatibility pass:
        # an existing database from before the source_id column existed
        # would otherwise raise OperationalError on the first real query,
        # not fail at startup where it'd be noticed immediately).
        self._upgrade_schema()

        # Create tables (no-op for an existing, now-upgraded "chunks" table;
        # builds the full current schema for a brand-new database)
        SQLModel.metadata.create_all(self.engine)

        logger.info(f"Initialized metadata store at {db_path}")

    # Columns added after the original schema, with their SQL definitions.
    # Applied idempotently to any pre-existing `chunks` table.
    _ADDED_COLUMNS = {
        "content_hash": "VARCHAR NOT NULL DEFAULT ''",
        "source_id": "VARCHAR NOT NULL DEFAULT ''",
        "accession_number": "VARCHAR",
        "attribution": "VARCHAR NOT NULL DEFAULT ''",
        "period_end": "DATE",
    }

    def _upgrade_schema(self):
        """Add every missing post-v1 column to a pre-existing `chunks`
        table, idempotently, then backfill `content_hash` exactly.

        2026-10: this previously added only `source_id`. A database from
        before `content_hash` existed (the real ~42k-chunk corpus) therefore
        failed EVERY chunk query with "no such column: chunks.content_hash",
        so /query and /sentiment errored on the real data while every test
        (built on fresh databases) passed.

        content_hash is backfilled as sha256(text) - an exact property of
        the stored text, not a guess. source_id / accession_number /
        attribution are left empty (unknown) here; the provenance
        migration script fills them only where it can prove the value.
        """
        import hashlib

        inspector = inspect(self.engine)
        if "chunks" not in inspector.get_table_names():
            return
        existing = {col["name"] for col in inspector.get_columns("chunks")}
        missing = [c for c in self._ADDED_COLUMNS if c not in existing]
        try:
            with self.engine.begin() as conn:
                for col in missing:
                    logger.warning(
                        f"Upgrading 'chunks' table at {self.db_path}: adding column "
                        f"'{col}' (idempotent, one-time)"
                    )
                    conn.execute(
                        text(
                            f"ALTER TABLE chunks ADD COLUMN {col} {self._ADDED_COLUMNS[col]}"
                        )
                    )
                rows = conn.execute(
                    text("SELECT chunk_id, text FROM chunks WHERE content_hash = ''")
                ).fetchall()
                if rows:
                    logger.warning(f"Backfilling content_hash for {len(rows)} chunks")
                    conn.execute(
                        text("UPDATE chunks SET content_hash = :h WHERE chunk_id = :c"),
                        [
                            {
                                "h": hashlib.sha256(t.encode("utf-8")).hexdigest(),
                                "c": cid,
                            }
                            for cid, t in rows
                        ],
                    )
        except Exception as e:
            raise RuntimeError(
                f"Failed to upgrade 'chunks' table at {self.db_path}: {e}. This "
                "MetadataStore cannot start against a schema its own queries "
                "require - fix the underlying issue (e.g. file permissions, disk "
                "space) and retry."
            ) from e

    def add_chunks(self, chunks: list[Chunk]):
        """Add chunks to the database.

        Args:
            chunks: List of Chunk objects to add
        """
        if not chunks:
            return

        with Session(self.engine) as session:
            for chunk in chunks:
                # Create ChunkRecord
                record = ChunkRecord(
                    chunk_id=chunk.chunk_id,
                    ticker=chunk.ticker,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    doc_type=chunk.doc_type,
                    source=chunk.source,
                    section=chunk.section,
                    date=chunk.date,
                    url=chunk.url,
                    chunk_index=chunk.chunk_index,
                    total_chunks=chunk.total_chunks,
                    content_hash=chunk.content_hash,
                    source_id=chunk.source_id,
                    accession_number=chunk.accession_number,
                    attribution=chunk.attribution,
                    period_end=chunk.period_end,
                )

                # Merge (insert or update on conflict) - this already
                # overwrites `text`/`content_hash` in place when the same
                # chunk_id is re-ingested with changed content, so SQLite
                # never goes stale here; the stale-data risk was isolated to
                # the embedding cache/FAISS layer (see VectorStore.add).
                session.merge(record)

            session.commit()

        logger.info(f"Added {len(chunks)} chunks to metadata store")

    @staticmethod
    def _to_chunk(record: ChunkRecord) -> Chunk:
        """Convert a ChunkRecord (DB row) into the plain Chunk domain object."""
        return Chunk(
            chunk_id=record.chunk_id,
            ticker=record.ticker,
            text=record.text,
            token_count=record.token_count,
            doc_type=record.doc_type,
            source=record.source,
            section=record.section,
            date=record.date,
            url=record.url,
            chunk_index=record.chunk_index,
            total_chunks=record.total_chunks,
            content_hash=record.content_hash,
            source_id=record.source_id,
            accession_number=record.accession_number,
            attribution=record.attribution or "",
            period_end=record.period_end,
        )

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Retrieve a chunk by ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Chunk object or None if not found
        """
        with Session(self.engine) as session:
            record = session.get(ChunkRecord, chunk_id)

            if not record:
                return None

            return self._to_chunk(record)

    def get_chunks_by_ticker(
        self, ticker: str, doc_type: str | None = None
    ) -> list[Chunk]:
        """Get all chunks for a ticker.

        Args:
            ticker: Ticker symbol
            doc_type: Optional document type filter

        Returns:
            List of Chunk objects
        """
        with Session(self.engine) as session:
            statement = select(ChunkRecord).where(ChunkRecord.ticker == ticker)

            if doc_type:
                statement = statement.where(ChunkRecord.doc_type == doc_type)

            records = session.exec(statement).all()

            return [self._to_chunk(r) for r in records]

    def version_info(self) -> dict:
        """content_digest / corpus_version of the stored chunks (see
        alphasignal.store.versioning) plus counts per chunker version."""
        from alphasignal.store.versioning import (
            chunker_versions,
            content_digest,
            corpus_digest,
        )

        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT chunk_id, content_hash, source_id, accession_number, "
                    "period_end, date, url, ticker, doc_type, section FROM chunks"
                )
            ).fetchall()
        return {
            "corpus_version": corpus_digest(tuple(r) for r in rows),
            "content_digest": content_digest((r[0], r[1]) for r in rows),
            "chunks": len(rows),
            "chunker_versions": chunker_versions(r[0] for r in rows),
        }

    def get_filing_periods(
        self, ticker: str, with_section: bool = False
    ) -> list[tuple]:
        """(chunk_id, doc_type, period_end[, section]) for every filing chunk
        of a ticker whose reporting period is known."""
        with Session(self.engine) as session:
            rows = session.exec(
                select(
                    ChunkRecord.chunk_id,
                    ChunkRecord.doc_type,
                    ChunkRecord.period_end,
                    ChunkRecord.section,
                ).where(
                    ChunkRecord.ticker == ticker,
                    ChunkRecord.doc_type.in_(["10-K", "10-Q"]),
                    ChunkRecord.period_end.is_not(None),
                )
            ).all()
            if with_section:
                return [tuple(r) for r in rows]
            return [(cid, dt, pe) for cid, dt, pe, _ in rows]

    def get_chunks_by_date_range(
        self, start: date_type, end: date_type, ticker: str | None = None
    ) -> list[Chunk]:
        """Get chunks within a date range.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)
            ticker: Optional ticker filter

        Returns:
            List of Chunk objects
        """
        with Session(self.engine) as session:
            statement = select(ChunkRecord).where(
                ChunkRecord.date >= start, ChunkRecord.date <= end
            )

            if ticker:
                statement = statement.where(ChunkRecord.ticker == ticker)

            records = session.exec(statement).all()

            return [self._to_chunk(r) for r in records]

    def get_chunk_ids_by_source_id(self, source_id: str) -> list[str]:
        """Get all chunk_ids belonging to a source document/article.

        Exact-equality lookup against the persisted `source_id` column -
        preferred over string-prefix matching on chunk_id, since it can
        never be ambiguous regardless of what characters appear in
        tickers, hashes, or IDs (no wildcard/escaping semantics involved
        at all). Used to find a source's *previously* stored chunk_ids so
        re-ingestion can detect and remove ones the new chunking run no
        longer produced (orphans left behind when a document shrinks, or
        its chunk boundaries shift) - see IngestionPipeline.store_chunks.

        Args:
            source_id: Exact source-document/article identity, e.g.
                "aapl_10k_a1b2c3d4" (see Chunk.source_id).

        Returns:
            List of matching chunk_ids currently in the store. A blank
            source_id always returns [] - it means "unknown source," never
            "match every row with a blank source_id."
        """
        if not source_id:
            return []
        with Session(self.engine) as session:
            statement = select(ChunkRecord.chunk_id).where(
                ChunkRecord.source_id == source_id
            )
            return list(session.exec(statement).all())

    def get_chunk_ids_with_prefix(self, prefix: str) -> list[str]:
        """Get all chunk_ids sharing a source-document chunk_id prefix.

        Legacy/fallback path: prefer get_chunk_ids_by_source_id() (exact
        equality against the persisted source_id column) wherever the
        source_id is known. This LIKE-based method exists for chunk_id
        prefixes whose rows predate the source_id column (source_id==""
        on those rows) - it is correctly escaped and delimited (a literal,
        escaped underscore boundary is required right after the prefix),
        so "document:1" cannot match "document:10_0000", but a real
        persisted ownership field is still the safer default.

        Args:
            prefix: Source-document chunk_id prefix, e.g. "aapl_10k_a1b2c3d4"

        Returns:
            List of matching chunk_ids currently in the store
        """
        # Escape LIKE wildcards ('_' and '%') in the prefix itself - chunk_id
        # prefixes routinely contain literal underscores (e.g.
        # "aapl_10k_a1b2c3d4"), which must not be treated as
        # match-any-character wildcards.
        escaped = prefix.replace("\\", "\\\\").replace("_", "\\_").replace("%", "\\%")
        with Session(self.engine) as session:
            statement = select(ChunkRecord.chunk_id).where(
                ChunkRecord.chunk_id.like(f"{escaped}\\_%", escape="\\")
            )
            return list(session.exec(statement).all())

    def delete_chunks(self, chunk_ids: list[str]):
        """Delete chunks by ID.

        Args:
            chunk_ids: Chunk identifiers to remove
        """
        if not chunk_ids:
            return

        with Session(self.engine) as session:
            for chunk_id in chunk_ids:
                record = session.get(ChunkRecord, chunk_id)
                if record is not None:
                    session.delete(record)
            session.commit()

        logger.info(f"Deleted {len(chunk_ids)} chunk(s) from metadata store")

    def get_all_chunk_ids(self) -> list[str]:
        """Get all chunk IDs.

        Returns:
            List of chunk IDs
        """
        with Session(self.engine) as session:
            statement = select(ChunkRecord.chunk_id)
            return list(session.exec(statement).all())

    def get_content_hashes(self) -> dict[str, str]:
        """Return {chunk_id: content_hash} for every chunk."""
        with Session(self.engine) as session:
            rows = session.exec(select(ChunkRecord.chunk_id, ChunkRecord.content_hash))
            return {cid: h for cid, h in rows.all()}

    def count(self) -> int:
        """Get total number of chunks.

        Returns:
            Number of chunks in the database
        """
        with Session(self.engine) as session:
            statement = select(func.count()).select_from(ChunkRecord)
            return session.exec(statement).one()
