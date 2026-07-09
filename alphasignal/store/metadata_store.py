"""SQLite metadata store module."""

import logging
from datetime import date as date_type
from datetime import datetime
from pathlib import Path

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

        # Create tables
        SQLModel.metadata.create_all(self.engine)

        logger.info(f"Initialized metadata store at {db_path}")

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
                    total_chunks=chunk.total_chunks
                )

                # Merge (insert or update on conflict)
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
            total_chunks=record.total_chunks
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
        self,
        ticker: str,
        doc_type: str | None = None
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

    def get_chunks_by_date_range(
        self,
        start: date_type,
        end: date_type,
        ticker: str | None = None
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
                ChunkRecord.date >= start,
                ChunkRecord.date <= end
            )

            if ticker:
                statement = statement.where(ChunkRecord.ticker == ticker)

            records = session.exec(statement).all()

            return [self._to_chunk(r) for r in records]

    def get_all_chunk_ids(self) -> list[str]:
        """Get all chunk IDs.

        Returns:
            List of chunk IDs
        """
        with Session(self.engine) as session:
            statement = select(ChunkRecord.chunk_id)
            return list(session.exec(statement).all())

    def count(self) -> int:
        """Get total number of chunks.

        Returns:
            Number of chunks in the database
        """
        with Session(self.engine) as session:
            statement = select(func.count()).select_from(ChunkRecord)
            return session.exec(statement).one()
