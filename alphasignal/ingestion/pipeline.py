"""Ingestion pipeline orchestration."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alphasignal.embeddings.cache import EmbeddingCache
from alphasignal.embeddings.embedder import Embedder
from alphasignal.ingestion import Chunk, RawArticle, RawDocument
from alphasignal.ingestion.chunker import SemanticChunker
from alphasignal.ingestion.edgar import EDGARIngester
from alphasignal.ingestion.news import NewsIngester
from alphasignal.store.metadata_store import MetadataStore
from alphasignal.store.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Result of full ingestion pipeline."""

    ticker: str
    chunks_created: int
    chunks_embedded: int
    chunks_stored: int


class IngestionPipeline:
    """Orchestrates the full ingestion pipeline."""

    def __init__(
        self,
        config: dict,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        metadata_store: MetadataStore | None = None,
    ):
        """Initialize ingestion pipeline.

        Args:
            config: Configuration dictionary
            embedder: Shared Embedder instance. If omitted, a new one is
                built from config (standalone/script usage). When run behind
                the API, pass the same instance used by HybridRetriever so
                the embedding cache is shared.
            vector_store: Shared VectorStore instance. If omitted, a new one
                is built from config. When run behind the API, pass the same
                instance used by HybridRetriever so newly ingested vectors
                are immediately visible to /query without a process restart.
            metadata_store: Shared MetadataStore instance. If omitted, a new
                one is built from config.
        """
        self.config = config

        # Set up data directories
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

        # Initialize EDGAR ingester
        self.edgar_ingester = EDGARIngester(
            config=config, download_dir=str(self.data_dir / "edgar_raw")
        )

        # Initialize news ingester
        self.news_ingester = NewsIngester(config=config)

        # Initialize semantic chunker
        self.chunker = SemanticChunker(config=config)

        storage_config = config.get("storage", {})

        # Initialize embedding cache and embedder (unless shared instance provided)
        if embedder is not None:
            self.embedder = embedder
        else:
            cache_path = storage_config.get(
                "embeddings_cache_path", "data/embeddings_cache"
            )
            self.embedding_cache = EmbeddingCache(f"{cache_path}/cache.npy")
            self.embedder = Embedder(config=config, cache=self.embedding_cache)

        # Initialize metadata store (unless shared instance provided)
        if metadata_store is not None:
            self.metadata_store = metadata_store
        else:
            db_path = storage_config.get("sqlite_db_path", "data/metadata.db")
            self.metadata_store = MetadataStore(db_path)

        # Initialize vector store (unless shared instance provided)
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            index_path = storage_config.get("faiss_index_path", "data/faiss_index")
            self.vector_store = VectorStore(index_path)
            self.vector_store.load()

    def ingest_ticker_edgar(
        self,
        ticker: str,
        filing_types: list[str] | None = None,
        years_back: int | None = None,
    ) -> list[RawDocument]:
        """Ingest SEC EDGAR filings for a ticker.

        Args:
            ticker: Stock ticker symbol
            filing_types: Filing types to fetch. Defaults to config value.
            years_back: Years of historical filings to fetch. Defaults to config value.

        Returns:
            List of RawDocument objects
        """
        logger.info(f"Starting EDGAR ingestion for {ticker}")

        # Get filing parameters from config, unless overridden by caller
        edgar_config = self.config.get("ingestion", {}).get("edgar", {})
        if filing_types is None:
            filing_types = edgar_config.get("filing_types", ["10-K", "10-Q"])
        if years_back is None:
            years_back = edgar_config.get("years_back", 2)

        # Fetch filings
        raw_docs = self.edgar_ingester.fetch_filings(
            ticker=ticker, filing_types=filing_types, years_back=years_back
        )

        logger.info(
            f"Completed EDGAR ingestion for {ticker}: {len(raw_docs)} documents"
        )

        return raw_docs

    def ingest_ticker_news(self, ticker: str) -> list[RawArticle]:
        """Ingest financial news articles for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of RawArticle objects
        """
        logger.info(f"Starting news ingestion for {ticker}")

        # Fetch articles
        articles = self.news_ingester.fetch_articles(ticker)

        logger.info(f"Completed news ingestion for {ticker}: {len(articles)} articles")

        return articles

    def ingest_ticker(
        self,
        ticker: str,
        filing_types: list[str] | None = None,
        years_back: int | None = None,
    ) -> tuple[list[RawDocument], list[RawArticle]]:
        """Ingest both SEC filings and news articles for a ticker.

        Args:
            ticker: Stock ticker symbol
            filing_types: Filing types to fetch. Defaults to config value.
            years_back: Years of historical filings to fetch. Defaults to config value.

        Returns:
            Tuple of (raw_documents, raw_articles)
        """
        logger.info(f"Starting full ingestion for {ticker}")

        # Ingest EDGAR filings
        raw_docs = self.ingest_ticker_edgar(
            ticker, filing_types=filing_types, years_back=years_back
        )

        # Ingest news articles
        articles = self.ingest_ticker_news(ticker)

        logger.info(
            f"Completed full ingestion for {ticker}: "
            f"{len(raw_docs)} filings, {len(articles)} articles"
        )

        return raw_docs, articles

    def chunk_documents(self, docs: list[RawDocument]) -> list[Chunk]:
        """Chunk raw documents into semantic chunks.

        Args:
            docs: List of RawDocument objects

        Returns:
            List of Chunk objects
        """
        logger.info(f"Chunking {len(docs)} documents")

        all_chunks = []
        for doc in docs:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {len(docs)} documents")

        return all_chunks

    def chunk_articles(self, articles: list[RawArticle]) -> list[Chunk]:
        """Chunk raw articles into semantic chunks.

        Args:
            articles: List of RawArticle objects

        Returns:
            List of Chunk objects
        """
        logger.info(f"Chunking {len(articles)} articles")

        all_chunks = []
        for article in articles:
            chunks = self.chunker.chunk_article(article)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {len(articles)} articles")

        return all_chunks

    def ingest_and_chunk_ticker(
        self,
        ticker: str,
        filing_types: list[str] | None = None,
        years_back: int | None = None,
    ) -> list[Chunk]:
        """Ingest and chunk all content for a ticker.

        Args:
            ticker: Stock ticker symbol
            filing_types: Filing types to fetch. Defaults to config value.
            years_back: Years of historical filings to fetch. Defaults to config value.

        Returns:
            List of Chunk objects
        """
        logger.info(f"Starting full ingestion and chunking for {ticker}")

        # Ingest documents and articles
        raw_docs, articles = self.ingest_ticker(
            ticker, filing_types=filing_types, years_back=years_back
        )

        # Chunk documents
        doc_chunks = self.chunk_documents(raw_docs)

        # Chunk articles
        article_chunks = self.chunk_articles(articles)

        # Combine all chunks
        all_chunks = doc_chunks + article_chunks

        # Log breakdown by doc_type
        doc_type_counts = {}
        for chunk in all_chunks:
            doc_type_counts[chunk.doc_type] = doc_type_counts.get(chunk.doc_type, 0) + 1

        logger.info(
            f"Completed full ingestion and chunking for {ticker}: "
            f"{len(all_chunks)} total chunks "
            f"({', '.join(f'{dtype}: {count}' for dtype, count in doc_type_counts.items())})"
        )

        return all_chunks

    def store_chunks(self, chunks: list[Chunk], embeddings: dict[str, np.ndarray]):
        """Store chunks and their embeddings.

        Args:
            chunks: List of Chunk objects
            embeddings: Dictionary mapping chunk_id → embedding
        """
        if not chunks:
            return

        # Store metadata
        self.metadata_store.add_chunks(chunks)

        # Prepare embeddings for vector store
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        embeddings_array = np.array([embeddings[cid] for cid in chunk_ids])

        # Store embeddings
        self.vector_store.add(embeddings_array, chunk_ids)

        logger.info(f"Stored {len(chunks)} chunks")

    def full_ingest(
        self,
        ticker: str,
        filing_types: list[str] | None = None,
        years_back: int | None = None,
    ) -> IngestResult:
        """Run complete ingestion pipeline: ingest → chunk → embed → store.

        Args:
            ticker: Stock ticker symbol
            filing_types: Filing types to fetch. Defaults to config value.
            years_back: Years of historical filings to fetch. Defaults to config value.

        Returns:
            IngestResult with counts
        """
        logger.info(f"Starting full ingestion pipeline for {ticker}")

        # Ingest and chunk
        chunks = self.ingest_and_chunk_ticker(
            ticker, filing_types=filing_types, years_back=years_back
        )

        # Embed
        embeddings = self.embedder.embed_chunks(chunks)

        # Store
        self.store_chunks(chunks, embeddings)

        logger.info(
            f"Completed full ingestion pipeline for {ticker}: "
            f"{len(chunks)} chunks, {len(embeddings)} embeddings"
        )

        return IngestResult(
            ticker=ticker,
            chunks_created=len(chunks),
            chunks_embedded=len(embeddings),
            chunks_stored=len(chunks),
        )

    def ingest_historical_filings(
        self,
        ticker: str,
        after: str,
        before: str,
        filing_types: list[str] | None = None,
    ) -> IngestResult:
        """Backfill SEC filings for an explicit historical window (e.g. a
        pre-COVID period like 2015-01-01 to 2019-12-31).

        EDGAR-only, deliberately - unlike full_ingest(), this does not touch
        news. RSS feeds (NewsIngester's only source) have no historical
        query capability at all; calling ingest_ticker_news() here would
        just refetch today's feed again, redundant with a regular
        full_ingest() run and not actually historical.

        Safe to call repeatedly / on overlapping windows: the embedding
        cache, MetadataStore (upsert via session.merge()), and VectorStore
        (explicit chunk_id dedup) are all idempotent on the content-derived
        chunk_id, so re-processing an already-ingested filing costs nothing
        extra - no wasted OpenAI calls, no duplicate vectors.

        Args:
            ticker: Stock ticker symbol
            after: Absolute start date (YYYY-MM-DD)
            before: Absolute end date (YYYY-MM-DD)
            filing_types: Filing types to fetch. Defaults to config value.

        Returns:
            IngestResult with counts
        """
        logger.info(
            f"Starting historical filings backfill for {ticker}: {after} to {before}"
        )

        edgar_config = self.config.get("ingestion", {}).get("edgar", {})
        if filing_types is None:
            filing_types = edgar_config.get("filing_types", ["10-K", "10-Q"])

        raw_docs = self.edgar_ingester.fetch_filings(
            ticker=ticker, filing_types=filing_types, after=after, before=before
        )

        chunks = self.chunk_documents(raw_docs)
        embeddings = self.embedder.embed_chunks(chunks)
        self.store_chunks(chunks, embeddings)

        logger.info(
            f"Completed historical filings backfill for {ticker}: "
            f"{len(chunks)} chunks, {len(embeddings)} embeddings"
        )

        return IngestResult(
            ticker=ticker,
            chunks_created=len(chunks),
            chunks_embedded=len(embeddings),
            chunks_stored=len(chunks),
        )
