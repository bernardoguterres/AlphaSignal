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

    def __init__(self, config: dict):
        """Initialize ingestion pipeline.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Set up data directories
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

        # Initialize EDGAR ingester
        self.edgar_ingester = EDGARIngester(
            config=config,
            download_dir=str(self.data_dir / "edgar_raw")
        )

        # Initialize news ingester
        self.news_ingester = NewsIngester(config=config)

        # Initialize semantic chunker
        self.chunker = SemanticChunker(config=config)

        # Initialize embedding cache and embedder
        storage_config = config.get("storage", {})
        cache_path = storage_config.get("embeddings_cache_path", "data/embeddings_cache")
        self.embedding_cache = EmbeddingCache(f"{cache_path}/cache.pkl")
        self.embedder = Embedder(config=config, cache=self.embedding_cache)

        # Initialize metadata store
        db_path = storage_config.get("sqlite_db_path", "data/metadata.db")
        self.metadata_store = MetadataStore(db_path)

        # Initialize vector store
        index_path = storage_config.get("faiss_index_path", "data/faiss_index")
        self.vector_store = VectorStore(index_path)
        self.vector_store.load()

    def ingest_ticker_edgar(self, ticker: str) -> list[RawDocument]:
        """Ingest SEC EDGAR filings for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of RawDocument objects
        """
        logger.info(f"Starting EDGAR ingestion for {ticker}")

        # Get filing parameters from config
        edgar_config = self.config.get("ingestion", {}).get("edgar", {})
        filing_types = edgar_config.get("filing_types", ["10-K", "10-Q"])
        years_back = edgar_config.get("years_back", 2)

        # Fetch filings
        raw_docs = self.edgar_ingester.fetch_filings(
            ticker=ticker,
            filing_types=filing_types,
            years_back=years_back
        )

        logger.info(f"Completed EDGAR ingestion for {ticker}: {len(raw_docs)} documents")

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

    def ingest_ticker(self, ticker: str) -> tuple[list[RawDocument], list[RawArticle]]:
        """Ingest both SEC filings and news articles for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Tuple of (raw_documents, raw_articles)
        """
        logger.info(f"Starting full ingestion for {ticker}")

        # Ingest EDGAR filings
        raw_docs = self.ingest_ticker_edgar(ticker)

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

    def ingest_and_chunk_ticker(self, ticker: str) -> list[Chunk]:
        """Ingest and chunk all content for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of Chunk objects
        """
        logger.info(f"Starting full ingestion and chunking for {ticker}")

        # Ingest documents and articles
        raw_docs, articles = self.ingest_ticker(ticker)

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

    def full_ingest(self, ticker: str) -> IngestResult:
        """Run complete ingestion pipeline: ingest → chunk → embed → store.

        Args:
            ticker: Stock ticker symbol

        Returns:
            IngestResult with counts
        """
        logger.info(f"Starting full ingestion pipeline for {ticker}")

        # Ingest and chunk
        chunks = self.ingest_and_chunk_ticker(ticker)

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
            chunks_stored=len(chunks)
        )
