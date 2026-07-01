"""Main FastAPI application for AlphaSignal."""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from alphasignal.api.routes import health, query, sentiment, ingest, metrics
from alphasignal.api.schemas import ErrorResponse
from alphasignal.api.state import AppState
from alphasignal.api.exceptions import AlphaSignalError
from alphasignal.embeddings.cache import EmbeddingCache
from alphasignal.embeddings.embedder import Embedder
from alphasignal.generation.generator import RAGGenerator
from alphasignal.generation.sentiment import SentimentExtractor
from alphasignal.ingestion.pipeline import IngestionPipeline
from alphasignal.monitoring.metrics import MetricsCollector
from alphasignal.retrieval.reranker import CrossEncoderReranker
from alphasignal.retrieval.retriever import HybridRetriever
from alphasignal.store.metadata_store import MetadataStore
from alphasignal.store.vector_store import VectorStore

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure root logger from LOG_LEVEL env var (default INFO).

    Call once at process startup. Uses the same format as AlphaLive so logs
    are parseable consistently across the Alpha system.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    setup_logging()
    # Startup
    logger.info("=" * 80)
    logger.info("AlphaSignal starting up")
    logger.info("=" * 80)

    start_time = time.time()

    # Load environment variables
    load_dotenv()
    logger.info("✓ Loaded environment variables")

    # Load configuration
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"✓ Loaded configuration for {len(config.get('tickers', []))} tickers")

    # Initialize storage paths
    storage_config = config.get("storage", {})
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    faiss_index_path = storage_config.get("faiss_index_path", "data/faiss_index")
    sqlite_db_path = storage_config.get("sqlite_db_path", "data/metadata.db")
    embeddings_cache_path = storage_config.get(
        "embeddings_cache_path", "data/embeddings_cache"
    )

    # Initialize vector store
    vector_store = VectorStore(faiss_index_path, dim=1536)
    vector_store.load()
    logger.info(f"✓ Vector store loaded: {len(vector_store)} vectors")

    # Initialize metadata store
    metadata_store = MetadataStore(sqlite_db_path)
    chunks_indexed = metadata_store.count()
    logger.info(f"✓ Metadata store connected: {chunks_indexed} chunks indexed")

    # Initialize embedding cache and embedder
    embedding_cache = EmbeddingCache(f"{embeddings_cache_path}/cache.pkl")
    embedder = Embedder(config, embedding_cache)
    logger.info(f"✓ Embedder initialized: {config['embeddings']['model']}")

    # Initialize ingestion pipeline, sharing the vector/metadata stores and
    # embedder with the retriever so newly ingested data is immediately
    # visible to /query without a process restart
    pipeline = IngestionPipeline(
        config,
        embedder=embedder,
        vector_store=vector_store,
        metadata_store=metadata_store,
    )
    logger.info("✓ Ingestion pipeline initialized")

    # Initialize retriever
    retriever = HybridRetriever(config, embedder, vector_store, metadata_store)

    # Build BM25 index
    logger.info("Building BM25 index from existing data...")
    retriever.build_bm25_index()
    bm25_docs = len(retriever.bm25_chunk_ids) if retriever.bm25_index else 0
    logger.info(f"✓ BM25 index built: {bm25_docs} documents")

    # Initialize reranker
    reranker = CrossEncoderReranker()
    logger.info("✓ Cross-encoder reranker loaded")

    # Initialize generator
    generator = RAGGenerator(config)
    logger.info(f"✓ RAG generator initialized: {config['generation']['model']}")

    # Initialize sentiment extractor
    sentiment_extractor = SentimentExtractor(config)
    logger.info(f"✓ Sentiment extractor initialized")

    # Initialize metrics collector
    metrics_collector = MetricsCollector()
    logger.info("✓ Metrics collector initialized")

    # Create app state
    app_state = AppState(
        config=config,
        pipeline=pipeline,
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        sentiment_extractor=sentiment_extractor,
        metrics_collector=metrics_collector,
        start_time=start_time,
    )

    # Store in app state
    app.state.app_state = app_state

    # Startup summary
    elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info("STARTUP SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Chunks indexed:      {chunks_indexed}")
    logger.info(f"FAISS vectors:       {len(vector_store)}")
    logger.info(f"BM25 documents:      {bm25_docs}")
    logger.info(f"Embedding model:     {config['embeddings']['model']}")
    logger.info(f"Generation model:    {config['generation']['model']}")
    logger.info(f"Startup time:        {elapsed:.2f}s")
    logger.info("=" * 80)
    logger.info("✓ AlphaSignal ready to serve requests")
    logger.info("=" * 80)

    yield

    # Shutdown
    logger.info("AlphaSignal shutting down")


# Create FastAPI app
app = FastAPI(
    title="AlphaSignal",
    version="0.1.0",
    description="Financial RAG system for sentiment analysis and signal extraction",
    lifespan=lifespan,
)

# Add CORS middleware (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request method, path, status, and duration."""
    start_time = time.time()

    response = await call_next(request)

    duration_ms = int((time.time() - start_time) * 1000)
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {duration_ms}ms"
    )

    return response


# Exception handler for AlphaSignal errors
@app.exception_handler(AlphaSignalError)
async def alphasignal_error_handler(request: Request, exc: AlphaSignalError):
    """Handle AlphaSignal custom errors and return ErrorResponse."""
    logger.error(f"AlphaSignal error [{exc.code}]: {exc.message}", exc_info=True)

    error_response = ErrorResponse(error=exc.message, code=exc.code, detail=exc.detail)

    return JSONResponse(status_code=500, content=error_response.model_dump())


# Exception handler for unhandled errors
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unhandled exceptions and return ErrorResponse."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    error_response = ErrorResponse(
        error="Internal Server Error", code="INTERNAL_ERROR", detail=str(exc)
    )

    return JSONResponse(status_code=500, content=error_response.model_dump())


# Register routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(sentiment.router, prefix="/sentiment", tags=["sentiment"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(metrics.router, prefix="/metrics", tags=["metrics"])
