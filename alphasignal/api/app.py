"""Main FastAPI application for AlphaSignal."""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from alphasignal.api.routes import health, query, sentiment, ingest, metrics
from alphasignal.api.dependencies import require_api_key
from alphasignal.api.schemas import ErrorResponse
from alphasignal.api.state import AppState
from alphasignal.generation.generator import RAGGenerator
from alphasignal.generation.sentiment import SentimentExtractor
from alphasignal.ingestion.pipeline import IngestionPipeline
from alphasignal.monitoring.metrics import MetricsCollector
from alphasignal.retrieval.reranker import CrossEncoderReranker
from alphasignal.retrieval.retriever import HybridRetriever
from alphasignal.scripts._common import build_storage_components

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
    # load_dotenv() must run before the ALPHASIGNAL_API_KEY check below, or a
    # key set only in .env (not already in the shell environment) reads as
    # unset here even though it ends up loaded - and enforced - moments
    # later, since os.environ is process-wide and require_api_key() (per-
    # request) runs after this whole startup block finishes either way. The
    # bug was purely this log message lying about the actual security state,
    # not a real enforcement gap.
    load_dotenv()
    if os.getenv("ALPHASIGNAL_API_KEY", ""):
        logger.info(
            "API-key auth ENABLED (ALPHASIGNAL_API_KEY set) - all routes except /health require X-API-Key"
        )
    else:
        logger.warning(
            "API-key auth DISABLED (ALPHASIGNAL_API_KEY not set) - fine locally, "
            "but set it before any public deploy or anyone can spend the OpenAI budget"
        )
    # Startup
    logger.info("=" * 80)
    logger.info("AlphaSignal starting up")
    logger.info("=" * 80)

    start_time = time.time()

    logger.info("Loaded environment variables")

    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

        logger.info(
            f"Loaded configuration for {len(config.get('tickers', []))} tickers"
        )

    vector_store, metadata_store, embedder = build_storage_components(config)
    logger.info(f"Vector store loaded: {len(vector_store)} vectors")
    chunks_indexed = metadata_store.count()
    logger.info(f"Metadata store connected: {chunks_indexed} chunks indexed")
    logger.info(f"Embedder initialized: {config['embeddings']['model']}")

    # Pass the shared vector/metadata stores and embedder to the pipeline so
    # newly ingested data is immediately visible to /query without a restart
    pipeline = IngestionPipeline(
        config,
        embedder=embedder,
        vector_store=vector_store,
        metadata_store=metadata_store,
    )
    logger.info("Ingestion pipeline initialized")

    retriever = HybridRetriever(config, embedder, vector_store, metadata_store)

    logger.info("Building BM25 index from existing data...")
    retriever.build_bm25_index()
    bm25_docs = len(retriever.bm25_chunk_ids) if retriever.bm25_index else 0
    logger.info(f"BM25 index built: {bm25_docs} documents")

    reranker = CrossEncoderReranker()
    logger.info("Cross-encoder reranker loaded")

    generator = RAGGenerator(config)
    logger.info(f"RAG generator initialized: {config['generation']['model']}")

    sentiment_extractor = SentimentExtractor(config)
    logger.info("Sentiment extractor initialized")

    metrics_collector = MetricsCollector()
    logger.info("Metrics collector initialized")

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

    app.state.app_state = app_state

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
    logger.info("AlphaSignal ready to serve requests")
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
    # allow_credentials=True + allow_origins=["*"] is an invalid combination
    # per the CORS spec (browsers reject it) and unnecessary here - auth is
    # via the X-API-Key header (require_api_key), not cookies, so
    # credentialed CORS isn't needed (audit finding, 2026-07-14).
    allow_credentials=False,
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


# Exception handler for unhandled errors
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unhandled exceptions and return ErrorResponse.

    detail intentionally omits str(exc): unhandled exceptions can carry
    internals (file paths, SQL, stack frames, API key fragments in error
    strings) that shouldn't reach a client. The full exception is still
    logged server-side via exc_info=True above (audit finding: previously
    detail=str(exc) leaked this to every caller, 2026-07-14).
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    error_response = ErrorResponse(
        error="Internal Server Error",
        code="INTERNAL_ERROR",
        detail="An unexpected error occurred",
    )

    return JSONResponse(status_code=500, content=error_response.model_dump())


# Register routers
# /health stays open (Railway's healthcheck can't send headers); everything
# else requires X-API-Key once ALPHASIGNAL_API_KEY is set on the server.
_auth = [Depends(require_api_key)]
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(query.router, prefix="/query", tags=["query"], dependencies=_auth)
app.include_router(
    sentiment.router, prefix="/sentiment", tags=["sentiment"], dependencies=_auth
)
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"], dependencies=_auth)
app.include_router(
    metrics.router, prefix="/metrics", tags=["metrics"], dependencies=_auth
)
