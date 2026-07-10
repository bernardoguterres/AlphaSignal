"""Ingestion endpoint for AlphaSignal API."""

import logging
import time

from fastapi import APIRouter, Depends

from alphasignal.api.dependencies import (
    get_metrics_collector,
    get_pipeline,
    get_retriever,
)
from alphasignal.api.schemas import (
    BatchIngestRequest,
    BatchIngestResponse,
    IngestRequest,
    IngestResponse,
)
from alphasignal.ingestion.pipeline import IngestionPipeline
from alphasignal.monitoring.metrics import MetricsCollector
from alphasignal.retrieval.retriever import HybridRetriever

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/batch", response_model=BatchIngestResponse)
def ingest_batch(
    body: BatchIngestRequest,
    pipeline: IngestionPipeline = Depends(get_pipeline),
    retriever: HybridRetriever = Depends(get_retriever),
    metrics_collector: MetricsCollector = Depends(get_metrics_collector),
) -> BatchIngestResponse:
    """Ingest multiple tickers in sequence.

    Args:
        body: Batch ingest request with list of tickers
        pipeline: Ingestion pipeline instance
        retriever: Hybrid retriever instance
        metrics_collector: Metrics collector instance

    Returns:
        BatchIngestResponse with results for each ticker
    """
    start_time = time.time()

    logger.info(f"Starting batch ingestion for {len(body.tickers)} tickers")

    results = []

    # Process each ticker sequentially to avoid rate limits
    for ticker in body.tickers:
        ticker = ticker.upper()
        ticker_start = time.time()

        try:
            logger.info(f"Batch ingesting {ticker}...")

            # Run full ingestion
            result = pipeline.full_ingest(ticker)

            ticker_latency = int((time.time() - ticker_start) * 1000)
            metrics_collector.record_ingest(ticker_latency)

            results.append(
                IngestResponse(
                    ticker=ticker,
                    status="completed",
                    chunks_created=result.chunks_created,
                    chunks_stored=result.chunks_stored,
                    latency_ms=ticker_latency,
                )
            )

        except Exception as e:
            logger.error(f"Error batch ingesting {ticker}: {e}", exc_info=True)
            metrics_collector.record_error()

            ticker_latency = int((time.time() - ticker_start) * 1000)

            results.append(
                IngestResponse(
                    ticker=ticker,
                    status="failed",
                    chunks_created=0,
                    chunks_stored=0,
                    latency_ms=ticker_latency,
                )
            )

    # Rebuild BM25 index once after all ingestions
    logger.info("Rebuilding BM25 index after batch ingestion...")
    retriever.build_bm25_index()

    total_latency_ms = int((time.time() - start_time) * 1000)

    logger.info(
        f"Batch ingestion complete: "
        f"{len(results)} tickers processed in {total_latency_ms}ms"
    )

    return BatchIngestResponse(results=results, total_latency_ms=total_latency_ms)


@router.post("/{ticker}", response_model=IngestResponse)
def ingest(
    ticker: str,
    body: IngestRequest | None = None,
    pipeline: IngestionPipeline = Depends(get_pipeline),
    retriever: HybridRetriever = Depends(get_retriever),
    metrics_collector: MetricsCollector = Depends(get_metrics_collector),
) -> IngestResponse:
    """Trigger full ingestion pipeline: ingest → chunk → embed → store.

    Args:
        ticker: Stock ticker symbol
        body: Optional IngestRequest with filing parameters
        pipeline: Ingestion pipeline instance
        retriever: Hybrid retriever instance
        metrics_collector: Metrics collector instance

    Returns:
        IngestResponse with ingestion and storage results
    """
    start_time = time.time()

    try:
        ticker = ticker.upper()
        logger.info(f"Starting ingestion for {ticker}")

        # Run full ingestion pipeline, honoring per-request filing overrides if provided
        filing_types = body.filing_types if body else None
        years_back = body.years_back if body else None
        result = pipeline.full_ingest(
            ticker, filing_types=filing_types, years_back=years_back
        )

        # Rebuild BM25 index after ingestion
        logger.info("Rebuilding BM25 index after ingestion...")
        retriever.build_bm25_index()

        latency_ms = int((time.time() - start_time) * 1000)
        metrics_collector.record_ingest(latency_ms)

        logger.info(
            f"Ingestion complete for {ticker}: "
            f"{result.chunks_created} chunks, {latency_ms}ms"
        )

        return IngestResponse(
            ticker=ticker,
            status="completed",
            chunks_created=result.chunks_created,
            chunks_stored=result.chunks_stored,
            latency_ms=latency_ms,
        )

    except Exception as e:
        logger.error(f"Error ingesting ticker {ticker}: {e}", exc_info=True)
        metrics_collector.record_error()

        latency_ms = int((time.time() - start_time) * 1000)

        return IngestResponse(
            ticker=ticker,
            status="failed",
            chunks_created=0,
            chunks_stored=0,
            latency_ms=latency_ms,
        )
