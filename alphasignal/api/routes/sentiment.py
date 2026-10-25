"""Sentiment analysis endpoint for AlphaSignal API."""

import logging
import time
from datetime import date, datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from alphasignal.api.dependencies import (
    get_config,
    get_metadata_store,
    get_metrics_collector,
    get_sentiment_extractor,
    validate_ticker_in_config,
)
from alphasignal.api.schemas import SentimentResponse
from alphasignal.generation.sentiment import SentimentExtractor
from alphasignal.monitoring.metrics import MetricsCollector
from alphasignal.store.metadata_store import MetadataStore

router = APIRouter()
logger = logging.getLogger(__name__)


def _utc_today() -> date:
    """News dates are UTC calendar dates, so "today" is the UTC date; the
    server's local date would hide evening news on a host behind UTC."""
    return datetime.now(timezone.utc).date()


@router.get("/{ticker}", response_model=SentimentResponse)
def get_sentiment(
    ticker: str,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    metadata_store: MetadataStore = Depends(get_metadata_store),
    sentiment_extractor: SentimentExtractor = Depends(get_sentiment_extractor),
    config: dict = Depends(get_config),
    metrics_collector: MetricsCollector = Depends(get_metrics_collector),
) -> SentimentResponse:
    """Get sentiment signals for a specific ticker.

    Args:
        ticker: Stock ticker symbol (1-5 characters)
        date_from: Optional start date filter
        date_to: Optional end date filter
        metadata_store: Metadata store instance
        sentiment_extractor: Sentiment extractor instance
        config: Configuration dictionary
        metrics_collector: Metrics collector instance

    Returns:
        SentimentResponse with signals and latest score
    """
    start_time = time.time()

    # Validate ticker length before checking the allowlist (distinct 422 vs 404)
    if not (1 <= len(ticker) <= 5):
        raise HTTPException(status_code=422, detail="Ticker must be 1-5 characters")

    ticker = validate_ticker_in_config(ticker, config)

    logger.info(f"Getting sentiment for {ticker}")

    try:
        # Point-in-time: as_of = date_to (or today). Documents dated after
        # as_of are never used, so a historical query cannot see the future.
        as_of = date_to or _utc_today()
        start_date = date_from if date_from else date.min
        chunks = metadata_store.get_chunks_by_date_range(
            start=start_date, end=as_of, ticker=ticker
        )
        coverage = _coverage(metadata_store, ticker)
        logger.info(f"Found {len(chunks)} chunks for {ticker} up to {as_of}")

        agg = sentiment_extractor.aggregate(ticker, chunks, as_of=as_of)
        latency_ms = int((time.time() - start_time) * 1000)
        metrics_collector.record_sentiment(latency_ms)

        common = dict(
            ticker=ticker,
            latency_ms=latency_ms,
            as_of=as_of,
            freshness_days=sentiment_extractor.freshness_days,
            last_document_date=agg.last_document_date,
            coverage=coverage,
        )
        if agg.status == "no_data":
            return SentimentResponse(
                signals=[],
                latest_score=None,
                # "no_data" means no company-attributed evidence; chunks may
                # still exist (body-only mentions, unattributed legacy rows).
                data_available=bool(chunks),
                status="no_data",
                **common,
            )
        if agg.status == "stale":
            return SentimentResponse(
                signals=[],
                latest_score=None,
                data_available=True,
                status="stale",
                **common,
            )
        return SentimentResponse(
            signals=agg.signals,
            latest_score=agg.score,
            status=agg.status,
            degraded=agg.status == "degraded",
            degradation_reason=agg.degradation_reason,
            reliable_chunk_count=agg.reliable_chunk_count,
            total_chunk_count=agg.total_chunk_count,
            documents_used=agg.documents_used,
            **common,
        )

    except Exception as e:
        logger.error(f"Error getting sentiment for {ticker}: {e}", exc_info=True)
        metrics_collector.record_error()
        raise


def _coverage(metadata_store: MetadataStore, ticker: str) -> dict:
    """What the corpus holds for `ticker`: per doc_type first/last date and
    calendar years with no documents between them (e.g. the 2020-2023 gap).
    Reported so a consumer never mistakes the corpus for complete."""
    from sqlalchemy import text as sql

    out = {}
    with metadata_store.engine.connect() as conn:
        rows = conn.execute(
            sql(
                "SELECT doc_type, min(date), max(date), group_concat(DISTINCT "
                "strftime('%Y', date)) FROM chunks WHERE ticker = :t GROUP BY doc_type"
            ),
            {"t": ticker},
        ).fetchall()
    for doc_type, first, last, years in rows:
        present = sorted({int(y) for y in (years or "").split(",") if y})
        missing = (
            [y for y in range(present[0], present[-1] + 1) if y not in present]
            if present
            else []
        )
        out[doc_type] = {
            "first": str(first),
            "last": str(last),
            "years_missing": missing,
        }
    return out


@router.get("/{ticker}/summary")
def get_sentiment_summary(
    ticker: str,
    metadata_store: MetadataStore = Depends(get_metadata_store),
    sentiment_extractor: SentimentExtractor = Depends(get_sentiment_extractor),
    config: dict = Depends(get_config),
    metrics_collector: MetricsCollector = Depends(get_metrics_collector),
):
    """Get aggregate sentiment summary for a ticker.

    Args:
        ticker: Stock ticker symbol
        metadata_store: Metadata store instance
        sentiment_extractor: Sentiment extractor instance
        config: Configuration dictionary
        metrics_collector: Metrics collector instance

    Returns:
        JSON with aggregate sentiment statistics
    """
    start_time = time.time()

    ticker = validate_ticker_in_config(ticker, config)

    logger.info(f"Getting sentiment summary for {ticker}")

    try:
        as_of = _utc_today()
        chunks = metadata_store.get_chunks_by_date_range(
            start=date.min, end=as_of, ticker=ticker
        )
        agg = sentiment_extractor.aggregate(ticker, chunks, as_of=as_of)
        latency_ms = int((time.time() - start_time) * 1000)
        metrics_collector.record_sentiment(latency_ms)
        sigs = agg.signals  # document-level, newest first, reliable only
        if not sigs:
            return {
                "ticker": ticker,
                "status": agg.status,
                "period_days": 0,
                "avg_score": None,
                "trend": "unknown",
                "signal_count": 0,
                "most_recent_date": agg.last_document_date,
                "latency_ms": latency_ms,
            }
        avg = sum(x.score for x in sigs) / len(sigs)
        trend = "stable"
        if len(sigs) >= 3:
            recent = sum(x.score for x in sigs[:3]) / 3
            trend = (
                "improving"
                if recent > avg + 0.1
                else "declining" if recent < avg - 0.1 else "stable"
            )
        dates = [x.date for x in sigs]
        return {
            "ticker": ticker,
            "status": agg.status,
            "period_days": (max(dates) - min(dates)).days,
            "avg_score": round(avg, 3),
            "aggregate_score": agg.score,
            "trend": trend,
            "signal_count": len(sigs),
            "degraded": agg.status == "degraded",
            "most_recent_date": max(dates),
            "freshness_days": sentiment_extractor.freshness_days,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        logger.error(
            f"Error getting sentiment summary for {ticker}: {e}", exc_info=True
        )
        metrics_collector.record_error()
        raise
