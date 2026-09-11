"""Sentiment analysis endpoint for AlphaSignal API."""

import logging
import time
from datetime import date
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
        # Get chunks for ticker
        if date_from or date_to:
            # Use date range query
            start_date = date_from if date_from else date.min
            end_date = date_to if date_to else date.max
            chunks = metadata_store.get_chunks_by_date_range(
                start=start_date, end=end_date, ticker=ticker
            )
        else:
            # Get all chunks for ticker
            chunks = metadata_store.get_chunks_by_ticker(ticker)

        logger.info(f"Found {len(chunks)} chunks for {ticker}")

        # Handle no chunks - data_available=False makes "never ingested"
        # explicit and distinct from a genuinely neutral 0.0 score.
        if not chunks:
            latency_ms = int((time.time() - start_time) * 1000)
            metrics_collector.record_sentiment(latency_ms)
            return SentimentResponse(
                ticker=ticker,
                signals=[],
                latest_score=None,
                latency_ms=latency_ms,
                data_available=False,
                status="no_data",
            )

        # Extract sentiment signals
        signals = sentiment_extractor.extract_ticker_sentiment(ticker, chunks)

        # Signals are sorted by date descending, so the first reliable one
        # (if any) is the most recent genuine prediction. An unreliable
        # signal's score is a provider/parsing placeholder, not a real
        # prediction, so it must never be surfaced as latest_score even
        # when it's the most recent chunk chronologically (requirement:
        # never substitute a fabricated value for a missing prediction).
        reliable_signals = [s for s in signals if s.reliable]
        total_chunk_count = len(signals)
        reliable_chunk_count = len(reliable_signals)

        if reliable_chunk_count == 0:
            status, degraded, degradation_reason, latest_score = (
                "degraded",
                True,
                "full_extraction_failure",
                None,
            )
        elif reliable_chunk_count < total_chunk_count:
            status, degraded, degradation_reason, latest_score = (
                "degraded",
                True,
                "partial_extraction_failure",
                reliable_signals[0].score,
            )
        else:
            status, degraded, degradation_reason, latest_score = (
                "ok",
                False,
                None,
                signals[0].score,
            )

        latency_ms = int((time.time() - start_time) * 1000)
        metrics_collector.record_sentiment(latency_ms)

        logger.info(f"Extracted {len(signals)} sentiment signals for {ticker}")

        return SentimentResponse(
            ticker=ticker,
            signals=signals,
            latest_score=latest_score,
            latency_ms=latency_ms,
            status=status,
            degraded=degraded,
            degradation_reason=degradation_reason,
            reliable_chunk_count=reliable_chunk_count,
            total_chunk_count=total_chunk_count,
        )

    except Exception as e:
        logger.error(f"Error getting sentiment for {ticker}: {e}", exc_info=True)
        metrics_collector.record_error()
        raise


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
        # Get all chunks for ticker
        chunks = metadata_store.get_chunks_by_ticker(ticker)

        if not chunks:
            latency_ms = int((time.time() - start_time) * 1000)
            metrics_collector.record_sentiment(latency_ms)
            return {
                "ticker": ticker,
                "period_days": 0,
                "avg_score": None,
                "trend": "unknown",
                "signal_count": 0,
                "most_recent_date": None,
                "latency_ms": latency_ms,
            }

        # Extract sentiment signals
        signals = sentiment_extractor.extract_ticker_sentiment(ticker, chunks)

        if not signals:
            latency_ms = int((time.time() - start_time) * 1000)
            metrics_collector.record_sentiment(latency_ms)
            return {
                "ticker": ticker,
                "period_days": 0,
                "avg_score": None,
                "trend": "unknown",
                "signal_count": 0,
                "most_recent_date": None,
                "latency_ms": latency_ms,
            }

        # Aggregate statistics use only RELIABLE signals - an unreliable
        # signal's score is a provider/parsing placeholder (typically 0.0),
        # not a real prediction, and mixing it into avg_score/trend would
        # silently pull the aggregate toward neutral for reasons unrelated
        # to actual sentiment (audit finding, 2026-09-11: this previously
        # averaged over ALL signals unconditionally, the same leakage
        # already fixed for latest_score in GET /{ticker}).
        reliable_signals = [s for s in signals if s.reliable]
        degraded = len(reliable_signals) < len(signals)

        if not reliable_signals:
            latency_ms = int((time.time() - start_time) * 1000)
            metrics_collector.record_sentiment(latency_ms)
            return {
                "ticker": ticker,
                "period_days": 0,
                "avg_score": None,
                "trend": "unknown",
                "signal_count": len(signals),
                "reliable_signal_count": 0,
                "degraded": True,
                "most_recent_date": None,
                "latency_ms": latency_ms,
            }

        scores = [s.score for s in reliable_signals]
        avg_score = sum(scores) / len(scores)

        # Determine trend
        if len(reliable_signals) >= 3:
            # Compare recent 3 vs older signals
            recent_avg = sum(s.score for s in reliable_signals[:3]) / 3
            if recent_avg > avg_score + 0.1:
                trend = "improving"
            elif recent_avg < avg_score - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Calculate period over all signals' dates (a date range is
        # observational, not a prediction, so unreliable entries don't
        # corrupt it the way including their scores would).
        dates = [s.date for s in signals]
        most_recent = max(dates)
        oldest = min(dates)
        period_days = (most_recent - oldest).days

        latency_ms = int((time.time() - start_time) * 1000)
        metrics_collector.record_sentiment(latency_ms)

        return {
            "ticker": ticker,
            "period_days": period_days,
            "avg_score": round(avg_score, 3),
            "trend": trend,
            "signal_count": len(signals),
            "reliable_signal_count": len(reliable_signals),
            "degraded": degraded,
            "most_recent_date": most_recent,
            "latency_ms": latency_ms,
        }

    except Exception as e:
        logger.error(
            f"Error getting sentiment summary for {ticker}: {e}", exc_info=True
        )
        metrics_collector.record_error()
        raise
