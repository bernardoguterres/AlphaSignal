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

    # Validate and normalize ticker
    ticker = ticker.upper()

    # Validate ticker length
    if not (1 <= len(ticker) <= 5):
        raise HTTPException(status_code=422, detail="Ticker must be 1-5 characters")

    # Validate ticker is in config
    valid_tickers = config.get("tickers", [])
    if ticker not in valid_tickers:
        raise HTTPException(
            status_code=404, detail=f"Ticker {ticker} not found in knowledge base"
        )

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

        # Handle no chunks
        if not chunks:
            latency_ms = int((time.time() - start_time) * 1000)
            metrics_collector.record_sentiment(latency_ms)
            return SentimentResponse(
                ticker=ticker, signals=[], latest_score=None, latency_ms=latency_ms
            )

        # Extract sentiment signals
        signals = sentiment_extractor.extract_ticker_sentiment(ticker, chunks)

        # Get latest score
        latest_score = signals[0].score if signals else None

        latency_ms = int((time.time() - start_time) * 1000)
        metrics_collector.record_sentiment(latency_ms)

        logger.info(f"Extracted {len(signals)} sentiment signals for {ticker}")

        return SentimentResponse(
            ticker=ticker,
            signals=signals,
            latest_score=latest_score,
            latency_ms=latency_ms,
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

    # Validate and normalize ticker
    ticker = ticker.upper()

    # Validate ticker is in config
    valid_tickers = config.get("tickers", [])
    if ticker not in valid_tickers:
        raise HTTPException(
            status_code=404, detail=f"Ticker {ticker} not found in knowledge base"
        )

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

        # Calculate statistics
        scores = [s.score for s in signals]
        avg_score = sum(scores) / len(scores)

        # Determine trend
        if len(signals) >= 3:
            # Compare recent 3 vs older signals
            recent_avg = sum(s.score for s in signals[:3]) / 3
            if recent_avg > avg_score + 0.1:
                trend = "improving"
            elif recent_avg < avg_score - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Calculate period
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
            "most_recent_date": most_recent,
            "latency_ms": latency_ms,
        }

    except Exception as e:
        logger.error(
            f"Error getting sentiment summary for {ticker}: {e}", exc_info=True
        )
        metrics_collector.record_error()
        raise
