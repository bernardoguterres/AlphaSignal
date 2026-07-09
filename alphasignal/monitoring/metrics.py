"""Metrics collection and monitoring."""

import logging
from collections import deque
from typing import Deque, Dict

import numpy as np

logger = logging.getLogger(__name__)

# Rolling window for percentile computation. Bounded so a long-running
# process doesn't grow these lists forever - "count" below still tracks
# the true lifetime total, only the percentile sample window is capped.
MAX_LATENCY_SAMPLES = 1000


class MetricsCollector:
    """Collects and aggregates API metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self._query_latencies: Deque[float] = deque(maxlen=MAX_LATENCY_SAMPLES)
        self._ingest_latencies: Deque[float] = deque(maxlen=MAX_LATENCY_SAMPLES)
        self._sentiment_latencies: Deque[float] = deque(maxlen=MAX_LATENCY_SAMPLES)
        self._query_count: int = 0
        self._ingest_count: int = 0
        self._sentiment_count: int = 0
        self._error_count: int = 0

    def record_query(self, latency_ms: float):
        """Record a query request latency.

        Args:
            latency_ms: Latency in milliseconds
        """
        self._query_latencies.append(latency_ms)
        self._query_count += 1

    def record_ingest(self, latency_ms: float):
        """Record an ingestion request latency.

        Args:
            latency_ms: Latency in milliseconds
        """
        self._ingest_latencies.append(latency_ms)
        self._ingest_count += 1

    def record_sentiment(self, latency_ms: float):
        """Record a sentiment request latency.

        Args:
            latency_ms: Latency in milliseconds
        """
        self._sentiment_latencies.append(latency_ms)
        self._sentiment_count += 1

    def record_error(self):
        """Record an error occurrence."""
        self._error_count += 1

    @staticmethod
    def _percentiles(latencies) -> Dict[str, float]:
        """Compute p50/p95/p99 for a list of latencies.

        Args:
            latencies: Latencies in milliseconds

        Returns:
            Dictionary with p50, p95, p99 percentiles
        """
        if not latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        arr = np.array(latencies)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    def get_summary(self) -> Dict:
        """Get complete metrics summary.

        Returns:
            Dictionary with all collected metrics
        """
        return {
            "query": {
                "count": self._query_count,
                "percentiles": self._percentiles(self._query_latencies),
            },
            "ingest": {
                "count": self._ingest_count,
                "percentiles": self._percentiles(self._ingest_latencies),
            },
            "sentiment": {
                "count": self._sentiment_count,
                "percentiles": self._percentiles(self._sentiment_latencies),
            },
            "errors": {"count": self._error_count},
        }
