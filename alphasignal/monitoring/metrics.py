"""Metrics collection and monitoring."""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and aggregates API metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self._query_latencies: List[float] = []
        self._ingest_latencies: List[float] = []
        self._sentiment_latencies: List[float] = []
        self._error_count: int = 0

    def record_query(self, latency_ms: float):
        """Record a query request latency.

        Args:
            latency_ms: Latency in milliseconds
        """
        self._query_latencies.append(latency_ms)

    def record_ingest(self, latency_ms: float):
        """Record an ingestion request latency.

        Args:
            latency_ms: Latency in milliseconds
        """
        self._ingest_latencies.append(latency_ms)

    def record_sentiment(self, latency_ms: float):
        """Record a sentiment request latency.

        Args:
            latency_ms: Latency in milliseconds
        """
        self._sentiment_latencies.append(latency_ms)

    def record_error(self):
        """Record an error occurrence."""
        self._error_count += 1

    def get_query_percentiles(self) -> Dict[str, float]:
        """Get query latency percentiles.

        Returns:
            Dictionary with p50, p95, p99 percentiles
        """
        if not self._query_latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        latencies = np.array(self._query_latencies)
        return {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99))
        }

    def get_ingest_percentiles(self) -> Dict[str, float]:
        """Get ingestion latency percentiles.

        Returns:
            Dictionary with p50, p95, p99 percentiles
        """
        if not self._ingest_latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        latencies = np.array(self._ingest_latencies)
        return {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99))
        }

    def get_sentiment_percentiles(self) -> Dict[str, float]:
        """Get sentiment latency percentiles.

        Returns:
            Dictionary with p50, p95, p99 percentiles
        """
        if not self._sentiment_latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        latencies = np.array(self._sentiment_latencies)
        return {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99))
        }

    def get_summary(self) -> Dict:
        """Get complete metrics summary.

        Returns:
            Dictionary with all collected metrics
        """
        return {
            "query": {
                "count": len(self._query_latencies),
                "percentiles": self.get_query_percentiles()
            },
            "ingest": {
                "count": len(self._ingest_latencies),
                "percentiles": self.get_ingest_percentiles()
            },
            "sentiment": {
                "count": len(self._sentiment_latencies),
                "percentiles": self.get_sentiment_percentiles()
            },
            "errors": {
                "count": self._error_count
            }
        }
