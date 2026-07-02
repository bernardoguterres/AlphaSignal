"""Tests for the MetricsCollector."""

from alphasignal.monitoring.metrics import MetricsCollector


def test_metrics_collector_starts_empty():
    """Test that a fresh collector reports zero counts and zeroed percentiles."""
    collector = MetricsCollector()

    summary = collector.get_summary()

    assert summary["query"]["count"] == 0
    assert summary["query"]["percentiles"] == {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    assert summary["ingest"]["count"] == 0
    assert summary["sentiment"]["count"] == 0
    assert summary["errors"]["count"] == 0


def test_record_query_updates_count_and_percentiles():
    """Test that record_query accumulates latencies and computes real percentiles."""
    collector = MetricsCollector()

    for latency in [10.0, 20.0, 30.0, 40.0, 50.0]:
        collector.record_query(latency)

    summary = collector.get_summary()

    assert summary["query"]["count"] == 5
    # p50 of [10,20,30,40,50] should be the median, 30.0
    assert summary["query"]["percentiles"]["p50"] == 30.0
    # p99 should be close to the max
    assert summary["query"]["percentiles"]["p99"] <= 50.0
    assert summary["query"]["percentiles"]["p99"] >= 40.0


def test_record_ingest_and_sentiment_are_tracked_independently():
    """Test that ingest and sentiment latencies don't leak into other categories."""
    collector = MetricsCollector()

    collector.record_ingest(100.0)
    collector.record_ingest(200.0)
    collector.record_sentiment(50.0)

    summary = collector.get_summary()

    assert summary["ingest"]["count"] == 2
    assert summary["sentiment"]["count"] == 1
    assert summary["query"]["count"] == 0


def test_record_error_increments_error_count():
    """Test that record_error increments the running error counter."""
    collector = MetricsCollector()

    collector.record_error()
    collector.record_error()
    collector.record_error()

    summary = collector.get_summary()
    assert summary["errors"]["count"] == 3
