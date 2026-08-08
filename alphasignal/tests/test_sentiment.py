"""Tests for sentiment extraction."""

import json
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from alphasignal.generation import SentimentResult
from alphasignal.generation.sentiment import SentimentExtractor
from alphasignal.ingestion import Chunk


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {"generation": {"model": "gpt-4o-mini", "sentiment_cache_hours": 24}}


@pytest.fixture
def test_chunk():
    """Create a test chunk."""
    return Chunk(
        chunk_id="aapl_10k_test_0001",
        ticker="AAPL",
        text="Apple Inc. reported strong revenue growth driven by exceptional iPhone sales and robust demand across all product categories.",
        token_count=50,
        doc_type="10-K",
        source="SEC EDGAR",
        section="item_7",
        date=date(2024, 10, 31),
        url=None,
        chunk_index=0,
        total_chunks=1,
    )


@pytest.fixture
def test_chunks():
    """Create test chunks with various dates."""
    today = date.today()
    return [
        Chunk(
            chunk_id=f"aapl_test_{i:04d}",
            ticker="AAPL",
            text=f"Apple revenue analysis for period {i}.",
            token_count=50,
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_7",
            date=today - timedelta(days=i * 30),  # Spread across months
            url=None,
            chunk_index=i,
            total_chunks=15,
        )
        for i in range(15)
    ]


def test_sentiment_extractor_returns_valid_score(test_config, test_chunk):
    """Test that sentiment extraction returns valid scores."""
    with patch("alphasignal.generation.sentiment.OpenAI") as MockOpenAI:
        # Mock OpenAI response with valid JSON
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "score": 0.75,
                "confidence": 0.85,
                "key_positive": [
                    "strong revenue growth",
                    "exceptional sales",
                    "robust demand",
                ],
                "key_negative": [],
                "summary": "Positive sentiment driven by strong sales performance.",
            }
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        extractor = SentimentExtractor(test_config)
        result = extractor.extract_sentiment(test_chunk)

        # Should return valid SentimentResult
        assert isinstance(result, SentimentResult)
        assert -1.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.score == 0.75
        assert result.confidence == 0.85
        assert len(result.key_positive) == 3
        assert "strong revenue growth" in result.key_positive


def test_sentiment_extractor_caches_results(test_config, test_chunk):
    """Test that sentiment extraction caches results."""
    with patch("alphasignal.generation.sentiment.OpenAI") as MockOpenAI:
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "score": 0.5,
                "confidence": 0.7,
                "key_positive": ["growth"],
                "key_negative": [],
                "summary": "Neutral to positive sentiment.",
            }
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        extractor = SentimentExtractor(test_config)

        # First call
        result1 = extractor.extract_sentiment(test_chunk)

        # Second call
        result2 = extractor.extract_sentiment(test_chunk)

        # OpenAI should only be called once
        assert mock_client.chat.completions.create.call_count == 1

        # Results should be identical
        assert result1.score == result2.score
        assert result1.confidence == result2.confidence


def test_sentiment_extractor_handles_json_error(test_config, test_chunk):
    """Test that sentiment extraction handles JSON parse errors."""
    with patch("alphasignal.generation.sentiment.OpenAI") as MockOpenAI:
        # Mock OpenAI response with invalid JSON
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Sorry, I can't do that."

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        extractor = SentimentExtractor(test_config)
        result = extractor.extract_sentiment(test_chunk)

        # Should return default result without raising
        assert isinstance(result, SentimentResult)
        assert result.score == 0.0
        assert result.confidence == 0.0
        assert result.summary == "Parse error"


def test_sentiment_extractor_nan_score_does_not_clamp_to_max_bullish(
    test_config, test_chunk
):
    """Regression test: NaN score/confidence from the LLM provider must
    NOT silently clamp to 1.0 (max bullish) - Python's max()/min() don't
    special-case NaN, so max(-1.0, min(1.0, nan)) previously evaluated to
    1.0, turning a provider malfunction into a false maximum-bullish
    signal. Must fail safe to neutral instead, like a JSON parse error."""
    with patch("alphasignal.generation.sentiment.OpenAI") as MockOpenAI:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        # json.dumps can't emit NaN in strict mode, but Python's json module
        # (used by _parse_sentiment_json) accepts the bare NaN/Infinity
        # tokens by default - exactly what a malformed provider response
        # could contain.
        mock_response.choices[0].message.content = (
            '{"score": NaN, "confidence": 0.9, "key_positive": [], '
            '"key_negative": [], "summary": "test"}'
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        extractor = SentimentExtractor(test_config)
        result = extractor.extract_sentiment(test_chunk)

        assert isinstance(result, SentimentResult)
        assert result.score == 0.0
        assert result.confidence == 0.0
        assert result.summary == "Invalid score from provider"


def test_sentiment_extractor_infinity_score_does_not_clamp_silently(
    test_config, test_chunk
):
    """Same regression as the NaN case, for +Infinity - clamping
    +Infinity to 1.0 happens to look "correct" numerically, but it still
    masks a genuine provider malfunction as a legitimate strong-buy score
    rather than flagging it."""
    with patch("alphasignal.generation.sentiment.OpenAI") as MockOpenAI:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"score": Infinity, "confidence": 0.9, "key_positive": [], '
            '"key_negative": [], "summary": "test"}'
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        extractor = SentimentExtractor(test_config)
        result = extractor.extract_sentiment(test_chunk)

        assert result.score == 0.0
        assert result.confidence == 0.0
        assert result.summary == "Invalid score from provider"


def test_sentiment_extractor_ordinary_out_of_range_score_still_clamps(
    test_config, test_chunk
):
    """Sanity check the fix didn't break legitimate clamping - an ordinary
    (finite) out-of-range value like 1.5 must still clamp to 1.0, not be
    treated as invalid."""
    with patch("alphasignal.generation.sentiment.OpenAI") as MockOpenAI:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "score": 1.5,
                "confidence": 0.9,
                "key_positive": [],
                "key_negative": [],
                "summary": "test",
            }
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        extractor = SentimentExtractor(test_config)
        result = extractor.extract_sentiment(test_chunk)

        assert result.score == 1.0
        assert result.summary != "Invalid score from provider"


def test_extract_ticker_sentiment_returns_most_recent(test_config, test_chunks):
    """Test that extract_ticker_sentiment returns at most 10 most recent chunks."""
    with patch("alphasignal.generation.sentiment.OpenAI") as MockOpenAI:
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "score": 0.6,
                "confidence": 0.8,
                "key_positive": ["revenue growth"],
                "key_negative": [],
                "summary": "Positive outlook.",
            }
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        extractor = SentimentExtractor(test_config)

        # Extract sentiment for 15 chunks (should only process 10)
        signals = extractor.extract_ticker_sentiment("AAPL", test_chunks)

        # Should return at most 10 signals
        assert len(signals) <= 10

        # Should be sorted by date descending
        dates = [s.date for s in signals]
        assert dates == sorted(dates, reverse=True)

        # All should be for AAPL
        assert all(s.ticker == "AAPL" for s in signals)


def test_sentiment_signal_score_range(test_config, test_chunks):
    """Test that all sentiment signals have scores in valid range."""
    with patch("alphasignal.generation.sentiment.OpenAI") as MockOpenAI:
        # Mock varying scores
        scores = [0.8, -0.3, 0.5, -0.7, 0.2, 0.9, -0.1, 0.4, -0.5, 0.6]

        def make_response(score):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps(
                {
                    "score": score,
                    "confidence": 0.8,
                    "key_positive": ["positive"],
                    "key_negative": ["negative"],
                    "summary": f"Sentiment score: {score}",
                }
            )
            return mock_response

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            make_response(s) for s in scores
        ]
        MockOpenAI.return_value = mock_client

        extractor = SentimentExtractor(test_config)
        signals = extractor.extract_ticker_sentiment("AAPL", test_chunks[:10])

        # All scores should be in [-1, 1] range
        assert all(-1.0 <= s.score <= 1.0 for s in signals)


def test_sentiment_positive_keywords_extracted(test_config, test_chunk):
    """Test that positive keywords are correctly extracted."""
    with patch("alphasignal.generation.sentiment.OpenAI") as MockOpenAI:
        # Mock response with specific positive keywords
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "score": 0.85,
                "confidence": 0.9,
                "key_positive": [
                    "exceptional growth",
                    "strong performance",
                    "record revenue",
                ],
                "key_negative": [],
                "summary": "Highly positive financial results.",
            }
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        extractor = SentimentExtractor(test_config)
        result = extractor.extract_sentiment(test_chunk)

        # Should contain specific positive keywords
        assert "exceptional growth" in result.key_positive
        assert "strong performance" in result.key_positive
        assert "record revenue" in result.key_positive
        assert len(result.key_positive) <= 3  # Max 3 as per spec


def test_sentiment_cache_ttl_respected(test_config, test_chunk):
    """Test that sentiment cache TTL is respected."""
    with patch("alphasignal.generation.sentiment.OpenAI") as MockOpenAI:
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "score": 0.5,
                "confidence": 0.7,
                "key_positive": [],
                "key_negative": [],
                "summary": "Neutral sentiment.",
            }
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        extractor = SentimentExtractor(test_config)

        # First call
        extractor.extract_sentiment(test_chunk)
        assert mock_client.chat.completions.create.call_count == 1

        # Manually expire cache by modifying cached timestamp
        cache_hours = extractor.cache_hours
        old_timestamp = datetime.now() - timedelta(hours=cache_hours + 1)
        if test_chunk.chunk_id in extractor._cache:
            cached_result, _ = extractor._cache[test_chunk.chunk_id]
            extractor._cache[test_chunk.chunk_id] = (cached_result, old_timestamp)

        # Second call (cache expired)
        extractor.extract_sentiment(test_chunk)

        # Should have called OpenAI a second time
        assert mock_client.chat.completions.create.call_count == 2
