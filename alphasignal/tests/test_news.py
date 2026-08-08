"""Tests for financial news ingestion."""

import time
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from alphasignal.ingestion import RawArticle
from alphasignal.ingestion.news import NewsIngester


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "ingestion": {
            "news": {
                "sources": [
                    {
                        "name": "Yahoo Finance",
                        "url_template": "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}",
                    },
                    {
                        "name": "Reuters",
                        "url_template": "https://feeds.reuters.com/reuters/companyNews",
                    },
                ],
                "max_articles_per_ticker": 50,
                "max_age_days": 90,
            }
        }
    }


@pytest.fixture
def news_ingester(test_config):
    """Create a NewsIngester instance."""
    return NewsIngester(test_config)


def mock_rss_feed(ticker: str, n_articles: int, days_old: int = 0) -> MagicMock:
    """Create a mock RSS feed with n_articles entries.

    Args:
        ticker: Ticker symbol to mention in articles
        n_articles: Number of articles to generate
        days_old: How many days old the articles should be

    Returns:
        Mock feedparser result
    """
    feed = MagicMock()
    feed.entries = []

    published_time = datetime.now() - timedelta(days=days_old)
    published_struct = time.struct_time(
        (
            published_time.year,
            published_time.month,
            published_time.day,
            12,
            0,
            0,
            0,
            0,
            0,
        )
    )

    class Entry(dict):
        """Dict subclass that allows attribute access."""

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)

        def __setattr__(self, key, value):
            self[key] = value

    for i in range(n_articles):
        entry = Entry(
            {
                "title": f"{ticker} Stock News Article {i + 1}",
                "link": f"https://example.com/article-{ticker}-{i + 1}",
                "published_parsed": published_struct,
                "updated_parsed": None,
            }
        )
        entry["summary"] = (
            f"This is a news article about {ticker} and its business operations. " * 10
        )

        feed.entries.append(entry)

    return feed


def test_fetch_articles_returns_raw_articles(news_ingester):
    """Test that fetch_articles returns list of RawArticle objects."""
    ticker = "AAPL"

    # Mock feedparser to return 10 articles
    with patch("alphasignal.ingestion.news.feedparser.parse") as mock_parse:
        mock_parse.return_value = mock_rss_feed(ticker, 10)

        # Fetch articles
        results = news_ingester.fetch_articles(ticker)

    # Assertions
    assert len(results) > 0
    assert all(isinstance(article, RawArticle) for article in results)
    assert all(article.ticker == ticker for article in results)
    assert all(article.source in ["Yahoo Finance", "Reuters"] for article in results)


def test_fetch_articles_filters_by_age(news_ingester):
    """Test that fetch_articles filters out old articles."""
    ticker = "AAPL"

    # Mock feedparser to return mixed old and recent articles
    with patch("alphasignal.ingestion.news.feedparser.parse") as mock_parse:
        # First call returns 5 recent articles, second call returns 5 old articles
        mock_parse.side_effect = [
            mock_rss_feed(ticker, 5, days_old=10),  # Recent
            mock_rss_feed(ticker, 5, days_old=100),  # Old (> 90 days)
        ]

        # Fetch articles
        results = news_ingester.fetch_articles(ticker)

    # Should only get the recent articles
    assert len(results) == 5

    # All articles should be within max_age_days
    cutoff_date = datetime.now() - timedelta(days=news_ingester.max_age_days)
    for article in results:
        article_datetime = datetime.combine(article.published_date, datetime.min.time())
        assert article_datetime >= cutoff_date


def test_fetch_articles_deduplicates_by_url(news_ingester):
    """Test that fetch_articles removes duplicate URLs."""
    ticker = "AAPL"

    # Create feed with duplicate URLs
    feed = MagicMock()
    # Use recent date to avoid age filtering
    recent_date = datetime.now() - timedelta(days=5)
    published_struct = time.struct_time(
        (recent_date.year, recent_date.month, recent_date.day, 12, 0, 0, 0, 0, 0)
    )

    class Entry(dict):
        """Dict subclass that allows attribute access."""

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)

        def __setattr__(self, key, value):
            self[key] = value

    # 3 articles, but 2 with the same URL
    entry1 = Entry(
        {
            "title": f"{ticker} Article 1",
            "link": "https://example.com/article-1",
            "published_parsed": published_struct,
            "updated_parsed": None,
        }
    )
    entry1["summary"] = f"This is about {ticker}. " * 10

    entry2 = Entry(
        {
            "title": f"{ticker} Article 2",
            "link": "https://example.com/article-1",  # Duplicate URL
            "published_parsed": published_struct,
            "updated_parsed": None,
        }
    )
    entry2["summary"] = f"This is also about {ticker}. " * 10

    entry3 = Entry(
        {
            "title": f"{ticker} Article 3",
            "link": "https://example.com/article-3",
            "published_parsed": published_struct,
            "updated_parsed": None,
        }
    )
    entry3["summary"] = f"Another article about {ticker}. " * 10

    feed.entries = [entry1, entry2, entry3]

    with patch("alphasignal.ingestion.news.feedparser.parse") as mock_parse:
        mock_parse.return_value = feed

        # Fetch articles
        results = news_ingester.fetch_articles(ticker)

    # Should only get 2 unique articles
    assert len(results) == 2

    # Check that URLs are unique
    urls = [article.url for article in results]
    assert len(urls) == len(set(urls))


def test_fetch_articles_filters_irrelevant(news_ingester):
    """Test that fetch_articles filters out irrelevant articles."""
    ticker = "AAPL"

    # Create feed with mixed relevant and irrelevant articles
    feed = MagicMock()
    # Use recent date to avoid age filtering
    recent_date = datetime.now() - timedelta(days=5)
    published_struct = time.struct_time(
        (recent_date.year, recent_date.month, recent_date.day, 12, 0, 0, 0, 0, 0)
    )

    class Entry(dict):
        """Dict subclass that allows attribute access."""

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)

        def __setattr__(self, key, value):
            self[key] = value

    # 5 AAPL articles
    aapl_entries = []
    for i in range(5):
        entry = Entry(
            {
                "title": f"Apple Stock News {i + 1}",
                "link": f"https://example.com/apple-{i + 1}",
                "published_parsed": published_struct,
                "updated_parsed": None,
            }
        )
        entry["summary"] = (
            f"Apple Inc. continues to innovate in consumer electronics. " * 10
        )
        aapl_entries.append(entry)

    # 3 unrelated articles (no mention of AAPL or Apple)
    unrelated_entries = []
    for i in range(3):
        entry = Entry(
            {
                "title": f"Tesla Stock Update {i + 1}",
                "link": f"https://example.com/tesla-{i + 1}",
                "published_parsed": published_struct,
                "updated_parsed": None,
            }
        )
        entry["summary"] = f"Tesla continues to lead in electric vehicles. " * 10
        unrelated_entries.append(entry)

    feed.entries = aapl_entries + unrelated_entries

    with patch("alphasignal.ingestion.news.feedparser.parse") as mock_parse:
        mock_parse.return_value = feed

        # Fetch articles for AAPL
        results = news_ingester.fetch_articles(ticker)

    # Should only get AAPL-relevant articles
    assert len(results) == 5


def test_parse_article_cleans_html(news_ingester):
    """Test that parse_article removes HTML tags from content."""
    # Create entry with HTML content (make sure it's > 50 chars after cleaning)
    entry = MagicMock()
    entry.get = MagicMock(return_value=None)
    entry.title = "Test Article"
    entry.link = "https://example.com/test"
    entry.summary = """
        <html>
        <body>
            <h1>Important News</h1>
            <p>This is <strong>important</strong> content about the company and its business
            operations. The company continues to expand its market presence and deliver value
            to shareholders through innovation and strategic partnerships.</p>
            <script>alert('remove me');</script>
            <p>More content here with <a href="#">links</a> and formatting that should be
            removed during parsing while preserving the actual text content.</p>
        </body>
        </html>
    """
    entry.published_parsed = time.struct_time((2024, 1, 1, 12, 0, 0, 0, 0, 0))
    entry.updated_parsed = None
    entry.content = None
    entry.description = None

    # Parse the article
    result = news_ingester.parse_article(entry)

    # Assertions
    assert result is not None
    assert "<html>" not in result.content
    assert "<p>" not in result.content
    assert "<strong>" not in result.content
    assert "alert" not in result.content
    assert "important" in result.content.lower()
    assert "content about the company" in result.content.lower()


def test_parse_article_returns_none_for_short_content(news_ingester):
    """Test that parse_article returns None for very short content."""
    # Create entry with short content (< 50 chars)
    entry = MagicMock()
    entry.get = MagicMock(return_value=None)
    entry.title = "Short Article"
    entry.link = "https://example.com/short"
    entry.summary = "Too short"  # Only 9 characters
    entry.published_parsed = time.struct_time((2024, 1, 1, 12, 0, 0, 0, 0, 0))
    entry.updated_parsed = None
    entry.content = None
    entry.description = None

    # Parse the article
    result = news_ingester.parse_article(entry)

    # Should return None for short content
    assert result is None


def test_is_relevant_matches_company_name(news_ingester):
    """Test that is_relevant matches company names, not just ticker symbols."""
    # Create article mentioning "Apple" but not "AAPL"
    article = RawArticle(
        ticker="AAPL",
        title="Apple Announces New Product Line",
        content="Apple Inc. has announced a new line of innovative products that will revolutionize the market.",
        published_date=date.today(),
        url="https://example.com/apple-news",
        source="Test Source",
    )

    # Should match on company name
    assert news_ingester.is_relevant(article, "AAPL") is True

    # Create article with no mention of Apple or AAPL
    unrelated_article = RawArticle(
        ticker="AAPL",
        title="General Market News",
        content="The stock market continues to show strong performance across various sectors.",
        published_date=date.today(),
        url="https://example.com/market-news",
        source="Test Source",
    )

    # Should not match
    assert news_ingester.is_relevant(unrelated_article, "AAPL") is False


def test_fetch_articles_stops_at_max_articles_per_ticker(news_ingester):
    """Test that fetch_articles stops once max_articles_per_ticker is reached."""
    ticker = "AAPL"
    news_ingester.max_articles_per_ticker = 3

    with patch("alphasignal.ingestion.news.feedparser.parse") as mock_parse:
        # Feed has 10 available articles, but the limit should cap results at 3
        mock_parse.return_value = mock_rss_feed(ticker, 10)

        results = news_ingester.fetch_articles(ticker)

    assert len(results) == 3


def test_parse_article_uses_content_field_when_no_summary(news_ingester):
    """Test that parse_article falls back to entry.content when summary is absent."""
    entry = MagicMock()
    entry.get = MagicMock(
        side_effect=lambda key, default=None: {
            "title": "Content Field Article",
            "link": "https://example.com/content-field",
        }.get(key, default)
    )
    entry.title = "Content Field Article"
    entry.link = "https://example.com/content-field"
    entry.published_parsed = time.struct_time((2024, 1, 1, 12, 0, 0, 0, 0, 0))
    entry.updated_parsed = None
    entry.summary = None
    entry.content = [
        {
            "value": "This is the article body coming from the content field, long enough to pass the minimum length check."
        }
    ]
    entry.description = None

    result = news_ingester.parse_article(entry)

    assert result is not None
    assert "article body coming from the content field" in result.content


def test_fetch_articles_handles_source_error_and_continues(news_ingester):
    """Test that an error formatting/fetching one source doesn't abort the whole run."""
    ticker = "AAPL"
    # url_template.format will raise KeyError because {bad_key} isn't a valid substitution
    news_ingester.sources = [
        {"name": "Broken Source", "url_template": "https://example.com/{bad_key}"},
    ]

    # Should not raise, should just log a warning and return no articles
    results = news_ingester.fetch_articles(ticker)

    assert results == []


def test_fetch_articles_handles_network_error(news_ingester):
    """Test that fetch_articles handles network errors gracefully."""
    ticker = "AAPL"

    # Mock feedparser to raise an exception
    with patch("alphasignal.ingestion.news.feedparser.parse") as mock_parse:
        mock_parse.side_effect = Exception("Network error")

        # Should return empty list, not raise
        results = news_ingester.fetch_articles(ticker)

        assert results == []
