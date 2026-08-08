"""Financial news ingestion module."""

import logging
from datetime import date, datetime, timedelta

import feedparser
from bs4 import BeautifulSoup

from alphasignal.ingestion import RawArticle

logger = logging.getLogger(__name__)

# Ticker to company name mapping
TICKER_TO_COMPANY = {
    "AAPL": ["Apple", "AAPL"],
    "MSFT": ["Microsoft", "MSFT"],
    "GOOGL": ["Google", "Alphabet", "GOOGL"],
    "AMZN": ["Amazon", "AMZN"],
    "NVDA": ["Nvidia", "NVDA"],
    "META": ["Meta", "Facebook", "META"],
    "TSLA": ["Tesla", "TSLA"],
    "JPM": ["JPMorgan", "JP Morgan", "JPM"],
    "GS": ["Goldman Sachs", "Goldman", "GS"],
    "MS": ["Morgan Stanley", "MS"],
}


class NewsIngester:
    """Ingests financial news from RSS feeds."""

    def __init__(self, config: dict):
        """Initialize news ingester.

        Args:
            config: Configuration dictionary containing ingestion.news settings
        """
        self.config = config

        # Get news config
        news_config = config.get("ingestion", {}).get("news", {})
        self.max_articles_per_ticker = news_config.get("max_articles_per_ticker", 50)
        self.max_age_days = news_config.get("max_age_days", 90)
        self.sources = news_config.get("sources", [])

    def fetch_articles(self, ticker: str) -> list[RawArticle]:
        """Fetch news articles for a ticker from RSS feeds.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of RawArticle objects
        """
        all_articles = []
        seen_urls = set()

        # Calculate cutoff date for filtering old articles
        cutoff_date = datetime.now() - timedelta(days=self.max_age_days)

        for source in self.sources:
            try:
                source_name = source.get("name", "Unknown")
                url_template = source.get("url_template", "")

                # Format URL with ticker
                url = url_template.format(ticker=ticker)

                logger.info(f"Fetching news from {source_name} for {ticker}")

                # Parse RSS feed
                feed = feedparser.parse(url)

                # Process each entry
                for entry in feed.entries:
                    # Parse article
                    article = self.parse_article(entry)
                    if not article:
                        continue

                    # Set ticker and source
                    article.ticker = ticker
                    article.source = source_name

                    # Filter by age
                    if (
                        datetime.combine(article.published_date, datetime.min.time())
                        < cutoff_date
                    ):
                        continue

                    # Filter by relevance
                    if not self.is_relevant(article, ticker):
                        continue

                    # Deduplicate by URL
                    if article.url in seen_urls:
                        continue

                    seen_urls.add(article.url)
                    all_articles.append(article)

                    # Check if we've hit the limit
                    if len(all_articles) >= self.max_articles_per_ticker:
                        break

                # Break outer loop if we've hit the limit
                if len(all_articles) >= self.max_articles_per_ticker:
                    break

            except Exception as e:
                logger.warning(f"Error fetching news from {source_name}: {e}")
                # Continue with next source, don't raise

        logger.info(f"Fetched {len(all_articles)} articles for {ticker}")
        return all_articles

    def parse_article(self, entry: feedparser.FeedParserDict) -> RawArticle | None:
        """Parse a feedparser entry into a RawArticle.

        Args:
            entry: Feedparser entry dict

        Returns:
            RawArticle or None if parsing fails or content is too short
        """
        try:
            # Extract title
            title = entry.get("title", "")

            # Extract link
            link = entry.get("link", "")

            # Extract published date
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            if published:
                published_date = date(
                    published.tm_year, published.tm_mon, published.tm_mday
                )
            else:
                published_date = date.today()

            # Extract content (try summary first, then content)
            content = ""
            if hasattr(entry, "summary") and entry.summary:
                content = entry.summary
            elif hasattr(entry, "content") and entry.content and len(entry.content) > 0:
                content = entry.content[0].get("value", "")
            elif hasattr(entry, "description") and entry.description:
                content = entry.description

            # Clean HTML from content
            if content:
                soup = BeautifulSoup(content, "lxml")
                content = soup.get_text()

                # Normalize whitespace
                content = " ".join(content.split())

            # Check minimum length
            if len(content) < 50:
                return None

            return RawArticle(
                ticker="",  # Will be set by caller
                title=title,
                content=content,
                published_date=published_date,
                url=link,
                source="",  # Will be set by caller
            )

        except Exception as e:
            logger.warning(f"Error parsing article: {e}")
            return None

    def is_relevant(self, article: RawArticle, ticker: str) -> bool:
        """Check if an article is relevant to a ticker.

        Args:
            article: RawArticle to check
            ticker: Ticker symbol

        Returns:
            True if article mentions ticker or company name
        """
        # Combine title and content for searching
        text = (article.title + " " + article.content).lower()

        # Get company names for this ticker
        company_names = TICKER_TO_COMPANY.get(ticker, [ticker])

        # Check if any company name appears in the text
        for name in company_names:
            if name.lower() in text:
                return True

        return False
