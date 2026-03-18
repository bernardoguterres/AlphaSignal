"""Ingestion package data schemas and exports."""

from dataclasses import dataclass
from datetime import date


@dataclass
class RawDocument:
    """Represents a raw SEC filing document."""
    ticker: str
    doc_type: str           # "10-K" or "10-Q"
    filing_date: date
    period_of_report: date
    source: str             # "SEC EDGAR"
    sections: dict[str, str]  # section_name → text
    file_path: str
    accession_number: str


@dataclass
class RawArticle:
    """Represents a raw financial news article."""
    ticker: str
    title: str
    content: str
    published_date: date
    url: str
    source: str             # "Yahoo Finance", "Reuters", etc.


@dataclass
class Chunk:
    """Represents a semantic chunk of text ready for embedding."""
    chunk_id: str           # deterministic: f"{ticker}_{source_hash}_{index:04d}"
    ticker: str
    text: str
    token_count: int
    doc_type: str           # "10-K", "10-Q", "news"
    source: str             # "SEC EDGAR" or feed name
    section: str | None     # e.g. "item_7" for MD&A, None for news
    date: date
    url: str | None         # for news articles
    chunk_index: int        # position in original document
    total_chunks: int       # total chunks from this document
