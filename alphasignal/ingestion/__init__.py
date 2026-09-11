"""Ingestion package data schemas and exports."""

import hashlib
from dataclasses import dataclass
from datetime import date


@dataclass
class RawDocument:
    """Represents a raw SEC filing document."""

    ticker: str
    doc_type: str  # "10-K" or "10-Q"
    filing_date: date
    period_of_report: date
    source: str  # "SEC EDGAR"
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
    source: str  # "Yahoo Finance", "Reuters", etc.


@dataclass
class Chunk:
    """Represents a semantic chunk of text ready for embedding."""

    chunk_id: str  # deterministic: f"{ticker}_{source_hash}_{index:04d}"
    ticker: str
    text: str
    token_count: int
    doc_type: str  # "10-K", "10-Q", "news"
    source: str  # "SEC EDGAR" or feed name
    section: str | None  # e.g. "item_7" for MD&A, None for news
    date: date
    url: str | None  # for news articles
    chunk_index: int  # position in original document
    total_chunks: int  # total chunks from this document
    # Deterministic content fingerprint (sha256 of `text`), auto-computed if
    # left blank. chunk_id encodes *source* identity (doc + section +
    # position), not content - two chunk_id-identical chunks whose text
    # differs (e.g. a filing re-ingested after an amendment) must be
    # detected as changed so stale embeddings/vectors get refreshed instead
    # of silently kept (audit finding, 2026-09-11).
    content_hash: str = ""
    # Exact source-document/article identity (the chunk_id prefix before
    # its trailing "_{index:04d}"), set explicitly by SemanticChunker.
    # Used for exact-match orphan-cleanup ownership lookups instead of
    # string-prefix/LIKE matching on chunk_id, which - even when correctly
    # escaped - is still a derived, indirect signal; a real persisted
    # column is unambiguous by construction (audit correction, 2026-09-11:
    # "prefer exact persisted source ownership fields over raw
    # string-prefix deletion where practical"). Left "" for chunks built
    # directly (tests, older rows) rather than via the chunker - callers
    # relying on ownership lookups must treat that as "unknown source,"
    # never silently matched.
    source_id: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
