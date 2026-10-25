"""Retrieval module for AlphaSignal."""

from dataclasses import dataclass, field
from datetime import date


@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk with scores."""

    chunk_id: str
    ticker: str
    text: str
    doc_type: str
    source: str
    section: str | None
    date: date
    url: str | None
    dense_score: float  # Cosine similarity from FAISS
    sparse_score: float  # BM25 score
    hybrid_score: float  # Weighted combination
    final_score: float | None = None  # After reranking
    accession_number: str | None = None  # SEC filing identity (provenance)
    attribution: str = ""  # filer / news_title / news_body / "" (unknown)
    content_hash: str = ""
    period_end: date | None = None  # filing's reporting period end (not filing date)
    # One-line source identity (form, reporting period, filing date) shown to
    # the reranker and the generator; "" when not built by the retriever.
    header: str = ""
    # "table" for financial-table chunks when the table lane is on, else
    # "prose"; match_columns are the table's column headings that denote the
    # requested period (empty: no period requested or none matched).
    lane: str = "prose"
    match_columns: list[str] = field(default_factory=list)
    basis: str = ""  # table chunks: the "Basis:" line (reported / PRO FORMA / ...)


__all__ = ["RetrievedChunk"]
