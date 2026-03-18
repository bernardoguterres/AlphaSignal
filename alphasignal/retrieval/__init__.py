"""Retrieval module for AlphaSignal."""

from dataclasses import dataclass
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


__all__ = ["RetrievedChunk"]
