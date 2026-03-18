"""Pydantic models for AlphaSignal API requests and responses."""

from datetime import date
from pydantic import BaseModel, Field


# Request models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=500)
    ticker_filter: list[str] | None = None
    date_from: date | None = None
    date_to: date | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class IngestRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=5, pattern=r"^[A-Z]+$")
    filing_types: list[str] = Field(default=["10-K", "10-Q"])
    years_back: int = Field(default=2, ge=1, le=5)


# Response models
class Citation(BaseModel):
    chunk_id: str
    ticker: str
    source: str
    date: date
    excerpt: str
    relevance_score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    latency_ms: int
    retrieval_scores: list[float]
    model_used: str


class SentimentSignal(BaseModel):
    ticker: str
    date: date
    score: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: str
    doc_type: str
    key_positive: list[str]
    key_negative: list[str]
    summary: str


class SentimentResponse(BaseModel):
    ticker: str
    signals: list[SentimentSignal]
    latest_score: float | None = None
    latency_ms: int


class IngestResponse(BaseModel):
    ticker: str
    status: str
    chunks_created: int
    documents_processed: int
    latency_ms: int


class BatchIngestRequest(BaseModel):
    """Request for batch ingestion."""

    tickers: list[str]


class BatchIngestResponse(BaseModel):
    """Response for batch ingestion."""

    results: list[IngestResponse]
    total_latency_ms: int


class HealthResponse(BaseModel):
    status: str
    version: str
    faiss_index_loaded: bool
    sqlite_connected: bool
    chunks_indexed: int
    uptime_seconds: float


class ErrorResponse(BaseModel):
    error: str
    code: str
    detail: str | None = None
