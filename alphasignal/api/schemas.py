"""Pydantic models for AlphaSignal API requests and responses."""

from datetime import date
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


# Request models
class QueryRequest(BaseModel):
    """Request body for the /query endpoint.

    Attributes:
        query: Natural-language question about a company or filing.
        ticker_filter: Optional single ticker to restrict retrieval to. Was
            previously typed as list[str], implying multi-ticker filtering,
            but the whole retrieval stack (HybridRetriever, VectorStore) only
            ever supports one ticker per query - the route silently used
            only the first list element and dropped the rest. Typed as a
            single str now to match what's actually supported.
        date_from: Earliest document date to include in retrieval.
        date_to: Latest document date to include in retrieval.
        top_k: Number of results to return (1–20, default 5).
    """

    query: str = Field(..., min_length=5, max_length=500)
    ticker_filter: str | None = None
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
    # `model_used` would normally conflict with Pydantic's protected `model_` namespace;
    # ConfigDict opt-out is required to suppress the UserWarning.
    model_config = ConfigDict(protected_namespaces=())

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
    # False when this signal came from a provider/parsing fallback rather
    # than a genuine chunk-level prediction (additive field, defaults True
    # so existing consumers reading only score/confidence are unaffected).
    reliable: bool = True


class SentimentResponse(BaseModel):
    ticker: str
    signals: list[SentimentSignal]
    latest_score: float | None = None
    latency_ms: int
    # Explicit, hard-to-miss "no data yet" signal (audit finding, 2026-07-14):
    # latest_score=None was already structurally correct for an empty
    # corpus, but nothing made this distinction hard to accidentally
    # collapse - a consuming client that does `score = latest_score or 0.0`
    # (or any similar null-coalescing) silently treats "never ingested" the
    # same as "genuinely neutral". data_available makes that distinction
    # explicit and impossible to miss in the schema itself. NOTE: this is
    # additive only - AlphaLive's client does not read this field yet and
    # needs a separate, cross-repo follow-up to actually consume it; until
    # then this field exists but doesn't change AlphaLive's behavior.
    data_available: bool = True
    # --- Degradation-visibility fields (2026-09-11) ---
    # All additive with backward-compatible defaults. Same AlphaLive caveat
    # as data_available above: not consumed cross-repo yet.
    #
    # "ok"       - genuine extraction result (including genuine neutral)
    # "no_data"  - no chunks were ingested for this ticker (data_available=False)
    # "degraded" - at least one chunk-level extraction fell back to a
    #              provider/parsing default instead of a real prediction
    # A Literal (not a bare str) so the finite set of valid values is a
    # structural, Pydantic-enforced guarantee - not just a documented
    # convention - and shows up as an enum in the generated OpenAPI schema.
    status: Literal["ok", "no_data", "degraded"] = "ok"
    degraded: bool = False
    # Categorical only, by construction (route code always passes a fixed
    # literal here, never str(exception)) - must never carry raw exception
    # text, stack traces, or provider/credential details.
    degradation_reason: (
        Literal["partial_extraction_failure", "full_extraction_failure"] | None
    ) = None
    # Chunks whose sentiment came from a genuine model prediction, out of
    # all chunks considered for this response.
    reliable_chunk_count: int = 0
    total_chunk_count: int = 0


class IngestResponse(BaseModel):
    ticker: str
    status: str
    chunks_created: int
    chunks_stored: int
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
