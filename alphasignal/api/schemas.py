"""Pydantic models for AlphaSignal API requests and responses."""

from datetime import date
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

# /sentiment response contract implemented by this service (docs/sentiment_contract.md).
SENTIMENT_CONTRACT_VERSION = "2"


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
        include_trace: Also return per-stage candidate ids (dense, sparse,
            merged pool sent to the reranker, reranked) for evaluation and
            debugging. Off by default.
    """

    query: str = Field(..., min_length=5, max_length=500)
    ticker_filter: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    include_trace: bool = False


class IngestRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=5, pattern=r"^[A-Z]+$")
    filing_types: list[str] = Field(default=["10-K", "10-Q"])
    years_back: int = Field(default=2, ge=1, le=5)
    include_news: bool = True  # False: re-ingest filings only


# Response models
class Citation(BaseModel):
    chunk_id: str
    ticker: str
    source: str
    date: date
    excerpt: str
    relevance_score: float
    # Provenance (2026-10): document identity a reader can follow back.
    doc_type: str | None = None
    url: str | None = None  # EDGAR filing index page / article URL
    accession_number: str | None = None
    attribution: str | None = None  # filer / news_title / news_body


class SentenceSupport(BaseModel):
    sentence: str
    sources: list[int]
    verdict: Literal[
        "supported_lexically", "weak_overlap", "number_not_in_source", "uncited_claim"
    ]
    missing_numbers: list[str] = []
    word_overlap: float | None = None


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
    # Whether dense retrieval actually contributed this call ("hybrid") or
    # the run degraded to BM25 only ("sparse_only"), plus candidate counts.
    retrieval: dict = {}
    # Lexical citation-support screening (generation/support.py) - a
    # heuristic flagging claims to inspect, not proof of support.
    citation_support: list[SentenceSupport] = []
    # Chunks passed to the generator (reranked top_k), in rank order - lets
    # retrieval be evaluated separately from what the answer chose to cite.
    retrieved_chunk_ids: list[str] = []
    # Per-stage candidate ids, only when the request set include_trace.
    trace: dict | None = None


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
    # Document identity / link (document-level aggregation, 2026-10)
    document_id: str | None = None
    url: str | None = None


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
    # "stale"   - documents exist, but none within the freshness window:
    #             latest_score is null; last_document_date says how old.
    status: Literal["ok", "no_data", "degraded", "stale"] = "ok"
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
    # --- Aggregation / freshness contract (2026-10) ---
    # latest_score is the document-level, recency-weighted aggregate (see
    # generation/sentiment.py), not a single chunk's score.
    method: str = "document_recency_weighted_v2"
    as_of: date | None = None
    freshness_days: int | None = None
    documents_used: int = 0
    last_document_date: date | None = None
    # What the corpus actually holds for this ticker (never implies
    # complete coverage): per doc_type earliest/latest dates + years absent.
    coverage: dict = {}


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
    # Service identity: clients (AlphaLive) check these before trusting the
    # URL they were given - a different app answering on the same port
    # returns 200 too, so status codes alone prove nothing.
    service: Literal["alphasignal"] = "alphasignal"
    sentiment_contract: str = SENTIMENT_CONTRACT_VERSION
    status: str
    version: str
    faiss_index_loaded: bool
    sqlite_connected: bool
    chunks_indexed: int
    uptime_seconds: float
    # Dense-retrieval readiness: vectors whose content hash is recorded
    # (only those can contribute dense candidates). "unavailable" means
    # every query runs BM25-only.
    vectors_total: int = 0
    vectors_verified: int = 0
    dense_retrieval: Literal["full", "partial", "unavailable"] = "unavailable"
    # Active corpus version (alphasignal.store.versioning): corpus_version
    # covers chunk content + provenance; corpus_consistent means the FAISS
    # index holds exactly the stored chunks (content_digest == index_digest).
    corpus_version: str | None = None
    content_digest: str | None = None
    index_digest: str | None = None
    corpus_consistent: bool | None = None
    chunker_versions: dict[str, int] = {}


class ErrorResponse(BaseModel):
    error: str
    code: str
    detail: str | None = None
