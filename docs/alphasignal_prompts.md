# AlphaSignal — Claude Code Session Prompts

Paste each prompt in full at the start of a new Claude Code session.
Always work from the project root. Each session assumes the previous is complete.

---

## SESSION 1 — Project Skeleton

```
I am building AlphaSignal: a standalone financial RAG system that ingests SEC EDGAR filings
and financial news, extracts sentiment signals, and exposes them via a FastAPI REST API.
It will eventually feed sentiment scores into AlphaLab (a separate backtesting platform)
as strategy features.

Your job in this session is to create the complete project skeleton. No business logic yet —
just structure, config, app wiring, and a passing test suite.

─── EXACT DIRECTORY STRUCTURE TO CREATE ───────────────────────────────────────────────────

alphasignal/
├── ingestion/
│   ├── __init__.py
│   ├── edgar.py           # stub only
│   ├── news.py            # stub only
│   ├── chunker.py         # stub only
│   └── pipeline.py        # stub only
├── embeddings/
│   ├── __init__.py
│   ├── embedder.py        # stub only
│   └── cache.py           # stub only
├── store/
│   ├── __init__.py
│   ├── vector_store.py    # stub only
│   └── metadata_store.py  # stub only
├── retrieval/
│   ├── __init__.py
│   ├── retriever.py       # stub only
│   ├── reranker.py        # stub only
│   └── evaluator.py       # stub only
├── generation/
│   ├── __init__.py
│   ├── generator.py       # stub only
│   └── sentiment.py       # stub only
├── api/
│   ├── __init__.py
│   ├── app.py             # FULLY IMPLEMENTED
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py      # FULLY IMPLEMENTED
│   │   ├── query.py       # stub returning 501 Not Implemented
│   │   ├── sentiment.py   # stub returning 501 Not Implemented
│   │   └── ingest.py      # stub returning 501 Not Implemented
│   └── schemas.py         # FULLY IMPLEMENTED — all Pydantic v2 models
├── evaluation/
│   ├── __init__.py
│   ├── golden_set.json    # empty array [] for now
│   └── run_eval.py        # stub only
├── monitoring/
│   ├── __init__.py
│   └── metrics.py         # stub only
├── tests/
│   ├── __init__.py
│   ├── conftest.py        # pytest fixtures: test client, temp dirs, mock config
│   ├── test_health.py     # FULLY IMPLEMENTED — must pass
│   ├── test_chunker.py    # stub with placeholder test
│   ├── test_retriever.py  # stub with placeholder test
│   ├── test_evaluator.py  # stub with placeholder test
│   ├── test_sentiment.py  # stub with placeholder test
│   └── test_api.py        # stub with placeholder test
├── scripts/
│   ├── build_corpus.py    # stub only
│   └── benchmark.py       # stub only
├── config.yaml            # FULLY IMPLEMENTED — see spec below
├── .env.example           # FULLY IMPLEMENTED
├── requirements.txt       # FULLY IMPLEMENTED — all dependencies pinned
├── .gitignore
└── README.md              # one-paragraph placeholder

─── CONFIG.YAML SPEC ───────────────────────────────────────────────────────────────────────

tickers:
  - AAPL
  - MSFT
  - GOOGL
  - AMZN
  - NVDA
  - META
  - TSLA
  - JPM
  - GS
  - MS

ingestion:
  edgar:
    filing_types: ["10-K", "10-Q"]
    years_back: 2
    rate_limit_delay: 0.5
  news:
    sources:
      - name: "Yahoo Finance"
        url_template: "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
      - name: "Reuters"
        url_template: "https://feeds.reuters.com/reuters/companyNews"
    max_articles_per_ticker: 50
    max_age_days: 90

chunking:
  target_tokens: 300
  min_tokens: 100
  max_tokens: 400
  overlap_tokens: 50

embeddings:
  model: "text-embedding-ada-002"
  batch_size: 100
  max_retries: 3
  retry_delay: 1.0

retrieval:
  dense_candidates: 50
  sparse_candidates: 50
  rerank_candidates: 20
  final_top_k: 5
  hybrid_weights:
    bm25: 0.4
    dense: 0.6

generation:
  model: "gpt-4o-mini"
  max_tokens: 1000
  temperature: 0.1
  sentiment_cache_hours: 24

storage:
  faiss_index_path: "data/faiss_index"
  sqlite_db_path: "data/metadata.db"
  embeddings_cache_path: "data/embeddings_cache"

api:
  host: "0.0.0.0"
  port: 8000
  reload: true

─── SCHEMAS.PY SPEC ────────────────────────────────────────────────────────────────────────

Implement all of these Pydantic v2 models in api/schemas.py:

# Request models
class QueryRequest(BaseModel):
    query: str                          # min length 5, max 500
    ticker_filter: list[str] | None     # optional: filter results to these tickers
    date_from: date | None              # optional: only retrieve docs after this date
    date_to: date | None                # optional: only retrieve docs before this date
    top_k: int = 5                      # 1–20

class IngestRequest(BaseModel):
    ticker: str                         # uppercase, 1-5 chars
    filing_types: list[str] = ["10-K", "10-Q"]
    years_back: int = 2                 # 1–5

# Response models
class Citation(BaseModel):
    chunk_id: str
    ticker: str
    source: str                         # e.g. "10-Q 2024-Q3" or "Reuters"
    date: date
    excerpt: str                        # first 200 chars of chunk text
    relevance_score: float

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    latency_ms: int
    retrieval_scores: list[float]
    model_used: str

class SentimentSignal(BaseModel):
    date: date
    score: float                        # -1.0 to 1.0
    confidence: float                   # 0.0 to 1.0
    source: str
    key_positive: list[str]
    key_negative: list[str]
    summary: str

class SentimentResponse(BaseModel):
    ticker: str
    signals: list[SentimentSignal]
    latest_score: float | None
    latency_ms: int

class IngestResponse(BaseModel):
    ticker: str
    status: str                         # "started", "completed", "failed"
    chunks_created: int
    documents_processed: int
    latency_ms: int

class HealthResponse(BaseModel):
    status: str                         # "healthy"
    version: str                        # "0.1.0"
    faiss_index_loaded: bool
    sqlite_connected: bool
    chunks_indexed: int
    uptime_seconds: float

class ErrorResponse(BaseModel):
    error: str
    code: str
    detail: str | None

─── APP.PY REQUIREMENTS ────────────────────────────────────────────────────────────────────

api/app.py must:
- Create FastAPI app with title="AlphaSignal", version="0.1.0"
- Load config.yaml on startup using pyyaml, store on app.state.config
- Load .env using python-dotenv
- Register all four routers with prefixes: /health, /query, /sentiment, /ingest
- Add startup event that logs: "AlphaSignal starting up", config summary, port
- Add CORS middleware (allow all origins for development)
- Add request timing middleware that logs method + path + status + duration_ms for every request
- Return ErrorResponse on unhandled exceptions (500)
- App must start with: uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

─── HEALTH ENDPOINT REQUIREMENTS ───────────────────────────────────────────────────────────

GET /health must return HealthResponse with:
- status: "healthy"
- version: "0.1.0"
- faiss_index_loaded: False (until vector store is implemented)
- sqlite_connected: False (until metadata store is implemented)
- chunks_indexed: 0
- uptime_seconds: actual seconds since app startup

─── TEST REQUIREMENTS ──────────────────────────────────────────────────────────────────────

tests/conftest.py must provide:
- client fixture: TestClient wrapping the FastAPI app
- tmp_data_dir fixture: temporary directory for FAISS/SQLite during tests
- mock_config fixture: config dict with test values (small batch sizes, test paths)

tests/test_health.py must include and PASS:
- test_health_returns_200: GET /health returns 200
- test_health_response_schema: response matches HealthResponse schema exactly
- test_health_status_healthy: response.status == "healthy"
- test_health_uptime_positive: response.uptime_seconds > 0

─── REQUIREMENTS.TXT — PIN THESE EXACT PACKAGES ────────────────────────────────────────────

fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
pydantic-settings==2.5.2
pyyaml==6.0.2
python-dotenv==1.0.1
openai==1.51.0
faiss-cpu==1.8.0
rank-bm25==0.2.2
sentence-transformers==3.1.1
sec-edgar-downloader==5.0.5
feedparser==6.0.11
beautifulsoup4==4.12.3
lxml==5.3.0
sqlmodel==0.0.21
tiktoken==0.7.0
numpy==1.26.4
httpx==0.27.2
pytest==8.3.3
pytest-asyncio==0.24.0
pytest-cov==5.0.0

─── .ENV.EXAMPLE ───────────────────────────────────────────────────────────────────────────

OPENAI_API_KEY=your_openai_api_key_here
ALPHASIGNAL_ENV=development

─── DEFINITION OF DONE ─────────────────────────────────────────────────────────────────────

Session 1 is complete when:
1. `uvicorn api.app:app --reload` starts without errors
2. `GET http://localhost:8000/health` returns valid HealthResponse JSON
3. `pytest tests/test_health.py -v` shows 4 passing tests
4. `pytest tests/ -v` runs without import errors (other tests may skip/xfail)
5. All __init__.py files exist and are importable
6. config.yaml loads correctly on startup (visible in startup logs)
```

---

## SESSION 2 — SEC EDGAR Ingestion

```
I am building AlphaSignal, a financial RAG system. The project skeleton from Session 1
is complete: FastAPI app starts, /health returns 200, all modules exist as stubs.

Your job in this session is to fully implement the SEC EDGAR ingestion pipeline.

─── CONTEXT ────────────────────────────────────────────────────────────────────────────────

The ingestion pipeline fetches SEC 10-K and 10-Q filings for a list of tickers,
cleans and parses the HTML/text content, and returns structured document objects
ready for chunking in the next session.

We use the `sec-edgar-downloader` library (already in requirements.txt).

─── IMPLEMENT: ingestion/edgar.py ──────────────────────────────────────────────────────────

class EDGARIngester:

    __init__(self, config: dict, download_dir: str = "data/edgar_raw")
        - Store config, set up download directory
        - Set rate_limit_delay from config (default 0.5s between requests)
        - Initialise a Downloader from sec_edgar_downloader

    fetch_filings(self, ticker: str, filing_types: list[str], years_back: int) -> list[RawDocument]
        - Download filings using Downloader for each filing_type
        - Respect rate_limit_delay between downloads
        - Return list of RawDocument objects (see schema below)
        - Log: ticker, filing_type, number of filings fetched
        - Handle: ticker not found, network errors, empty results — never raise, always return []

    parse_filing(self, filing_path: Path) -> str
        - Read filing file (HTML or TXT)
        - If HTML: use BeautifulSoup with lxml parser
          - Remove: <script>, <style>, <table> tags (tables are often just formatting)
          - Extract text from remaining elements
          - Normalise whitespace (collapse multiple spaces/newlines to single)
        - If TXT: strip SEC header boilerplate (lines before "FORM TYPE:" section)
        - Return cleaned text string
        - If text < 500 characters after cleaning: return "" (malformed filing)

    extract_sections(self, text: str, doc_type: str) -> dict[str, str]
        - For 10-K filings, attempt to identify these sections by regex on headers:
            "item_1": "Business"
            "item_1a": "Risk Factors"
            "item_7": "Management's Discussion and Analysis"
            "item_7a": "Quantitative and Qualitative Disclosures About Market Risk"
            "item_8": "Financial Statements"
        - For 10-Q filings:
            "item_1": "Financial Statements"
            "item_2": "Management's Discussion and Analysis"
            "item_3": "Quantitative and Qualitative Disclosures"
            "item_4": "Controls and Procedures"
        - If section extraction fails (filing doesn't follow standard format):
            return {"full_text": text}  — do not raise
        - Each section value: cleaned text of that section only

─── DATA SCHEMAS (add to ingestion/__init__.py) ─────────────────────────────────────────────

@dataclass
class RawDocument:
    ticker: str
    doc_type: str           # "10-K" or "10-Q"
    filing_date: date
    period_of_report: date
    source: str             # "SEC EDGAR"
    sections: dict[str, str]  # section_name → text
    file_path: str
    accession_number: str

─── IMPLEMENT: ingestion/pipeline.py (partial — EDGAR only) ────────────────────────────────

class IngestionPipeline:

    __init__(self, config: dict)
        - Instantiate EDGARIngester
        - Set up data directories

    ingest_ticker_edgar(self, ticker: str) -> list[RawDocument]
        - Call EDGARIngester.fetch_filings with config values
        - Log start, progress, completion
        - Return list of RawDocument

─── IMPLEMENT: api/routes/ingest.py (EDGAR only for now) ───────────────────────────────────

POST /ingest/{ticker}
    - Accept IngestRequest body
    - Call pipeline.ingest_ticker_edgar(ticker)
    - Return IngestResponse with:
        status: "completed"
        documents_processed: len(raw_docs)
        chunks_created: 0  (chunking comes in Session 4)
        latency_ms: actual ms
    - On error: return IngestResponse with status: "failed", include error in detail

─── IMPLEMENT: tests/test_edgar.py ─────────────────────────────────────────────────────────

Create this test file with the following tests.
Use pytest-mock or unittest.mock to mock sec_edgar_downloader — do NOT make real network calls in tests.

test_fetch_filings_returns_raw_documents
    - Mock Downloader.get() to return without error
    - Mock filing files on disk with realistic SEC HTML content
    - Assert fetch_filings returns list[RawDocument] with correct ticker and doc_type

test_parse_filing_strips_html_tags
    - Create temp HTML file with <script>, <style>, and paragraph content
    - Assert parse_filing returns only the paragraph text, no HTML tags

test_parse_filing_handles_malformed_html
    - Create temp file with only 100 chars of gibberish
    - Assert parse_filing returns "" (below 500 char threshold)

test_extract_sections_10k_standard_format
    - Create text with standard 10-K section headers ("Item 1.", "Item 1A.", "Item 7.")
    - Assert extract_sections returns dict with expected keys populated

test_extract_sections_falls_back_to_full_text
    - Create text with no recognisable section headers
    - Assert extract_sections returns {"full_text": text}

test_fetch_filings_handles_network_error
    - Mock Downloader.get() to raise requests.exceptions.RequestException
    - Assert fetch_filings returns [] without raising

test_ingest_endpoint_returns_200
    - Mock IngestionPipeline.ingest_ticker_edgar to return 3 RawDocuments
    - POST /ingest/AAPL with valid IngestRequest
    - Assert 200, documents_processed == 3

─── DEFINITION OF DONE ─────────────────────────────────────────────────────────────────────

Session 2 is complete when:
1. `pytest tests/test_edgar.py -v` shows 7 passing tests
2. `pytest tests/ -v` still shows Session 1 health tests passing
3. EDGARIngester can be instantiated without errors
4. POST /ingest/{ticker} endpoint returns valid IngestResponse
5. No hardcoded API keys or file paths — everything from config.yaml
```

---

## SESSION 3 — News RSS Ingestion

```
I am building AlphaSignal, a financial RAG system. Sessions 1 and 2 are complete:
FastAPI skeleton works, SEC EDGAR ingestion is implemented and tested.

Your job in this session is to implement the financial news RSS ingestion pipeline.

─── CONTEXT ────────────────────────────────────────────────────────────────────────────────

We ingest financial news from RSS feeds (Yahoo Finance, Reuters) for configured tickers.
News articles complement SEC filings: filings are authoritative but infrequent (quarterly),
news is noisy but current (daily). Together they give the RAG system both depth and recency.

We use `feedparser` (already in requirements.txt).

─── IMPLEMENT: ingestion/news.py ───────────────────────────────────────────────────────────

class NewsIngester:

    __init__(self, config: dict)
        - Store config
        - Set max_articles_per_ticker from config
        - Set max_age_days from config (filter out old articles)

    fetch_articles(self, ticker: str) -> list[RawArticle]
        - For each source in config.ingestion.news.sources:
            - Format URL using url_template.format(ticker=ticker)
            - Fetch and parse RSS feed using feedparser
            - Extract articles: title, content/summary, published_date, url, source_name
            - Filter: only articles mentioning the ticker symbol in title OR content
            - Filter: only articles within max_age_days of today
            - Limit: max_articles_per_ticker total across all sources
        - Deduplicate by URL
        - Return list of RawArticle
        - On any network error: log warning, return partial results (never raise)

    parse_article(self, entry: feedparser.FeedParserDict) -> RawArticle | None
        - Extract: title, link, published (parse to date), summary/content
        - Clean HTML from content using BeautifulSoup
        - If content < 50 characters: return None (too short to be useful)
        - Return RawArticle

    is_relevant(self, article: RawArticle, ticker: str) -> bool
        - Return True if ticker appears in title or content (case-insensitive)
        - Also return True for common company names mapped from ticker:
            AAPL → ["Apple", "AAPL"]
            MSFT → ["Microsoft", "MSFT"]
            GOOGL → ["Google", "Alphabet", "GOOGL"]
            AMZN → ["Amazon", "AMZN"]
            NVDA → ["Nvidia", "NVDA"]
            META → ["Meta", "Facebook", "META"]
            TSLA → ["Tesla", "TSLA"]
            JPM → ["JPMorgan", "JP Morgan", "JPM"]
            GS → ["Goldman Sachs", "Goldman", "GS"]
            MS → ["Morgan Stanley", "MS"]
        - For unknown tickers: only match ticker symbol exactly

─── DATA SCHEMAS (add to ingestion/__init__.py) ─────────────────────────────────────────────

@dataclass
class RawArticle:
    ticker: str
    title: str
    content: str
    published_date: date
    url: str
    source: str             # "Yahoo Finance", "Reuters", etc.

─── UPDATE: ingestion/pipeline.py ──────────────────────────────────────────────────────────

Add to IngestionPipeline:

    ingest_ticker_news(self, ticker: str) -> list[RawArticle]
        - Call NewsIngester.fetch_articles(ticker)
        - Log article count
        - Return articles

    ingest_ticker(self, ticker: str) -> tuple[list[RawDocument], list[RawArticle]]
        - Call both ingest_ticker_edgar and ingest_ticker_news
        - Return both results
        - Log total: X filings, Y articles ingested for {ticker}

─── UPDATE: api/routes/ingest.py ───────────────────────────────────────────────────────────

Update POST /ingest/{ticker} to:
    - Call pipeline.ingest_ticker() (both EDGAR + news)
    - Return IngestResponse with:
        documents_processed: len(raw_docs) + len(articles)
        chunks_created: 0 (still — chunking in Session 4)

─── IMPLEMENT: tests/test_news.py ──────────────────────────────────────────────────────────

Mock feedparser.parse() throughout — no real network calls.

Build a helper fixture: mock_rss_feed(ticker, n_articles) that returns a realistic
feedparser result dict with n_articles entries, all mentioning the ticker.

test_fetch_articles_returns_raw_articles
    - Mock feed with 10 AAPL articles
    - Assert returns list of RawArticle with ticker="AAPL"

test_fetch_articles_filters_by_age
    - Mock feed with 5 recent + 5 old articles (older than max_age_days)
    - Assert only 5 recent articles returned

test_fetch_articles_deduplicates_by_url
    - Mock feed with 3 articles where 2 have the same URL
    - Assert only 2 articles returned

test_fetch_articles_filters_irrelevant
    - Mock feed with 5 AAPL articles and 3 articles about unrelated topics
    - Assert only AAPL-relevant articles returned

test_parse_article_cleans_html
    - Create feedparser entry with HTML-heavy content
    - Assert returned RawArticle.content has no HTML tags

test_parse_article_returns_none_for_short_content
    - Create feedparser entry with 30-character content
    - Assert parse_article returns None

test_is_relevant_matches_company_name
    - Create article mentioning "Apple" but not "AAPL"
    - Assert is_relevant(article, "AAPL") returns True

test_fetch_articles_handles_network_error
    - Mock feedparser.parse() to raise Exception
    - Assert fetch_articles returns [] without raising

─── DEFINITION OF DONE ─────────────────────────────────────────────────────────────────────

Session 3 is complete when:
1. `pytest tests/test_news.py -v` shows 8 passing tests
2. `pytest tests/ -v` shows all previous tests still passing
3. NewsIngester can be instantiated and called without errors
4. POST /ingest/{ticker} returns combined EDGAR + news document count
5. Ticker-to-company-name mapping is in a constant (not buried in logic)
```

---

## SESSION 4 — Semantic Chunking

```
I am building AlphaSignal, a financial RAG system. Sessions 1–3 are complete:
FastAPI skeleton, SEC EDGAR ingestion, and news RSS ingestion are implemented and tested.

Your job in this session is to implement the semantic chunking pipeline that converts
raw documents and articles into consistently-sized, well-tagged chunks ready for embedding.

─── CONTEXT ────────────────────────────────────────────────────────────────────────────────

Chunking strategy is one of the most impactful decisions in a RAG system.
Naive fixed-token splitting ignores document structure and cuts sentences mid-thought.
Our chunker must:
  - Respect document structure (sections for filings, paragraphs for articles)
  - Produce chunks in the 200–400 token range
  - Never cut a sentence in the middle
  - Tag every chunk with rich metadata for filtered retrieval later
  - Overlap adjacent chunks by ~50 tokens to preserve context at boundaries

We use `tiktoken` for token counting (already in requirements.txt).

─── DATA SCHEMA (add to ingestion/__init__.py) ──────────────────────────────────────────────

@dataclass
class Chunk:
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

─── IMPLEMENT: ingestion/chunker.py ────────────────────────────────────────────────────────

class SemanticChunker:

    __init__(self, config: dict)
        - Store config chunking params: target_tokens, min_tokens, max_tokens, overlap_tokens
        - Initialise tiktoken encoder: cl100k_base (same tokenizer as ada-002)

    count_tokens(self, text: str) -> int
        - Return number of tokens in text using tiktoken

    split_into_sentences(self, text: str) -> list[str]
        - Split text into sentences using punctuation rules:
            - Split on ". " followed by capital letter
            - Split on ".\n"
            - Split on "? " and "! "
            - Do NOT split on: "U.S.", "e.g.", "i.e.", "etc.", decimal numbers like "3.14"
        - Filter out sentences shorter than 10 characters
        - Return list of sentence strings

    chunk_text(self, text: str, overlap_tokens: int = None) -> list[str]
        - Core chunking logic:
            1. Split into sentences
            2. Greedily accumulate sentences until approaching max_tokens
            3. When adding next sentence would exceed max_tokens: save current chunk, start new
            4. New chunk starts with last overlap_tokens worth of sentences from previous chunk
            5. If a single sentence exceeds max_tokens: split it at max_tokens boundary
               (this is the only place we do hard token-count splits)
        - Never produce a chunk with token_count < min_tokens unless it's the last chunk
          and the document is short
        - Return list of text strings

    chunk_document(self, doc: RawDocument) -> list[Chunk]
        - For each section in doc.sections:
            - Call chunk_text on section text
            - Create Chunk objects with metadata from doc + section name
        - Assign sequential chunk_index across all sections
        - Generate deterministic chunk_id:
            import hashlib
            source_hash = hashlib.md5(f"{doc.ticker}{doc.filing_date}{doc.accession_number}".encode()).hexdigest()[:8]
            chunk_id = f"{doc.ticker.lower()}_{doc.doc_type.lower().replace('-','')}_{source_hash}_{index:04d}"
        - Return list of Chunk

    chunk_article(self, article: RawArticle) -> list[Chunk]
        - If article.content token_count <= max_tokens: return as single Chunk
        - Else: call chunk_text and create multiple Chunks
        - chunk_id: use URL hash
            source_hash = hashlib.md5(article.url.encode()).hexdigest()[:8]
            chunk_id = f"{article.ticker.lower()}_news_{source_hash}_{index:04d}"
        - Return list of Chunk

─── UPDATE: ingestion/pipeline.py ──────────────────────────────────────────────────────────

Add to IngestionPipeline:

    __init__: also instantiate SemanticChunker

    chunk_documents(self, docs: list[RawDocument]) -> list[Chunk]
        - Call chunker.chunk_document for each doc
        - Log: total chunks created from X documents

    chunk_articles(self, articles: list[RawArticle]) -> list[Chunk]
        - Call chunker.chunk_article for each article
        - Log: total chunks created from X articles

    ingest_and_chunk_ticker(self, ticker: str) -> list[Chunk]
        - Call ingest_ticker to get docs + articles
        - Call chunk_documents + chunk_articles
        - Return combined list of chunks
        - Log: ticker, total chunks, breakdown by doc_type

─── UPDATE: api/routes/ingest.py ───────────────────────────────────────────────────────────

Update POST /ingest/{ticker}:
    - Call pipeline.ingest_and_chunk_ticker(ticker)
    - Return IngestResponse with:
        chunks_created: len(chunks)
        documents_processed: actual count

─── IMPLEMENT: tests/test_chunker.py ───────────────────────────────────────────────────────

test_chunk_text_respects_max_tokens
    - Create text of ~1200 tokens
    - Assert all chunks have token_count <= max_tokens (400)

test_chunk_text_respects_min_tokens
    - Create text of ~1000 tokens
    - Assert all chunks except possibly the last have token_count >= min_tokens (100)

test_chunk_text_overlap_preserves_context
    - Create text with 10 clear sentences
    - Assert consecutive chunks share at least one sentence (from overlap)

test_chunk_text_does_not_split_sentences
    - Create text with 10 sentences, each exactly 30 tokens
    - Assert no chunk starts or ends in the middle of a sentence

test_split_sentences_handles_abbreviations
    - Input: "The U.S. economy grew 3.5% in Q3. Analysts were surprised."
    - Assert splits into exactly 2 sentences (not 4 at each ".")

test_chunk_document_assigns_correct_metadata
    - Create a RawDocument with ticker="AAPL", doc_type="10-K", two sections
    - Assert all returned Chunks have ticker="AAPL", doc_type="10-K"
    - Assert chunk_ids are unique across all chunks
    - Assert chunk_index is sequential

test_chunk_document_deterministic_ids
    - Chunk the same RawDocument twice
    - Assert chunk_ids are identical both times

test_chunk_article_single_chunk_for_short_article
    - Create RawArticle with 200-token content
    - Assert chunk_article returns exactly 1 Chunk

test_count_tokens_consistent_with_tiktoken
    - Assert count_tokens("hello world") == 2
    - Assert count_tokens("") == 0

─── DEFINITION OF DONE ─────────────────────────────────────────────────────────────────────

Session 4 is complete when:
1. `pytest tests/test_chunker.py -v` shows 9 passing tests
2. `pytest tests/ -v` shows all previous tests still passing
3. POST /ingest/{ticker} returns non-zero chunks_created (using mocked ingestion)
4. Chunk IDs are deterministic (same input → same output every time)
5. No chunk in any test has token_count > max_tokens (400)
```

---

## SESSION 5 — Embeddings and Storage

```
I am building AlphaSignal, a financial RAG system. Sessions 1–4 are complete:
skeleton, EDGAR ingestion, news ingestion, and semantic chunking are implemented and tested.

Your job in this session is to implement the embedding pipeline, FAISS vector store,
and SQLite metadata store. After this session, we can ingest → chunk → embed → store
a full end-to-end pass for any ticker.

─── IMPLEMENT: embeddings/cache.py ─────────────────────────────────────────────────────────

class EmbeddingCache:
    """Persistent cache mapping chunk_id → embedding vector to avoid re-embedding."""

    __init__(self, cache_path: str)
        - Load existing cache from disk (pickle or numpy .npz) if it exists
        - Store as dict[str, np.ndarray]

    get(self, chunk_id: str) -> np.ndarray | None

    set(self, chunk_id: str, embedding: np.ndarray)

    get_many(self, chunk_ids: list[str]) -> tuple[dict[str, np.ndarray], list[str]]
        - Returns (cached_embeddings, uncached_chunk_ids)

    save(self)
        - Persist cache to disk

    __len__(self) -> int

─── IMPLEMENT: embeddings/embedder.py ──────────────────────────────────────────────────────

class Embedder:

    __init__(self, config: dict, cache: EmbeddingCache)
        - Store config: model name, batch_size, max_retries, retry_delay
        - Initialise openai.OpenAI client (reads OPENAI_API_KEY from env)
        - Store cache

    embed_texts(self, texts: list[str]) -> np.ndarray
        - Embed list of texts using OpenAI ada-002
        - Process in batches of batch_size
        - For each batch: call client.embeddings.create()
        - On rate limit (429) or server error (5xx): exponential backoff retry
            delay = retry_delay * (2 ** attempt), max 3 retries
        - Return np.ndarray of shape (len(texts), 1536)

    embed_chunks(self, chunks: list[Chunk]) -> dict[str, np.ndarray]
        - Check cache first: skip already-embedded chunks
        - Embed only uncached chunks
        - Save new embeddings to cache
        - Return dict mapping chunk_id → embedding for ALL chunks (cached + new)
        - Log: X chunks embedded, Y served from cache

─── IMPLEMENT: store/metadata_store.py ─────────────────────────────────────────────────────

Use SQLModel for the ORM.

class ChunkRecord(SQLModel, table=True):
    chunk_id: str = Field(primary_key=True)
    ticker: str = Field(index=True)
    text: str
    token_count: int
    doc_type: str = Field(index=True)
    source: str
    section: str | None
    date: date = Field(index=True)
    url: str | None
    chunk_index: int
    total_chunks: int
    created_at: datetime = Field(default_factory=datetime.utcnow)

class MetadataStore:

    __init__(self, db_path: str)
        - Create SQLite database and tables if not exist
        - Store engine

    add_chunks(self, chunks: list[Chunk])
        - Bulk insert ChunkRecords, skip duplicates (chunk_id conflict = update)

    get_chunk(self, chunk_id: str) -> Chunk | None

    get_chunks_by_ticker(self, ticker: str, doc_type: str | None = None) -> list[Chunk]

    get_chunks_by_date_range(self, start: date, end: date, ticker: str | None = None) -> list[Chunk]

    get_all_chunk_ids(self) -> list[str]

    count(self) -> int

─── IMPLEMENT: store/vector_store.py ───────────────────────────────────────────────────────

class VectorStore:

    __init__(self, index_path: str, dim: int = 1536)
        - Set index_path and dim
        - self.index = None  (loaded lazily)
        - self.chunk_ids: list[str] = []  (parallel list: position → chunk_id)

    _create_index(self) -> faiss.Index
        - Return faiss.IndexFlatIP(self.dim)
        - (Inner product on normalised vectors = cosine similarity)

    load(self)
        - If index file exists: load with faiss.read_index, load chunk_ids from json
        - Else: create fresh index
        - Log: loaded X vectors from disk OR created fresh index

    save(self)
        - faiss.write_index to {index_path}/index.faiss
        - Save chunk_ids to {index_path}/chunk_ids.json

    add(self, embeddings: np.ndarray, chunk_ids: list[str])
        - Normalise embeddings: vectors / ||vectors||
        - self.index.add(normalised)
        - self.chunk_ids.extend(chunk_ids)
        - Call save() after adding

    search(self, query_embedding: np.ndarray, k: int = 20,
           filter_ids: set[str] | None = None) -> list[tuple[str, float]]
        - Normalise query_embedding
        - Search index for top min(k * 3, len(chunk_ids)) candidates
          (oversample to allow for filtering)
        - Filter by filter_ids if provided
        - Return top k as list of (chunk_id, score) tuples, score in [0, 1]

    __len__(self) -> int
        - Return self.index.ntotal if index loaded else 0

─── UPDATE: ingestion/pipeline.py ──────────────────────────────────────────────────────────

Update IngestionPipeline:

    __init__: also instantiate Embedder, MetadataStore, VectorStore
        - Call vector_store.load() on startup

    store_chunks(self, chunks: list[Chunk], embeddings: dict[str, np.ndarray])
        - metadata_store.add_chunks(chunks)
        - vector_store.add(embeddings_array, chunk_ids)
        - Log: stored X chunks

    full_ingest(self, ticker: str) -> IngestResult
        - ingest_and_chunk_ticker → chunks
        - embedder.embed_chunks → embeddings
        - store_chunks(chunks, embeddings)
        - Return dataclass with counts

─── UPDATE: api/routes/health.py ───────────────────────────────────────────────────────────

Update GET /health:
    - faiss_index_loaded: True if vector_store is loaded and index exists
    - sqlite_connected: True if metadata_store is connected
    - chunks_indexed: metadata_store.count()

─── UPDATE: api/routes/ingest.py ───────────────────────────────────────────────────────────

Update POST /ingest/{ticker}:
    - Call pipeline.full_ingest(ticker)
    - Mock OpenAI calls during tests using pytest monkeypatch

─── IMPLEMENT: tests/test_store.py ─────────────────────────────────────────────────────────

test_vector_store_add_and_search
    - Create VectorStore with tmp path
    - Add 10 random 1536-dim embeddings
    - Search with one of them as query
    - Assert top result is that chunk_id with score > 0.99

test_vector_store_persists_to_disk
    - Add 5 vectors, call save(), create new VectorStore instance, call load()
    - Assert len(new_store) == 5

test_vector_store_normalises_embeddings
    - Add un-normalised embeddings (large magnitudes)
    - Search and assert scores are in [0, 1] range

test_metadata_store_add_and_retrieve
    - Create MetadataStore with tmp SQLite path
    - Add 5 chunks with ticker="AAPL"
    - Assert get_chunks_by_ticker("AAPL") returns 5 chunks

test_metadata_store_deduplicates
    - Add same chunk twice (same chunk_id)
    - Assert count() == 1

test_embedding_cache_hit_and_miss
    - Set embedding for "chunk_001"
    - get_many(["chunk_001", "chunk_002"])
    - Assert cached == {"chunk_001": ...}, uncached == ["chunk_002"]

test_embedder_uses_cache
    - Pre-populate cache with 5 chunk_ids
    - Call embed_chunks with those 5 + 2 new chunks
    - Assert OpenAI API called only once (for 2 new chunks), not for cached 5
    - Use unittest.mock to assert call count

─── DEFINITION OF DONE ─────────────────────────────────────────────────────────────────────

Session 5 is complete when:
1. `pytest tests/test_store.py -v` shows 7 passing tests
2. `pytest tests/ -v` shows all previous tests passing
3. GET /health returns faiss_index_loaded: true and chunks_indexed > 0
   after calling POST /ingest (with mocked OpenAI)
4. Embedding cache correctly skips re-embedding on second ingest call
5. FAISS index survives app restart (load from disk)
```

---

## SESSION 6 — Hybrid Retrieval Pipeline

```
I am building AlphaSignal, a financial RAG system. Sessions 1–5 are complete:
skeleton, ingestion, chunking, and embedding/storage are implemented and tested.

Your job in this session is to implement the full retrieval pipeline:
BM25 sparse retrieval + FAISS dense retrieval, combined and reranked with a cross-encoder.

─── CONTEXT ────────────────────────────────────────────────────────────────────────────────

Pure dense retrieval misses exact keyword matches (e.g. "operating margin Q3 2024").
Pure BM25 misses semantic similarity (e.g. query "profitability" vs chunk "net income").
Hybrid retrieval combines both. Reranking with a cross-encoder re-scores the top candidates
using full query-document attention, significantly improving precision.

─── IMPLEMENT: retrieval/retriever.py ──────────────────────────────────────────────────────

class HybridRetriever:

    __init__(self, vector_store: VectorStore, metadata_store: MetadataStore,
             embedder: Embedder, config: dict)
        - Store all components
        - self._bm25: BM25Okapi | None = None
        - self._bm25_chunk_ids: list[str] = []
        - Store retrieval config: dense_candidates, sparse_candidates,
          rerank_candidates, final_top_k, hybrid_weights

    build_bm25_index(self)
        - Load all chunk texts from metadata_store
        - Tokenize each text (simple whitespace tokenization + lowercase)
        - Build BM25Okapi index from rank_bm25
        - Store parallel chunk_ids list
        - Log: built BM25 index over X chunks

    _dense_search(self, query_embedding: np.ndarray,
                  k: int, filter_chunk_ids: set[str] | None) -> list[tuple[str, float]]
        - vector_store.search(query_embedding, k, filter_chunk_ids)
        - Return list of (chunk_id, score)

    _sparse_search(self, query: str,
                   k: int, filter_chunk_ids: set[str] | None) -> list[tuple[str, float]]
        - Tokenize query
        - Get BM25 scores for all chunks
        - If filter_chunk_ids: zero out scores for non-matching chunks
        - Return top k as (chunk_id, normalised_score) where normalised = score / max_score

    _merge_results(self, dense: list[tuple[str, float]],
                   sparse: list[tuple[str, float]]) -> list[tuple[str, float]]
        - Combine by chunk_id (union)
        - hybrid_score = bm25_weight * sparse_norm + dense_weight * dense_norm
        - Return sorted by hybrid_score descending

    retrieve(self, query: str,
             ticker_filter: list[str] | None = None,
             date_from: date | None = None,
             date_to: date | None = None,
             top_k: int | None = None) -> list[RetrievedChunk]
        - Build filter_chunk_ids set from metadata filters (if any filters specified)
        - Embed query using embedder
        - dense_results = _dense_search(embedding, dense_candidates, filter_chunk_ids)
        - sparse_results = _sparse_search(query, sparse_candidates, filter_chunk_ids)
        - merged = _merge_results(dense_results, sparse_results)
        - Fetch chunk texts from metadata_store for top rerank_candidates
        - Return list of RetrievedChunk objects (before reranking — reranking in reranker.py)
        - Log: query[:50], filter summary, candidate counts

─── DATA SCHEMA ────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk: Chunk
    dense_score: float | None
    sparse_score: float | None
    hybrid_score: float
    final_score: float | None   # set by reranker, None until reranked

─── IMPLEMENT: retrieval/reranker.py ───────────────────────────────────────────────────────

class CrossEncoderReranker:

    __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2")
        - Load CrossEncoder from sentence_transformers
        - Log: loaded reranker model

    rerank(self, query: str, candidates: list[RetrievedChunk],
           top_k: int) -> list[RetrievedChunk]
        - Create (query, chunk.text) pairs for each candidate
        - Call model.predict(pairs) to get relevance scores
        - Set candidate.final_score = score
        - Sort by final_score descending
        - Return top_k candidates

─── IMPLEMENT: retrieval/evaluator.py ──────────────────────────────────────────────────────

class RetrievalEvaluator:

    __init__(self, retriever: HybridRetriever, reranker: CrossEncoderReranker,
             metadata_store: MetadataStore)

    load_golden_set(self, path: str) -> list[dict]
        - Load evaluation/golden_set.json
        - Return list of {id, question, ticker, relevant_chunk_ids, answer_summary}

    evaluate(self, golden_set: list[dict], top_k: int = 10) -> EvalResults
        - For each question in golden_set:
            - retrieve(question, ticker_filter=[ticker]) → candidates
            - rerank(question, candidates, top_k) → ranked
            - Compute: reciprocal_rank, dcg, hit@3
        - Aggregate: MRR@10, NDCG@5, Hit@3
        - Return EvalResults dataclass

    compute_mrr(self, ranked_ids: list[str], relevant_ids: list[str]) -> float
        - Return 1/rank of first relevant result, or 0 if not found in top 10

    compute_ndcg(self, ranked_ids: list[str], relevant_ids: list[str], k: int = 5) -> float
        - Standard NDCG@k formula
        - Binary relevance: 1 if in relevant_ids, 0 otherwise

    compute_hit_at_k(self, ranked_ids: list[str], relevant_ids: list[str], k: int = 3) -> float
        - Return 1.0 if any relevant_id in top k, else 0.0

@dataclass
class EvalResults:
    mrr_at_10: float
    ndcg_at_5: float
    hit_at_3: float
    num_questions: int
    per_question: list[dict]  # detailed breakdown

─── IMPLEMENT: tests/test_retriever.py ─────────────────────────────────────────────────────

Use fixtures with pre-populated in-memory stores (no real embeddings — use random vectors).

test_hybrid_retriever_returns_results
    - Populate stores with 20 chunks across 3 tickers
    - Call retrieve("revenue growth") 
    - Assert returns list of RetrievedChunk, len <= final_top_k * 3

test_ticker_filter_limits_results
    - Populate with 10 AAPL + 10 MSFT chunks
    - retrieve("earnings", ticker_filter=["AAPL"])
    - Assert all returned chunks have ticker == "AAPL"

test_date_filter_limits_results
    - Populate with chunks dated 2023 and 2024
    - retrieve("guidance", date_from=date(2024,1,1))
    - Assert all returned chunks have date >= 2024-01-01

test_merge_results_combines_scores_correctly
    - Create mock dense=[(id1, 0.9), (id2, 0.7)] sparse=[(id2, 0.8), (id3, 0.6)]
    - Call _merge_results
    - Assert id2 gets hybrid score from both, id1 and id3 get score from one source only

test_reranker_sorts_by_final_score
    - Create 5 RetrievedChunk objects with known hybrid_scores
    - Call rerank with a query
    - Assert returned order reflects final_score (not hybrid_score)

test_evaluator_mrr_calculation
    - ranked_ids = ["a", "b", "c", "d"], relevant_ids = ["c"]
    - Assert compute_mrr == 1/3

test_evaluator_ndcg_calculation
    - ranked_ids = ["a", "b", "c"], relevant_ids = ["a", "c"]
    - Assert compute_ndcg is between 0 and 1

test_evaluator_hit_at_k
    - ranked_ids = ["a", "b", "c", "d"], relevant_ids = ["d"]
    - Assert compute_hit_at_k(k=3) == 0.0
    - Assert compute_hit_at_k(k=4) == 1.0

─── DEFINITION OF DONE ─────────────────────────────────────────────────────────────────────

Session 6 is complete when:
1. `pytest tests/test_retriever.py -v` shows 8 passing tests
2. `pytest tests/ -v` shows all previous tests passing
3. HybridRetriever.retrieve() returns results filtered correctly by ticker and date
4. CrossEncoderReranker loads the model without errors
5. EvalResults dataclass has all three metrics: mrr_at_10, ndcg_at_5, hit_at_3
```

---

## SESSION 7 — Generation and Sentiment Extraction

```
I am building AlphaSignal, a financial RAG system. Sessions 1–6 are complete:
skeleton, ingestion, chunking, storage, and hybrid retrieval with reranking are done.

Your job in this session is to implement the generation layer (RAG answer synthesis)
and the sentiment extraction pipeline. These are the two highest-value user-facing features.

─── IMPLEMENT: generation/generator.py ─────────────────────────────────────────────────────

class RAGGenerator:

    __init__(self, config: dict)
        - Initialise openai.OpenAI client
        - Store: model name, max_tokens, temperature from config

    build_prompt(self, query: str, chunks: list[RetrievedChunk]) -> tuple[str, str]
        - Build system + user messages for GPT-4o-mini
        - System message:
            "You are a financial research assistant with access to SEC filings and
             financial news. Answer questions accurately using only the provided context.
             Always cite your sources using [Source N] notation. If the context does not
             contain enough information to answer, say so explicitly — do not speculate."
        - User message:
            - List context chunks with index, source, date, ticker, text
            - Then: "Question: {query}\n\nAnswer:"
        - Return (system_message, user_message)

    generate(self, query: str, chunks: list[RetrievedChunk]) -> GenerationResult
        - Call build_prompt
        - Call GPT-4o-mini with messages
        - Parse [Source N] citations from response to link back to chunk objects
        - Return GenerationResult

    _parse_citations(self, answer: str,
                     chunks: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk]]
        - Find all [Source N] patterns in answer
        - Map each to the corresponding chunk (index in context = N-1)
        - Return (cleaned_answer, cited_chunks)

@dataclass
class GenerationResult:
    answer: str
    cited_chunks: list[RetrievedChunk]
    prompt_tokens: int
    completion_tokens: int
    model: str

─── IMPLEMENT: generation/sentiment.py ─────────────────────────────────────────────────────

class SentimentExtractor:

    __init__(self, config: dict)
        - Initialise openai.OpenAI client
        - Store: model, sentiment cache TTL (hours)
        - self._cache: dict[str, tuple[SentimentResult, datetime]] = {}

    SENTIMENT_PROMPT = """
    Analyse the sentiment of the following financial text excerpt.
    Respond ONLY with a valid JSON object in exactly this format, no other text:
    {
        "score": <float between -1.0 (very negative) and 1.0 (very positive)>,
        "confidence": <float between 0.0 and 1.0>,
        "key_positive": [<up to 3 short positive phrases from the text>],
        "key_negative": [<up to 3 short negative phrases from the text>],
        "summary": <one sentence summarising the sentiment and main topics>
    }
    """

    extract_sentiment(self, chunk: Chunk) -> SentimentResult
        - Check cache: return cached result if within TTL
        - Build prompt with SENTIMENT_PROMPT + chunk text
        - Call GPT-4o-mini with temperature=0
        - Parse JSON response (use json.loads, handle JSONDecodeError)
        - Validate score in [-1, 1] and confidence in [0, 1]
        - Cache result
        - Return SentimentResult

    extract_ticker_sentiment(self, ticker: str,
                              chunks: list[Chunk]) -> list[SentimentSignal]
        - Sort chunks by date descending, take most recent 10
        - Call extract_sentiment on each
        - Convert to SentimentSignal objects with chunk metadata
        - Return sorted by date descending

    _parse_sentiment_json(self, response_text: str) -> dict
        - json.loads(response_text.strip())
        - If JSONDecodeError: attempt to extract JSON substring with regex
        - If still fails: return default {"score": 0.0, "confidence": 0.0,
                                          "key_positive": [], "key_negative": [],
                                          "summary": "Parse error"}

@dataclass
class SentimentResult:
    score: float
    confidence: float
    key_positive: list[str]
    key_negative: list[str]
    summary: str

─── IMPLEMENT: tests/test_generation.py ────────────────────────────────────────────────────

Mock ALL OpenAI calls using unittest.mock.patch("openai.OpenAI").

test_rag_generator_builds_prompt_with_context
    - Create 3 RetrievedChunk objects
    - Call build_prompt("What is Apple's revenue?", chunks)
    - Assert user message contains all 3 chunk texts
    - Assert "[Source 1]", "[Source 2]", "[Source 3]" mentioned in instructions

test_rag_generator_parses_citations
    - Mock LLM response: "Revenue grew [Source 1] due to iPhone sales [Source 2]."
    - Assert _parse_citations returns 2 cited_chunks

test_rag_generator_handles_no_citations
    - Mock LLM response with no [Source N] patterns
    - Assert generate returns empty cited_chunks, answer unchanged

test_sentiment_extractor_returns_valid_score
    - Mock LLM response with valid JSON sentiment
    - Assert result.score is float in [-1.0, 1.0]
    - Assert result.confidence is float in [0.0, 1.0]

test_sentiment_extractor_caches_results
    - Call extract_sentiment on same chunk twice
    - Assert OpenAI called exactly once (second call from cache)

test_sentiment_extractor_handles_json_error
    - Mock LLM response as "Sorry, I can't do that"
    - Assert extract_sentiment returns default result (score=0.0) without raising

test_extract_ticker_sentiment_returns_most_recent
    - Create 15 chunks with dates spanning 3 years
    - Call extract_ticker_sentiment (mock OpenAI)
    - Assert returns at most 10 signals, sorted by date descending

─── IMPLEMENT: tests/test_sentiment.py ─────────────────────────────────────────────────────

test_sentiment_signal_score_range
    - Generate signals for AAPL chunks (mocked LLM)
    - Assert all scores in [-1, 1]

test_sentiment_positive_keywords_extracted
    - Mock LLM response with specific positive keywords
    - Assert key_positive contains those keywords

test_sentiment_cache_ttl_respected
    - Extract sentiment, advance mock clock by cache_hours + 1
    - Call again — assert OpenAI called a second time (cache expired)

─── DEFINITION OF DONE ─────────────────────────────────────────────────────────────────────

Session 7 is complete when:
1. `pytest tests/test_generation.py -v` shows 7 passing tests
2. `pytest tests/test_sentiment.py -v` shows 3 passing tests
3. `pytest tests/ -v` shows all previous tests passing
4. RAGGenerator.generate() handles missing citations gracefully
5. SentimentExtractor never raises on malformed LLM output
6. All OpenAI calls are mockable via standard unittest.mock
```

---

## SESSION 8 — API Endpoints and Integration

```
I am building AlphaSignal, a financial RAG system. Sessions 1–7 are complete:
all core components (ingestion, chunking, storage, retrieval, generation, sentiment) are built.

Your job in this session is to wire everything into the FastAPI routes,
add proper dependency injection, and write full integration tests for all endpoints.

─── DEPENDENCY INJECTION ARCHITECTURE ──────────────────────────────────────────────────────

All heavy components should be initialised ONCE at startup and shared via FastAPI dependency injection.
Do NOT re-instantiate them per request.

In api/app.py, add an AppState dataclass:

@dataclass
class AppState:
    config: dict
    pipeline: IngestionPipeline
    retriever: HybridRetriever
    reranker: CrossEncoderReranker
    generator: RAGGenerator
    sentiment_extractor: SentimentExtractor
    evaluator: RetrievalEvaluator
    start_time: float

On startup event:
    1. Load config.yaml
    2. Load .env
    3. Instantiate all components in dependency order
    4. Build BM25 index from existing data (retriever.build_bm25_index())
    5. Store as app.state.app_state
    6. Log startup summary: chunks_indexed, bm25_terms, model names

Create dependency functions in api/dependencies.py:

def get_app_state(request: Request) -> AppState:
    return request.app.state.app_state

def get_retriever(state: AppState = Depends(get_app_state)) -> HybridRetriever:
    return state.retriever

# etc. for each component

─── FULLY IMPLEMENT: api/routes/query.py ────────────────────────────────────────────────────

POST /query
    Request: QueryRequest
    Response: QueryResponse

    Implementation:
    1. Validate request (Pydantic handles this)
    2. Start timer
    3. retriever.retrieve(query, ticker_filter, date_from, date_to, top_k=20)
    4. reranker.rerank(query, candidates, top_k=request.top_k)
    5. generator.generate(query, top_k_chunks)
    6. Build QueryResponse:
        - answer: generation_result.answer
        - citations: convert cited_chunks to Citation objects
            (use chunk metadata for ticker, source, date, excerpt=text[:200])
        - latency_ms: actual elapsed ms
        - retrieval_scores: [c.final_score for c in top_k_chunks]
        - model_used: config generation model name
    7. Return QueryResponse

    Error handling:
    - No chunks found after retrieval: return answer="No relevant information found
      in the knowledge base for this query." with empty citations
    - OpenAI error: return 503 with ErrorResponse

─── FULLY IMPLEMENT: api/routes/sentiment.py ────────────────────────────────────────────────

GET /sentiment/{ticker}
    Query params: date_from (optional), date_to (optional)
    Response: SentimentResponse

    Implementation:
    1. Validate ticker (uppercase, 1-5 chars, must be in config tickers list)
    2. Start timer
    3. metadata_store.get_chunks_by_ticker(ticker) filtered by date range
    4. If no chunks: return SentimentResponse with empty signals, latest_score=None
    5. sentiment_extractor.extract_ticker_sentiment(ticker, chunks)
    6. Return SentimentResponse:
        - ticker: uppercase
        - signals: list of SentimentSignal
        - latest_score: signals[0].score if signals else None
        - latency_ms: actual elapsed ms

GET /sentiment/{ticker}/summary
    Response: JSON with aggregate stats:
    {
        "ticker": str,
        "period_days": int,  (days covered by available data)
        "avg_score": float,
        "trend": "improving" | "stable" | "declining",  (based on score direction)
        "signal_count": int,
        "most_recent_date": date,
        "latency_ms": int
    }

─── FULLY IMPLEMENT: api/routes/ingest.py ───────────────────────────────────────────────────

POST /ingest/{ticker}
    Request: IngestRequest
    Response: IngestResponse

    Implementation:
    1. Start timer
    2. pipeline.full_ingest(ticker)
    3. Rebuild BM25 index after ingestion (retriever.build_bm25_index())
    4. Return IngestResponse with actual counts and latency

POST /ingest/batch
    Request: {"tickers": list[str]}
    Response: {"results": list[IngestResponse], "total_latency_ms": int}
    - Ingest each ticker sequentially (not parallel — avoid rate limits)
    - Return all results even if some fail (status: "failed" for failed ones)

─── IMPLEMENT: tests/test_api.py ────────────────────────────────────────────────────────────

Mock all external calls: OpenAI, EDGAR downloader, feedparser.
Use TestClient from httpx.

test_query_endpoint_returns_200
    - Mock retriever to return 3 chunks, generator to return valid GenerationResult
    - POST /query {"query": "What is Apple revenue?", "ticker_filter": ["AAPL"]}
    - Assert 200, response matches QueryResponse schema

test_query_endpoint_returns_citations
    - Mock generator to return result with 2 cited chunks
    - Assert response.citations has 2 items with correct fields

test_query_endpoint_handles_empty_retrieval
    - Mock retriever to return []
    - Assert 200, answer contains "No relevant information"

test_sentiment_endpoint_returns_200
    - Mock metadata_store.get_chunks_by_ticker to return 5 chunks
    - Mock sentiment_extractor to return valid signals
    - GET /sentiment/AAPL
    - Assert 200, response matches SentimentResponse schema

test_sentiment_endpoint_invalid_ticker
    - GET /sentiment/INVALID_TICKER_TOO_LONG
    - Assert 422 (validation error)

test_sentiment_endpoint_unknown_ticker
    - GET /sentiment/ZZZZ  (not in config tickers)
    - Assert 404 with ErrorResponse

test_ingest_endpoint_triggers_pipeline
    - Mock pipeline.full_ingest to return IngestResult
    - POST /ingest/AAPL with IngestRequest
    - Assert pipeline.full_ingest called with "AAPL"
    - Assert 200 with IngestResponse

test_ingest_batch_processes_all_tickers
    - POST /ingest/batch {"tickers": ["AAPL", "MSFT", "GOOGL"]}
    - Assert response.results has 3 items

test_all_responses_include_latency_ms
    - Call each endpoint
    - Assert every response body has latency_ms > 0

─── IMPLEMENT: monitoring/metrics.py ────────────────────────────────────────────────────────

class MetricsCollector:

    __init__(self)
        - self._query_latencies: list[float] = []
        - self._ingest_latencies: list[float] = []
        - self._error_count: int = 0

    record_query(self, latency_ms: float)
    record_ingest(self, latency_ms: float)
    record_error(self)

    get_query_percentiles(self) -> dict
        - Return {"p50": float, "p95": float, "p99": float}
        - Use numpy percentile

    get_summary(self) -> dict
        - Return all metrics as a dict

Add GET /metrics endpoint in a new api/routes/metrics.py:
    - Return MetricsCollector.get_summary()
    - Include: query percentiles, ingest percentiles, error_count, chunks_indexed

─── DEFINITION OF DONE ─────────────────────────────────────────────────────────────────────

Session 8 is complete when:
1. `pytest tests/test_api.py -v` shows 9 passing tests
2. `pytest tests/ -v` shows all previous tests passing (target: >40 total)
3. POST /query returns QueryResponse with citations
4. GET /sentiment/{ticker} returns SentimentResponse with signals
5. All endpoints return latency_ms
6. GET /metrics returns query latency percentiles
7. No component instantiated more than once (check via startup logs)
```

---

## SESSION 9 — Evaluation Framework and Golden Set

```
I am building AlphaSignal, a financial RAG system. Sessions 1–8 are complete:
full pipeline from ingestion to API is built and tested.

Your job in this session is the most important one for the portfolio:
build the evaluation framework, create the golden set, run the full benchmark,
and populate EVALUATION.md with real numbers.

This session requires real API calls (OpenAI for embeddings + generation) and
real data (SEC filings). Budget ~$5 in API costs. Use a small subset first to verify.

─── PHASE 1: BUILD THE GOLDEN SET ──────────────────────────────────────────────────────────

Create evaluation/golden_set.json with exactly 50 Q&A pairs.
Cover these 10 tickers, 5 questions each:
    AAPL, MSFT, NVDA, JPM, GOOGL, AMZN, META, TSLA, GS, MS

Question types to include (mix across tickers):
    - Factual: "What was [ticker]'s revenue in [quarter]?"
    - Trend: "How has [ticker]'s operating margin changed over the last two quarters?"
    - Comparative: "What risk factors does [ticker] highlight related to AI competition?"
    - Sentiment: "Was [ticker]'s guidance positive or negative in the most recent filing?"
    - Specific: "What did [ticker] management say about [specific topic]?"

Format for each entry:
{
    "id": "001",
    "question": "...",
    "ticker": "AAPL",
    "relevant_chunk_ids": [],        <- fill in AFTER ingesting corpus
    "answer_summary": "...",         <- brief human-written expected answer
    "question_type": "factual"       <- factual | trend | comparative | sentiment | specific
}

IMPORTANT: relevant_chunk_ids will be populated after building the corpus in Phase 2.
Leave them as [] for now. We will fill them in during evaluation.

─── PHASE 2: BUILD THE CORPUS ───────────────────────────────────────────────────────────────

Implement scripts/build_corpus.py:

def main():
    1. Load config.yaml
    2. Instantiate full pipeline
    3. For each ticker in config.tickers:
        - Call pipeline.full_ingest(ticker)
        - Log progress: "Ingested AAPL: 45 chunks from 8 filings, 23 chunks from 50 articles"
    4. Print summary table:
        Ticker | Filings | Articles | Chunks | Date Range
    5. Save corpus stats to data/corpus_stats.json

Run this script and paste the output summary table into EVALUATION.md under "Corpus Statistics".

─── PHASE 3: ANNOTATE GOLDEN SET ───────────────────────────────────────────────────────────

Implement a helper script scripts/annotate_golden_set.py:

def annotate():
    1. Load golden_set.json
    2. For each question with empty relevant_chunk_ids:
        - Run retrieval with top_k=20
        - Print question + top 5 retrieved chunks with scores
        - Prompt: "Enter relevant chunk IDs (comma-separated), or press Enter to skip: "
        - Update relevant_chunk_ids
    3. Save updated golden_set.json

Use this to manually label the 50 questions. This takes ~30 minutes but is essential.
Only proceed to Phase 4 after all 50 questions are annotated.

─── PHASE 4: RUN BENCHMARK ACROSS CONFIGS ───────────────────────────────────────────────────

Implement scripts/benchmark.py:

CONFIGS_TO_BENCHMARK = [
    {
        "name": "Baseline: naive chunks + dense only",
        "chunking": "naive_512",      # fixed 512-token chunks, no overlap
        "retrieval": "dense_only",    # disable BM25
        "reranking": False,
    },
    {
        "name": "Semantic chunks + dense only",
        "chunking": "semantic",
        "retrieval": "dense_only",
        "reranking": False,
    },
    {
        "name": "Semantic chunks + hybrid",
        "chunking": "semantic",
        "retrieval": "hybrid",
        "reranking": False,
    },
    {
        "name": "Semantic chunks + hybrid + reranker",
        "chunking": "semantic",
        "retrieval": "hybrid",
        "reranking": True,
    },
]

For each config:
    1. If chunking == "naive_512": re-chunk corpus with naive splitter
       Else: use existing semantic chunks
    2. Re-embed and re-index if chunks changed
    3. Run RetrievalEvaluator.evaluate(golden_set, top_k=10)
    4. Record: MRR@10, NDCG@5, Hit@3, avg_latency_ms
    5. Print results table

Output format:
    Config                                  | MRR@10 | NDCG@5 | Hit@3 | Avg Latency
    ----------------------------------------|--------|--------|-------|------------
    Baseline: naive chunks + dense only     | 0.XX   | 0.XX   | 0.XX  | XXXms
    Semantic chunks + dense only            | 0.XX   | 0.XX   | 0.XX  | XXXms
    Semantic chunks + hybrid                | 0.XX   | 0.XX   | 0.XX  | XXXms
    Semantic chunks + hybrid + reranker     | 0.XX   | 0.XX   | 0.XX  | XXXms

─── WRITE: EVALUATION.MD ────────────────────────────────────────────────────────────────────

Create EVALUATION.md in the project root with this structure:

# AlphaSignal Retrieval Evaluation

## Overview
Brief paragraph: what we're evaluating, why it matters, what the corpus contains.

## Corpus Statistics
[paste build_corpus.py output table here]

## Evaluation Methodology
- Golden set: 50 manually annotated Q&A pairs
- 10 tickers (AAPL, MSFT, NVDA, JPM, GOOGL, AMZN, META, TSLA, GS, MS)
- 5 questions per ticker
- Question types: factual (N), trend (N), comparative (N), sentiment (N), specific (N)
- Metrics: MRR@10, NDCG@5, Hit@3
- Each config evaluated on all 50 questions

## Results
[paste benchmark.py output table here]

## Analysis
3-4 sentences: what drove the biggest improvement? Where does the system still struggle?
What would you try next?

## Query Latency (full pipeline: embed + retrieve + rerank + generate)
| Percentile | Latency |
|------------|---------|
| p50        | XXXms   |
| p95        | XXXms   |
| p99        | XXXms   |

## Failure Cases
5 examples of questions the system answers poorly, with diagnosis.

─── IMPLEMENT: tests/test_evaluator.py ─────────────────────────────────────────────────────

test_evaluator_loads_golden_set
    - evaluation/golden_set.json must have >= 50 entries
    - Assert all required fields present: id, question, ticker, relevant_chunk_ids

test_evaluator_mrr_perfect_score
    - ranked_ids = ["a", "b", "c"], relevant_ids = ["a"]
    - Assert compute_mrr == 1.0

test_evaluator_mrr_second_place
    - ranked_ids = ["a", "b", "c"], relevant_ids = ["b"]
    - Assert compute_mrr == 0.5

test_evaluator_ndcg_at_5_with_two_relevant
    - Construct ranked list where relevant docs at positions 1 and 3
    - Assert NDCG@5 > NDCG@5 when relevant docs at positions 2 and 5

test_evaluator_hit_at_3_true
    - relevant doc at position 2
    - Assert hit@3 == 1.0

test_evaluator_hit_at_3_false
    - relevant doc at position 5
    - Assert hit@3 == 0.0

test_golden_set_has_variety_of_question_types
    - Load golden_set.json
    - Assert at least 3 distinct question_type values present

─── DEFINITION OF DONE ─────────────────────────────────────────────────────────────────────

Session 9 is complete when:
1. evaluation/golden_set.json has exactly 50 annotated entries
2. `pytest tests/test_evaluator.py -v` shows 7 passing tests
3. scripts/benchmark.py runs and prints a 4-row results table
4. EVALUATION.md exists with real benchmark numbers (not placeholders)
5. EVALUATION.md includes corpus statistics and failure case analysis
6. The full pipeline config (hybrid + reranker) achieves MRR@10 > 0.5
   (if not: investigate and document why in EVALUATION.md)
```

---

## SESSION 10 — Polish, README, and Final QA

```
I am building AlphaSignal, a financial RAG system. Sessions 1–9 are complete:
full pipeline, API, evaluation framework, and EVALUATION.md with real benchmark numbers.

Your job in this session is to bring the project to portfolio quality:
comprehensive README, test coverage enforcement, docstring pass, error handling review,
and final QA checks.

─── WRITE: README.md ────────────────────────────────────────────────────────────────────────

Write a comprehensive README.md. It must include every section below.

## AlphaSignal

One-paragraph description: what it is, what problem it solves, how it fits with AlphaLab.

## Architecture

ASCII diagram showing the full pipeline:

    SEC EDGAR filings ──┐
                        ├──► Semantic Chunker ──► Embedder (ada-002) ──► FAISS + SQLite
    Financial news ─────┘                                                      │
                                                                               ▼
    Query ──────────────────────────────────────────────────► Hybrid Retriever (BM25 + Dense)
                                                                               │
                                                                               ▼
                                                                    Cross-Encoder Reranker
                                                                               │
                                                                               ▼
                                                              GPT-4o-mini + Citation Injection
                                                                               │
                                                                               ▼
                                                                    REST API (FastAPI)
                                                                               │
                                                               ┌───────────────┴───────────────┐
                                                               ▼                               ▼
                                                        AlphaLab sentiment              Dashboard (v2)
                                                        feature feed

## Quickstart

Step by step:
    git clone ...
    cd alphasignal
    cp .env.example .env  # add your OPENAI_API_KEY
    pip install -r requirements.txt
    python scripts/build_corpus.py  # ingest 10 tickers (~10 min, ~$2 API cost)
    uvicorn api.app:app --reload
    # API now at http://localhost:8000
    # Docs at http://localhost:8000/docs

## API Reference

Document all endpoints with curl examples:

### POST /query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What did Apple management say about AI features in the latest 10-K?",
    "ticker_filter": ["AAPL"],
    "top_k": 5
  }'
```
Response: (show example QueryResponse JSON)

### GET /sentiment/{ticker}
### POST /ingest/{ticker}
### GET /health
### GET /metrics

## Evaluation

Link to EVALUATION.md. Include the benchmark results table inline (copy from EVALUATION.md).
2-3 sentence summary of key findings.

## AlphaLab Integration

Section explaining how AlphaSignal's sentiment endpoint feeds into AlphaLab:
- GET /sentiment/{ticker} returns time-series sentiment scores
- AlphaLab's SentimentMomentum strategy consumes these as features
- Walk through the data contract (JSON format)

## Project Structure

Full annotated directory tree.

## Tech Stack

Table: Component | Technology | Why

## Development

How to run tests:
    pytest tests/ -v --cov=. --cov-report=term-missing

How to add a new ingestion source.
How to add a new retrieval strategy.

─── ENFORCE: test coverage ──────────────────────────────────────────────────────────────────

Run: pytest tests/ --cov=. --cov-report=term-missing

Identify any module with < 70% coverage.
Write additional tests until overall project coverage >= 70%.

Priority modules if coverage is low:
- generation/generator.py (citation parsing edge cases)
- retrieval/retriever.py (filter combinations)
- ingestion/chunker.py (sentence boundary edge cases)

─── DOCSTRING PASS ──────────────────────────────────────────────────────────────────────────

Every public class and method must have a docstring. Format: Google style.

Example:
    def retrieve(self, query: str, ticker_filter: list[str] | None = None) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a query using hybrid BM25 + dense search.

        Args:
            query: Natural language question or search string.
            ticker_filter: If provided, restrict results to these ticker symbols.

        Returns:
            List of RetrievedChunk objects sorted by hybrid score descending.
            Empty list if no chunks match or corpus is empty.
        """

─── ERROR HANDLING REVIEW ───────────────────────────────────────────────────────────────────

Check every external call and ensure it has proper error handling:

1. OpenAI API calls: handle RateLimitError, APIConnectionError, APIStatusError
   → Raise custom AlphaSignalError with code "OPENAI_ERROR"

2. FAISS operations: handle index-not-loaded state
   → Raise custom AlphaSignalError with code "INDEX_NOT_LOADED"

3. SQLite operations: handle locked database, corrupt database
   → Log and raise custom AlphaSignalError with code "DB_ERROR"

4. SEC EDGAR: handle rate limiting, missing ticker
   → Log warning, return empty list (never raise)

5. RSS feeds: handle malformed XML, timeout
   → Log warning, return partial results (never raise)

Define in a new file api/exceptions.py:

class AlphaSignalError(Exception):
    def __init__(self, message: str, code: str, detail: str | None = None):
        ...

Register exception handler in app.py:
    @app.exception_handler(AlphaSignalError)
    async def alphasignal_error_handler(request, exc):
        return JSONResponse(status_code=500,
                           content=ErrorResponse(error=exc.message,
                                                code=exc.code,
                                                detail=exc.detail).model_dump())

─── FINAL QA CHECKLIST ──────────────────────────────────────────────────────────────────────

Verify every item before declaring done:

Code quality:
[ ] `pytest tests/ -v` — all tests pass
[ ] `pytest tests/ --cov=. --cov-report=term-missing` — coverage >= 70%
[ ] No hardcoded API keys anywhere (grep for "sk-")
[ ] All config values read from config.yaml or .env
[ ] .env.example has all required variables documented
[ ] .gitignore includes: .env, data/, *.faiss, __pycache__, .venv

API:
[ ] GET /health returns 200 with correct schema
[ ] POST /query returns QueryResponse with citations and latency_ms
[ ] GET /sentiment/{ticker} returns SentimentResponse
[ ] GET /metrics returns latency percentiles
[ ] All endpoints return ErrorResponse on error (not HTML error pages)
[ ] FastAPI auto-docs work: http://localhost:8000/docs shows all endpoints

Documentation:
[ ] README.md has architecture diagram, quickstart, API reference, eval summary
[ ] EVALUATION.md has real benchmark numbers (not placeholders like 0.XX)
[ ] All public methods have docstrings
[ ] CONTRIBUTING section in README

Repository hygiene:
[ ] data/ directory gitignored (FAISS index + SQLite db should NOT be committed)
[ ] requirements.txt has all dependencies pinned
[ ] No print() statements in production code (use logging)
[ ] Logging uses structured format: "%(asctime)s %(name)s %(levelname)s %(message)s"

─── DEFINITION OF DONE ─────────────────────────────────────────────────────────────────────

Session 10 is complete when:
1. `pytest tests/ --cov=. --cov-report=term-missing` shows >= 70% coverage
2. `uvicorn api.app:app --reload` starts cleanly with no warnings
3. README.md passes a review: architecture diagram present, quickstart works,
   API reference has curl examples, eval table present
4. EVALUATION.md has real numbers, corpus stats, and failure case analysis
5. All 10 items in the QA checklist are ticked
6. The repository looks like professional open-source software, not a student project
```
