# `/sentiment/{ticker}` response contract

This documents the current, tested behavior of
`SentimentResponse` (see `alphasignal/api/schemas.py` and
`alphasignal/api/routes/sentiment.py`). All fields below marked "additive"
were added without changing the meaning or presence of any pre-existing
field - a consumer reading only `latest_score`, `signals[].score`,
`signals[].confidence`, `sources` (via `signals[].source`), `latency_ms`,
and `data_available` continues to work unmodified.

## The six states this contract distinguishes

| Situation | `data_available` | `status` | `degraded` | `degradation_reason` | `latest_score` |
|---|---|---|---|---|---|
| Genuine extraction, incl. genuine neutral | `true` | `"ok"` | `false` | `null` | real score (may be `0.0`) |
| No company-attributed evidence up to `as_of` (`data_available=false` when nothing is ingested at all; `true` when only body mentions or unattributed rows exist) | `false` / `true` | `"no_data"` | `false` | `null` | `null` |
| Data exists, but no company-attributed document inside the freshness window ending at `as_of` | `true` | `"stale"` | `false` | `null` | `null` (an old score is never presented as current) |
| Some chunks failed (provider/parse error, or quotes not found in the text) | `true` | `"degraded"` | `true` | `"partial_extraction_failure"` | aggregate over the **reliable** evidence only |
| Every chunk failed | `true` | `"degraded"` | `true` | `"full_extraction_failure"` | `null` (never fabricated as `0.0`) |
| Unhandled service/request failure | n/a - HTTP 5xx, no `SentimentResponse` body | | | | |

## New fields (additive, all with backward-compatible defaults)

- `status: Literal["ok", "no_data", "degraded", "stale"] = "ok"` - a `Literal` type, not a bare `str`, so the finite set of valid values is a structural, Pydantic-enforced guarantee (and shows up as an enum in the generated OpenAPI schema).
- `degraded: bool = False`.
- `degradation_reason: Literal["partial_extraction_failure", "full_extraction_failure"] | None = None` - also a `Literal` type (nullable), never a bare `str`.
- `reliable_chunk_count: int = 0`, `total_chunk_count: int = 0` - how many of the chunks considered for this response produced a genuine model prediction vs. a fallback.
- (2026-10) `method` (`document_recency_weighted_v2`), `as_of`, `freshness_days`, `documents_used`, `last_document_date`, `coverage` (per doc type: years present/missing), and `SentimentSignal.document_id`/`url`. `latest_score` is the document-level aggregate: chunk scores confidence-weighted within a document, documents weighted by confidence × recency (7-day half-life), company-attributed evidence only.
- `SentimentSignal.reliable: bool = True` - kept for compatibility. Since 2026-10 `signals` holds one entry per document that had at least one reliable chunk, so every returned signal is `reliable=true`; unreliable chunks are counted in `reliable_chunk_count`/`total_chunk_count` and excluded from the score, not returned.

## What counts as "reliable"

A chunk-level `SentimentResult` (`alphasignal/generation/sentiment.py`) is
marked `reliable=False` when:
- the provider's JSON response could not be parsed at all (regex/JSON fallback used), or
- the parsed score/confidence was non-finite (NaN/Infinity - a provider malfunction, not a real prediction), or
- the API call raised an exception, or
- a quoted `key_positive`/`key_negative` phrase does not occur in the chunk text (the model cited evidence that isn't there).

It is `reliable=True` whenever a real (possibly genuinely neutral, `score=0.0`) prediction was parsed from the provider. Genuine neutral and "provider fallback" both used to produce an identical `SentimentResult(score=0.0, confidence=0.0)` before this change - the `reliable` flag is what makes them distinguishable.

## What did NOT change

- Unhandled service/request failures still raise and are converted to a generic HTTP 500 by `alphasignal/api/app.py`'s global exception handler (or a route-specific 4xx, e.g. unknown ticker -> 404) - never disguised as a 200 with a neutral score.
- `latest_score` is still `None` for "no data," exactly as before - this contract only adds the ability to tell "no data" apart from "degraded extraction over real data," which previously both could produce `latest_score=None` (full degradation) or a real-looking `0.0` (previously indistinguishable from genuine neutral).
- AlphaLive's client reads `status`, `degraded`, `degradation_reason` and the chunk counts: `no_data`, `stale`, full degradation and contract violations are not evaluated and fall to AlphaLive's `ALPHASIGNAL_FAIL_POLICY` (entries only); a usable partial degradation is evaluated and tagged distinctly.
