# `/sentiment/{ticker}` response contract

This documents the current, tested behavior of
`SentimentResponse` (see `alphasignal/api/schemas.py` and
`alphasignal/api/routes/sentiment.py`). All fields below marked "additive"
were added without changing the meaning or presence of any pre-existing
field - a consumer reading only `latest_score`, `signals[].score`,
`signals[].confidence`, `sources` (via `signals[].source`), `latency_ms`,
and `data_available` continues to work unmodified.

## The five states this contract distinguishes

| Situation | `data_available` | `status` | `degraded` | `degradation_reason` | `latest_score` |
|---|---|---|---|---|---|
| Genuine extraction, incl. genuine neutral | `true` | `"ok"` | `false` | `null` | real score (may be `0.0`) |
| No chunks ingested for this ticker | `false` | `"no_data"` | `false` | `null` | `null` |
| Some chunks' extraction fell back to a provider/parsing default | `true` | `"degraded"` | `true` | `"partial_extraction_failure"` | most recent **reliable** score |
| Every chunk's extraction fell back | `true` | `"degraded"` | `true` | `"full_extraction_failure"` | `null` (never fabricated as `0.0`) |
| Unhandled service/request failure | n/a - HTTP 5xx, no `SentimentResponse` body | | | | |

## New fields (additive, all with backward-compatible defaults)

- `status: Literal["ok", "no_data", "degraded"] = "ok"` - a `Literal` type, not a bare `str`, so the finite set of valid values is a structural, Pydantic-enforced guarantee (and shows up as an enum in the generated OpenAPI schema).
- `degraded: bool = False`.
- `degradation_reason: Literal["partial_extraction_failure", "full_extraction_failure"] | None = None` - also a `Literal` type (nullable), never a bare `str`.
- `reliable_chunk_count: int = 0`, `total_chunk_count: int = 0` - how many of the chunks considered for this response produced a genuine model prediction vs. a fallback.
- `SentimentSignal.reliable: bool = True` - per-chunk visibility into which individual signals are genuine vs. fallback. Partial degradation does **not** discard the unreliable signal from `signals` - it's still returned, just flagged.

## What counts as "reliable"

A chunk-level `SentimentResult` (`alphasignal/generation/sentiment.py`) is
marked `reliable=False` when:
- the provider's JSON response could not be parsed at all (regex/JSON fallback used), or
- the parsed score/confidence was non-finite (NaN/Infinity - a provider malfunction, not a real prediction), or
- the API call raised an exception.

It is `reliable=True` whenever a real (possibly genuinely neutral, `score=0.0`) prediction was parsed from the provider. Genuine neutral and "provider fallback" both used to produce an identical `SentimentResult(score=0.0, confidence=0.0)` before this change - the `reliable` flag is what makes them distinguishable.

## What did NOT change

- Unhandled service/request failures still raise and are converted to a generic HTTP 500 by `alphasignal/api/app.py`'s global exception handler (or a route-specific 4xx, e.g. unknown ticker -> 404) - never disguised as a 200 with a neutral score.
- `latest_score` is still `None` for "no data," exactly as before - this contract only adds the ability to tell "no data" apart from "degraded extraction over real data," which previously both could produce `latest_score=None` (full degradation) or a real-looking `0.0` (previously indistinguishable from genuine neutral).
- AlphaLive's client does not yet read `status`/`degraded`/`degradation_reason`/`reliable_chunk_count`/`total_chunk_count`/`signals[].reliable`. Consuming this metadata to change AlphaLive's pre-execution gate behavior is a separate, future cross-repo change - out of scope for this pass.
