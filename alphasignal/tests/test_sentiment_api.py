"""GET /sentiment/{ticker} - document-level, freshness-bounded aggregation.

Real route + real SQLite metadata store with real chunks; only the
per-chunk model call (SentimentExtractor._score) is replaced. Expected
aggregates are computed by hand.
"""

import itertools
import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from alphasignal.generation import SentimentResult
from alphasignal.ingestion import Chunk
from alphasignal.tests import test_api as _api

client = _api.client  # reuse the API test client fixture

TODAY = datetime.now(timezone.utc).date()


def _chunk(
    doc,
    i,
    days_ago,
    text=None,
    doc_type="10-Q",
    attribution="filer",
    section="item_2",
    ticker="AAPL",
):
    return Chunk(
        chunk_id=f"{doc}_{i:04d}",
        ticker=ticker,
        text=text or f"{doc} chunk {i}: revenue grew and margins improved",
        token_count=10,
        doc_type=doc_type,
        source="SEC EDGAR" if doc_type != "news" else "Yahoo Finance",
        section=section if doc_type != "news" else None,
        date=TODAY - timedelta(days=days_ago),
        url=None,
        chunk_index=i,
        total_chunks=3,
        source_id=doc,
        attribution=attribution,
    )


@pytest.fixture
def env(client, tmp_path):  # noqa: F811
    st = client.app.state.app_state
    ext = st.sentiment_extractor
    ext.cache_path = tmp_path / "sentiment_cache.json"
    ext._cache = {}
    return client, st.pipeline.metadata_store, ext


def _scores(mapping):
    """_score replacement: chunk_id -> (score, confidence, reliable)."""

    def fake(chunk):
        s, c, ok = mapping[chunk.chunk_id]
        return (
            SentimentResult(s, c, [], [], f"summary {chunk.chunk_id}", reliable=ok),
            True,
        )

    return fake


def test_no_data(env):
    client, store, ext = env
    r = client.get("/sentiment/AAPL")
    d = r.json()
    assert (d["status"], d["data_available"], d["latest_score"]) == (
        "no_data",
        False,
        None,
    )
    assert '"latest_score":null' in r.text.replace(" ", "")


def test_stale_when_nothing_within_freshness_window(env):
    client, store, ext = env
    store.add_chunks([_chunk("aapl_10q_old", 0, days_ago=60)])
    with patch.object(ext, "_score") as score:
        d = client.get("/sentiment/AAPL").json()
    score.assert_not_called()  # stale evidence is never scored or used
    assert d["status"] == "stale" and d["latest_score"] is None
    assert d["data_available"] is True
    assert d["last_document_date"] == str(TODAY - timedelta(days=60))
    assert d["freshness_days"] == 30


def test_hand_computed_document_level_recency_weighted_aggregate(env):
    client, store, ext = env
    store.add_chunks(
        [
            _chunk("aapl_10q_a", 0, 0),
            _chunk("aapl_10q_a", 1, 0),
            _chunk("aapl_10q_b", 0, 7),
        ]
    )
    m = {
        "aapl_10q_a_0000": (0.6, 0.8, True),
        "aapl_10q_a_0001": (0.2, 0.4, True),
        "aapl_10q_b_0000": (-0.5, 1.0, True),
    }
    with patch.object(ext, "_score", side_effect=_scores(m)):
        d = client.get("/sentiment/AAPL").json()
    # doc A: (0.6*0.8 + 0.2*0.4)/1.2 = 0.46667, conf 0.6, age 0  -> w 0.6
    # doc B: -0.5, conf 1.0, age 7 = one half-life               -> w 0.5
    expected = (0.6 * (0.56 / 1.2) + 0.5 * -0.5) / 1.1
    assert d["status"] == "ok"
    assert d["latest_score"] == pytest.approx(expected, abs=1e-4)
    assert d["documents_used"] == 2
    assert [s["document_id"] for s in d["signals"]] == ["aapl_10q_a", "aapl_10q_b"]


def test_only_company_attributed_evidence_and_no_future_documents(env):
    client, store, ext = env
    store.add_chunks(
        [
            _chunk("aapl_news_title", 0, 1, doc_type="news", attribution="news_title"),
            _chunk("aapl_news_body", 0, 1, doc_type="news", attribution="news_body"),
            _chunk("aapl_10q_future", 0, -3),  # dated after as_of
        ]
    )
    seen = []

    def fake(chunk):
        seen.append(chunk.chunk_id)
        return SentimentResult(0.3, 0.9, [], [], "s", reliable=True), True

    with patch.object(ext, "_score", side_effect=fake):
        d = client.get(f"/sentiment/AAPL?date_to={TODAY}").json()
    assert seen == ["aapl_news_title_0000"]
    assert d["documents_used"] == 1


@pytest.mark.parametrize(
    "flags",
    [c for n in range(1, 5) for c in itertools.product([True, False], repeat=n)],
)
def test_contract_invariants_for_every_reliability_combination(env, flags):
    client, store, ext = env
    store.add_chunks([_chunk(f"aapl_10q_d{i}", 0, i) for i in range(len(flags))])
    m = {
        f"aapl_10q_d{i}_0000": ((0.4, 0.7, True) if ok else (0.0, 0.0, False))
        for i, ok in enumerate(flags)
    }
    with patch.object(ext, "_score", side_effect=_scores(m)):
        d = client.get("/sentiment/AAPL").json()
    assert d["status"] in {"ok", "degraded"}
    assert d["total_chunk_count"] == len(flags)
    assert d["reliable_chunk_count"] == sum(flags)
    if sum(flags) == 0:
        assert (d["status"], d["degradation_reason"], d["latest_score"]) == (
            "degraded",
            "full_extraction_failure",
            None,
        )
    elif sum(flags) < len(flags):
        assert d["status"] == "degraded" and d["latest_score"] is not None
        assert d["degradation_reason"] == "partial_extraction_failure"
    else:
        assert d["status"] == "ok" and d["degradation_reason"] is None
    for s in d["signals"]:
        assert s["reliable"] is True and -1 <= s["score"] <= 1


def test_quotes_not_in_text_make_a_result_unreliable(env):
    client, store, ext = env
    c = _chunk("aapl_10q_q", 0, 0, text="Revenue increased 5% on strong iPhone demand.")

    class Msg:
        def __init__(self, content):
            self.choices = [
                type("C", (), {"message": type("M", (), {"content": content})()})()
            ]

    injected = json.dumps(
        {
            "score": 1.0,
            "confidence": 1.0,
            "key_positive": ["record breaking quarter"],
            "key_negative": [],
            "summary": "x",
        }
    )
    with patch.object(
        ext.client.chat.completions, "create", return_value=Msg(injected)
    ):
        r, cacheable = ext._score(c)
    assert r.reliable is False
    assert cacheable is True  # a real model answer, kept as unreliable
    honest = json.dumps(
        {
            "score": 0.5,
            "confidence": 0.8,
            "key_positive": ["strong iPhone demand"],
            "key_negative": [],
            "summary": "x",
        }
    )
    with patch.object(ext.client.chat.completions, "create", return_value=Msg(honest)):
        assert ext._score(c)[0].reliable is True


def test_scores_are_cached_on_disk_and_reused(env, tmp_path):
    client, store, ext = env
    store.add_chunks([_chunk("aapl_10q_c", 0, 0)])
    with patch.object(
        ext, "_score", side_effect=_scores({"aapl_10q_c_0000": (0.2, 0.5, True)})
    ) as s:
        a = client.get("/sentiment/AAPL").json()
        b = client.get("/sentiment/AAPL").json()
    assert s.call_count == 1  # warm call served from cache
    assert a["latest_score"] == b["latest_score"]
    assert ext.cache_path.exists()
    from alphasignal.generation.sentiment import SentimentExtractor

    with patch("alphasignal.generation.sentiment.OpenAI"):
        fresh = SentimentExtractor(
            {
                **client.app.state.app_state.config,
                "sentiment": {"cache_path": str(ext.cache_path)},
            }
        )
    assert len(fresh._cache) == 1


def test_coverage_reports_missing_years(env):
    client, store, ext = env
    old = _chunk("aapl_10k_2019", 0, 0, doc_type="10-K")
    old.date = date(2019, 11, 1)
    new = _chunk("aapl_10k_2024", 0, 0, doc_type="10-K")
    new.date = date(2024, 11, 1)
    store.add_chunks([old, new])
    d = client.get("/sentiment/AAPL").json()
    assert d["coverage"]["10-K"]["years_missing"] == [2020, 2021, 2022, 2023]


def test_summary_uses_the_same_aggregate(env):
    client, store, ext = env
    store.add_chunks([_chunk(f"aapl_10q_s{i}", 0, i) for i in range(3)])
    m = {f"aapl_10q_s{i}_0000": (0.1 * (i + 1), 0.5, True) for i in range(3)}
    with patch.object(ext, "_score", side_effect=_scores(m)):
        d = client.get("/sentiment/AAPL/summary").json()
    assert d["status"] == "ok" and d["signal_count"] == 3
    assert d["avg_score"] == pytest.approx(0.2, abs=1e-3)


def test_openapi_status_enum_includes_stale(env):
    client, _, _ = env
    schema = client.app.openapi()
    s = json.dumps(
        schema["components"]["schemas"]["SentimentResponse"]["properties"]["status"]
    )
    for v in ("ok", "no_data", "degraded", "stale"):
        assert v in s


def test_quote_failures_are_cached_but_provider_errors_are_not(env):
    """Regression (2026-10 review): a result whose quotes failed verification
    used to be re-scored on every warm call (cold latency, and a non-
    deterministic model could flip ok/degraded). It is now cached as
    unreliable; provider failures are still retried."""
    client, store, ext = env
    store.add_chunks([_chunk("aapl_10q_qf", 0, 0), _chunk("aapl_10q_pe", 0, 1)])
    calls = []

    def fake(chunk):
        calls.append(chunk.chunk_id)
        if chunk.chunk_id == "aapl_10q_qf_0000":
            return SentimentResult(0.9, 0.9, [], [], "bad quotes", reliable=False), True
        return (
            SentimentResult(0.0, 0.0, [], [], "Provider error", reliable=False),
            False,
        )

    with patch.object(ext, "_score", side_effect=fake):
        a = client.get("/sentiment/AAPL").json()
        b = client.get("/sentiment/AAPL").json()
    assert calls.count("aapl_10q_qf_0000") == 1
    assert calls.count("aapl_10q_pe_0000") == 2
    assert a["status"] == b["status"] == "degraded"
    assert a["latest_score"] is None and b["latest_score"] is None


def test_no_data_with_only_body_mentions_reports_data_available(env):
    """no_data means no company-attributed evidence; when body-only chunks
    exist the response must not claim nothing is ingested."""
    client, store, ext = env
    store.add_chunks(
        [_chunk("aapl_news_b", 0, 0, doc_type="news", attribution="news_body")]
    )
    d = client.get("/sentiment/AAPL").json()
    assert (d["status"], d["data_available"], d["latest_score"]) == (
        "no_data",
        True,
        None,
    )


def test_default_as_of_is_the_utc_date():
    from alphasignal.api.routes.sentiment import _utc_today

    assert _utc_today() == datetime.now(timezone.utc).date()


def test_closing_fence_variants_are_defused():
    from alphasignal.generation.generator import _CLOSE_DOCUMENT
    from alphasignal.generation.sentiment import _CLOSE_EXCERPT

    for tag in ("</document>", "</DOCUMENT>", "</document >", "< /Document>"):
        assert "</document>" not in _CLOSE_DOCUMENT.sub("</ document>", tag).lower()
    for tag in ("</excerpt>", "</EXCERPT >"):
        assert "</excerpt>" not in _CLOSE_EXCERPT.sub("</ excerpt>", tag).lower()


def _completion(payload):
    from types import SimpleNamespace

    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
    )


def test_same_excerpt_is_scored_and_cached_per_ticker(env):
    _, _, ext = env
    excerpt = "Microsoft sales grew. Apple sales fell."
    msft = _chunk("msft", 0, 0, text=excerpt, ticker="MSFT")
    aapl = _chunk("aapl", 0, 0, text=excerpt, ticker="AAPL")

    def respond(*, messages, **kwargs):
        instructions = messages[1]["content"].split("<excerpt>", 1)[0]
        if "MSFT" in instructions:
            return _completion(
                {
                    "score": 0.8,
                    "confidence": 0.9,
                    "key_positive": ["Microsoft sales grew"],
                    "key_negative": [],
                }
            )
        assert "AAPL" in instructions
        return _completion(
            {
                "score": -0.8,
                "confidence": 0.9,
                "key_positive": [],
                "key_negative": ["Apple sales fell"],
            }
        )

    with patch.object(
        ext.client.chat.completions, "create", side_effect=respond
    ) as create:
        assert ext.extract_sentiment(msft).score == 0.8
        assert ext.extract_sentiment(aapl).score == -0.8
        assert ext.extract_sentiment(aapl).score == -0.8
        assert ext.extract_sentiment(msft).score == 0.8
        assert create.call_count == 2


@pytest.mark.parametrize("quotes", [[], ["  "], "Apple", ["Apple", 42]])
def test_unusable_quotes_do_not_produce_an_actionable_score(env, quotes):
    _, store, ext = env
    store.add_chunks([_chunk("apple", 0, 0, text="Apple reported higher revenue.")])
    payload = {
        "score": 0.8,
        "confidence": 0.9,
        "key_positive": quotes,
        "key_negative": [],
    }
    with patch.object(
        ext.client.chat.completions, "create", return_value=_completion(payload)
    ):
        response = env[0].get("/sentiment/AAPL")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["latest_score"] is None


@pytest.mark.parametrize(
    "payload",
    [
        [],
        None,
        {
            "score": 1.2,
            "confidence": 0.9,
            "key_positive": ["Apple"],
            "key_negative": [],
        },
        {
            "score": 0.8,
            "confidence": 1.2,
            "key_positive": ["Apple"],
            "key_negative": [],
        },
    ],
)
def test_invalid_model_payload_degrades_without_caching(env, payload):
    _, _, ext = env
    chunk = _chunk("apple", 0, 0, text="Apple reported higher revenue.")
    with patch.object(
        ext.client.chat.completions, "create", return_value=_completion(payload)
    ):
        result, cacheable = ext._score(chunk)
    assert not result.reliable
    assert not cacheable


def test_cache_publish_holds_the_writer_lock(env, monkeypatch):
    _, _, ext = env
    path_type = type(ext.cache_path)
    replace = path_type.replace
    locked_during_publish = []

    def checked_replace(path, target):
        locked_during_publish.append(ext._lock.locked())
        return replace(path, target)

    monkeypatch.setattr(path_type, "replace", checked_replace)
    ext._save_cache()
    assert locked_during_publish == [True]
    assert json.loads(ext.cache_path.read_text()) == {}


@pytest.mark.parametrize(
    "opening",
    [
        "The following discussion should be read in conjunction with the financial statements.",
        "Management’s discussion and analysis of the financial condition and results of operations. "
        "This discussion covers the results for the quarter that just ended and describes the basis of presentation. "
        "The quarterly report should be read together with the preceding annual report.",
    ],
)
def test_mdna_scoring_prefers_substance_over_procedural_opening(env, opening):
    _, _, ext = env
    chunks = [
        _chunk(
            "filing",
            0,
            0,
            text=opening,
        ),
        _chunk("filing", 1, 0, text="Revenue fell because customers reduced orders."),
        _chunk(
            "filing",
            2,
            0,
            text="Operating expenses increased as a result of higher costs.",
        ),
    ]
    assert ext._pick_chunks(chunks) == chunks[1:]
    assert ext._pick_chunks(chunks[:1]) == chunks[:1]


def test_shared_cache_keeps_results_from_independent_extractors(env):
    from alphasignal.generation.sentiment import SentimentExtractor

    _, _, first = env
    config = {
        "generation": {"model": first.model},
        "sentiment": {"cache_path": str(first.cache_path)},
    }
    with patch("alphasignal.generation.sentiment.OpenAI"):
        second = SentimentExtractor(config)
    aapl = _chunk("apple", 0, 0)
    msft = _chunk("microsoft", 0, 0, ticker="MSFT")
    first._score = _scores({aapl.chunk_id: (-0.8, 0.9, True)})
    second._score = _scores({msft.chunk_id: (0.8, 0.9, True)})
    first.extract_sentiment(aapl)
    second.extract_sentiment(msft)
    first._save_cache()
    second._save_cache()
    assert set(json.loads(first.cache_path.read_text())) == {
        first._key(aapl),
        second._key(msft),
    }


def test_simultaneous_cold_requests_share_one_model_result(env):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from time import sleep

    _, _, ext = env
    chunk = _chunk("apple", 0, 0)
    start = Barrier(2)

    def score(_):
        sleep(0.2)
        return SentimentResult(-0.8, 0.9, [], ["matched"], "example"), True

    def request():
        start.wait(timeout=5)
        return ext.extract_sentiment(chunk)

    with patch.object(ext, "_score", side_effect=score) as model:
        with ThreadPoolExecutor(2) as pool:
            results = [pool.submit(request) for _ in range(2)]
            assert [result.result(timeout=5).score for result in results] == [
                -0.8,
                -0.8,
            ]
        assert model.call_count == 1


def test_zero_confidence_evidence_cannot_set_the_gate_score(env):
    client, store, ext = env
    chunk = _chunk("apple", 0, 0)
    store.add_chunks([chunk])
    with patch.object(
        ext, "_score", side_effect=_scores({chunk.chunk_id: (0.9, 0.0, True)})
    ):
        result = client.get("/sentiment/AAPL").json()
    assert result["status"] == "degraded"
    assert result["latest_score"] is None
