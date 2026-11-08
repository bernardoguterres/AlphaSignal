"""Comprehensive Unit Tests for AlphaSignal Anti-Dilution Safeguard Policies.

Verifies:
1. Baseline dilution defect reproduction: routine neutral filings at Day 0 dilute an active
   5-day-old crisis (-0.88) from -0.3354 to -0.2071, which inappropriately flips the AlphaLive
   long entry gate (-0.30 threshold) from block to allow.
2. Candidate policies ("adverse_floor", "min_score_persistence", "neutral_capped"):
   safeguards block the diluted SYNTH_B case (keeping score < -0.30).
3. Genuine positive recoveries: post-crisis positive disclosures (+0.85) legitimately offset
   adverse shocks, lifting aggregate score to >= -0.30 without false penalties.
4. Causal ordering: pre-crisis positive disclosures (+0.80 at Day -10) cannot cure subsequent
   crises (-0.88 at Day -5).
5. Natural time decay: crisis from 25 days ago (> 3.5 half-lives) naturally attenuates above
   -0.30, avoiding permanent lockouts.
6. Control scenarios: pure neutral, pure positive, mild negative, and multiple adverse shocks.
7. Zero network calls offline execution guarantee.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from alphasignal.api.schemas import SentimentSignal
from alphasignal.generation.sentiment import (
    SentimentExtractor,
    SentimentResult,
    compute_anti_dilution_aggregate,
)
from alphasignal.ingestion import Chunk

TODAY = datetime.now(timezone.utc).date()
ALPHALIVE_LONG_ENTRY_GATE_THRESHOLD = -0.30


def _make_chunk(doc_id: str, ticker: str, days_ago: int, chunk_index: int = 0) -> Chunk:
    """Helper to generate mock Chunk items."""
    return Chunk(
        chunk_id=f"{doc_id}_{chunk_index:04d}",
        ticker=ticker,
        text=f"Sample disclosure text for {ticker} from {doc_id}",
        token_count=50,
        doc_type="10-Q",
        source="SEC EDGAR",
        section="item_2",
        date=TODAY - timedelta(days=days_ago),
        url=None,
        chunk_index=chunk_index,
        total_chunks=1,
        source_id=doc_id,
        attribution="filer",
    )


def _make_extractor(tmp_path, policy: str | None = None) -> SentimentExtractor:
    """Helper to build SentimentExtractor with configured anti-dilution policy."""
    sentiment_cfg = {
        "freshness_days": 30,
        "half_life_days": 7.0,
        "max_documents": 8,
        "max_chunks_per_document": 2,
        "cache_path": str(tmp_path / "cache.json"),
    }
    if policy is not None:
        sentiment_cfg["anti_dilution_policy"] = policy

    cfg = {
        "generation": {"model": "gpt-5.6-luna"},
        "sentiment": sentiment_cfg,
    }
    return SentimentExtractor(cfg)


# -----------------------------------------------------------------------------
# 1. Baseline Defect Reproduction & Policy Hardening
# -----------------------------------------------------------------------------


def test_baseline_dilution_defect_reproduction(tmp_path, monkeypatch):
    """Reproduce the defect documented in SENTIMENT_EVALUATION.md:
    In SYNTH_B, 1 neutral filing gives -0.3354 (blocking long entry).
    Adding a second distinct neutral filing at Day 0 dilutes the score to -0.2071,
    which flips the AlphaLive long entry gate (-0.30) to allow.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    ext = _make_extractor(tmp_path, policy=None)

    # 1 neutral at Day 0, 1 negative (-0.88) at Day -5
    chunks_1_neutral = [
        _make_chunk("doc0", "SYNTH_B", 0),
        _make_chunk("doc5", "SYNTH_B", 5),
    ]
    mock_scores = {
        "doc0_0000": SentimentResult(0.0, 0.98, [], [], "neutral", reliable=True),
        "doc5_0000": SentimentResult(-0.88, 0.99, [], ["cut"], "negative", reliable=True),
    }
    ext.extract_sentiment = lambda c: mock_scores[c.chunk_id]

    agg_1 = ext.aggregate("SYNTH_B", chunks_1_neutral, as_of=TODAY)
    assert agg_1.status == "ok"
    assert agg_1.score == pytest.approx(-0.3354, abs=1e-4)
    # Long entry action under 1 neutral: BLOCKED (-0.3354 < -0.30)
    assert agg_1.score < ALPHALIVE_LONG_ENTRY_GATE_THRESHOLD

    # 2 neutrals at Day 0, 1 negative (-0.88) at Day -5
    chunks_2_neutrals = [
        _make_chunk("doc0", "SYNTH_B", 0),
        _make_chunk("doc_extra_neutral", "SYNTH_B", 0),
        _make_chunk("doc5", "SYNTH_B", 5),
    ]
    mock_scores["doc_extra_neutral_0000"] = SentimentResult(
        0.0, 0.98, [], [], "neutral boilerplate", reliable=True
    )

    agg_2 = ext.aggregate("SYNTH_B", chunks_2_neutrals, as_of=TODAY)
    assert agg_2.status == "ok"
    assert agg_2.documents_used == 3
    # Dilution defect observed: score jumps to -0.2071
    assert agg_2.score == pytest.approx(-0.2071, abs=1e-4)
    # Long entry action flips to ALLOW under unconstrained baseline (-0.2071 >= -0.30)!
    assert agg_2.score >= ALPHALIVE_LONG_ENTRY_GATE_THRESHOLD
    assert agg_2.dilution_safeguard_active is False


def test_adverse_floor_blocks_diluted_synth_b(tmp_path, monkeypatch):
    """Candidate policy 'adverse_floor':
    The recency-decayed adverse floor prevents neutral boilerplate from diluting
    the active 5-day-old crisis, keeping the score at -0.5364 (< -0.30).
    """
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    ext = _make_extractor(tmp_path, policy="adverse_floor")

    chunks_2_neutrals = [
        _make_chunk("doc0", "SYNTH_B", 0),
        _make_chunk("doc_extra_neutral", "SYNTH_B", 0),
        _make_chunk("doc5", "SYNTH_B", 5),
    ]
    mock_scores = {
        "doc0_0000": SentimentResult(0.0, 0.98, [], [], "neutral", reliable=True),
        "doc_extra_neutral_0000": SentimentResult(0.0, 0.98, [], [], "neutral", reliable=True),
        "doc5_0000": SentimentResult(-0.88, 0.99, [], ["cut"], "negative", reliable=True),
    }
    ext.extract_sentiment = lambda c: mock_scores[c.chunk_id]

    agg = ext.aggregate("SYNTH_B", chunks_2_neutrals, as_of=TODAY)
    assert agg.status == "ok"
    assert agg.raw_score == pytest.approx(-0.2071, abs=1e-4)
    assert agg.dilution_safeguard_active is True
    # Decayed adverse floor: -0.88 * (0.5 ** (5 / 7)) = -0.5364
    assert agg.adverse_floor == pytest.approx(-0.5364, abs=1e-4)
    assert agg.score == pytest.approx(-0.5364, abs=1e-4)
    # Long entry gate: correctly BLOCKED!
    assert agg.score < ALPHALIVE_LONG_ENTRY_GATE_THRESHOLD


def test_min_score_persistence_blocks_diluted_synth_b(tmp_path, monkeypatch):
    """Candidate policy 'min_score_persistence':
    During the active 7-day half-life, the adverse shock's minimum score (-0.88)
    persists against neutral boilerplate dilution.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    ext = _make_extractor(tmp_path, policy="min_score_persistence")

    chunks = [
        _make_chunk("doc0", "SYNTH_B", 0),
        _make_chunk("doc_extra_neutral", "SYNTH_B", 0),
        _make_chunk("doc5", "SYNTH_B", 5),
    ]
    mock_scores = {
        "doc0_0000": SentimentResult(0.0, 0.98, [], [], "neutral", reliable=True),
        "doc_extra_neutral_0000": SentimentResult(0.0, 0.98, [], [], "neutral", reliable=True),
        "doc5_0000": SentimentResult(-0.88, 0.99, [], ["cut"], "negative", reliable=True),
    }
    ext.extract_sentiment = lambda c: mock_scores[c.chunk_id]

    agg = ext.aggregate("SYNTH_B", chunks, as_of=TODAY)
    assert agg.status == "ok"
    assert agg.raw_score == pytest.approx(-0.2071, abs=1e-4)
    assert agg.dilution_safeguard_active is True
    assert agg.adverse_floor == pytest.approx(-0.8800, abs=1e-4)
    assert agg.score == pytest.approx(-0.8800, abs=1e-4)
    # Long entry gate: correctly BLOCKED!
    assert agg.score < ALPHALIVE_LONG_ENTRY_GATE_THRESHOLD


def test_neutral_capped_blocks_diluted_synth_b(tmp_path, monkeypatch):
    """Candidate policy 'neutral_capped':
    Capping total neutral boilerplate weight at 1 filing prevents the denominator
    expansion, keeping score pinned at the single-filing baseline of -0.3354.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    ext = _make_extractor(tmp_path, policy="neutral_capped")

    chunks = [
        _make_chunk("doc0", "SYNTH_B", 0),
        _make_chunk("doc_extra_neutral", "SYNTH_B", 0),
        _make_chunk("doc5", "SYNTH_B", 5),
    ]
    mock_scores = {
        "doc0_0000": SentimentResult(0.0, 0.98, [], [], "neutral", reliable=True),
        "doc_extra_neutral_0000": SentimentResult(0.0, 0.98, [], [], "neutral", reliable=True),
        "doc5_0000": SentimentResult(-0.88, 0.99, [], ["cut"], "negative", reliable=True),
    }
    ext.extract_sentiment = lambda c: mock_scores[c.chunk_id]

    agg = ext.aggregate("SYNTH_B", chunks, as_of=TODAY)
    assert agg.status == "ok"
    assert agg.raw_score == pytest.approx(-0.2071, abs=1e-4)
    assert agg.dilution_safeguard_active is True
    assert agg.score == pytest.approx(-0.3354, abs=1e-4)
    # Long entry gate: correctly BLOCKED!
    assert agg.score < ALPHALIVE_LONG_ENTRY_GATE_THRESHOLD


def test_aggregate_adverse_floor_convenience_method(tmp_path, monkeypatch):
    """Verify that ext.aggregate_adverse_floor(...) provides a direct, ergonomic
    entry point that invokes the adverse_floor safeguard without manual config changes.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    # Base extractor has default policy=None
    ext = _make_extractor(tmp_path, policy=None)

    chunks = [
        _make_chunk("doc0", "SYNTH_B", 0),
        _make_chunk("doc_extra_neutral", "SYNTH_B", 0),
        _make_chunk("doc5", "SYNTH_B", 5),
    ]
    mock_scores = {
        "doc0_0000": SentimentResult(0.0, 0.98, [], [], "neutral", reliable=True),
        "doc_extra_neutral_0000": SentimentResult(0.0, 0.98, [], [], "neutral", reliable=True),
        "doc5_0000": SentimentResult(-0.88, 0.99, [], ["cut"], "negative", reliable=True),
    }
    ext.extract_sentiment = lambda c: mock_scores[c.chunk_id]

    agg = ext.aggregate_adverse_floor("SYNTH_B", chunks, as_of=TODAY)
    assert agg.status == "ok"
    assert agg.score == pytest.approx(-0.5364, abs=1e-4)
    assert agg.dilution_safeguard_active is True
    assert agg.score < ALPHALIVE_LONG_ENTRY_GATE_THRESHOLD


# -----------------------------------------------------------------------------
# 2. Genuine Positive Recovery Protection (No False Penalization)
# -----------------------------------------------------------------------------


def test_genuine_positive_recovery_not_penalized(tmp_path, monkeypatch):
    """Verify that when a genuine positive recovery arrives at Day 0 (score +0.85),
    the adverse floor safeguard does NOT falsely penalize the recovery.
    The directional recovery passes through and allows long entry (score >= -0.30).
    """
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    ext = _make_extractor(tmp_path, policy="adverse_floor")

    chunks = [
        _make_chunk("doc_recovery", "RECOV", 0),
        _make_chunk("doc_crisis", "RECOV", 5),
    ]
    mock_scores = {
        "doc_recovery_0000": SentimentResult(
            0.85, 0.98, ["record rebound"], [], "positive turnaround", reliable=True
        ),
        "doc_crisis_0000": SentimentResult(
            -0.88, 0.99, [], ["investigation"], "severe crisis", reliable=True
        ),
    }
    ext.extract_sentiment = lambda c: mock_scores[c.chunk_id]

    agg = ext.aggregate("RECOV", chunks, as_of=TODAY)
    assert agg.status == "ok"
    # Net directional score: (0.98 * 0.85 + 0.60337 * -0.88) / (0.98 + 0.60337) = +0.1908
    assert agg.score == pytest.approx(0.1908, abs=1e-3)
    assert agg.score > 0.0
    # Long entry is correctly ALLOWED
    assert agg.score >= ALPHALIVE_LONG_ENTRY_GATE_THRESHOLD
    # Safeguard was not forced because genuine positive evidence exists
    assert agg.dilution_safeguard_active is False
    assert agg.adverse_floor is None


def test_positive_recovery_with_neutral_filler_allowed(tmp_path, monkeypatch):
    """Verify that when a positive recovery arrives alongside routine neutral filings,
    the positive recovery is preserved and long entry is ALLOWED.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    ext = _make_extractor(tmp_path, policy="adverse_floor")

    chunks = [
        _make_chunk("doc_recovery", "RECOV", 0),
        _make_chunk("doc_filler1", "RECOV", 0),
        _make_chunk("doc_filler2", "RECOV", 0),
        _make_chunk("doc_crisis", "RECOV", 5),
    ]
    mock_scores = {
        "doc_recovery_0000": SentimentResult(0.85, 0.98, ["cleared"], [], "positive", reliable=True),
        "doc_filler1_0000": SentimentResult(0.0, 0.98, [], [], "neutral", reliable=True),
        "doc_filler2_0000": SentimentResult(0.0, 0.98, [], [], "neutral", reliable=True),
        "doc_crisis_0000": SentimentResult(-0.88, 0.99, [], ["cut"], "negative", reliable=True),
    }
    ext.extract_sentiment = lambda c: mock_scores[c.chunk_id]

    agg = ext.aggregate("RECOV", chunks, as_of=TODAY)
    assert agg.status == "ok"
    # Even with neutral filler, directional pass-through keeps score strongly positive
    assert agg.score >= ALPHALIVE_LONG_ENTRY_GATE_THRESHOLD
    assert agg.score > 0.0
    assert agg.dilution_safeguard_active is False


def test_moderate_positive_recovery_offsets_adverse_shock(tmp_path, monkeypatch):
    """Verify that a moderate positive turnaround (+0.40) dated after the crisis
    lifts the directional score to -0.0878, which is above -0.30 (allowing entry).
    """
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    ext = _make_extractor(tmp_path, policy="adverse_floor")

    chunks = [
        _make_chunk("doc_mod_recov", "MOD", 0),
        _make_chunk("doc_crisis", "MOD", 5),
    ]
    mock_scores = {
        "doc_mod_recov_0000": SentimentResult(0.40, 0.98, ["progress"], [], "moderate positive", reliable=True),
        "doc_crisis_0000": SentimentResult(-0.88, 0.99, [], ["loss"], "crisis", reliable=True),
    }
    ext.extract_sentiment = lambda c: mock_scores[c.chunk_id]

    agg = ext.aggregate("MOD", chunks, as_of=TODAY)
    assert agg.status == "ok"
    # Net directional score: (0.98 * 0.40 + 0.60337 * -0.88) / 1.58337 = -0.0878
    assert agg.score == pytest.approx(-0.0878, abs=1e-3)
    assert agg.score >= ALPHALIVE_LONG_ENTRY_GATE_THRESHOLD
    assert agg.dilution_safeguard_active is False


def test_insufficient_recovery_within_neutral_band_does_not_clear_gate(tmp_path, monkeypatch):
    """Verify that an ambiguous/weak disclosure (+0.05, within neutral band [-0.10, +0.10])
    does not qualify as a genuine positive recovery, so the adverse floor remains active.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    ext = _make_extractor(tmp_path, policy="adverse_floor")

    chunks = [
        _make_chunk("doc_weak", "WEAK", 0),
        _make_chunk("doc_crisis", "WEAK", 5),
    ]
    mock_scores = {
        "doc_weak_0000": SentimentResult(0.05, 0.98, [], [], "weak noise", reliable=True),
        "doc_crisis_0000": SentimentResult(-0.88, 0.99, [], ["loss"], "crisis", reliable=True),
    }
    ext.extract_sentiment = lambda c: mock_scores[c.chunk_id]

    agg = ext.aggregate("WEAK", chunks, as_of=TODAY)
    assert agg.status == "ok"
    # Floor enforces decayed shock (-0.5364), keeping long BLOCKED
    assert agg.score == pytest.approx(-0.5364, abs=1e-4)
    assert agg.score < ALPHALIVE_LONG_ENTRY_GATE_THRESHOLD
    assert agg.dilution_safeguard_active is True


# -----------------------------------------------------------------------------
# 3. Temporal Invariants, Causal Ordering, and Time Decay
# -----------------------------------------------------------------------------


def test_prior_positive_news_does_not_cure_subsequent_crisis(tmp_path, monkeypatch):
    """Causal ordering invariant:
    Positive disclosures at Day -10 (+0.80) occurred BEFORE a fraud crisis at Day -5 (-0.88).
    Subsequent routine filings at Day 0 must NOT use the prior earnings to dilute the crisis.
    The safeguard must treat the crisis as unrecovered and block long entry.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    ext = _make_extractor(tmp_path, policy="adverse_floor")

    chunks = [
        _make_chunk("doc_prior_good", "CAUSAL", 10),
        _make_chunk("doc_crisis", "CAUSAL", 5),
        _make_chunk("doc_neutral1", "CAUSAL", 0),
        _make_chunk("doc_neutral2", "CAUSAL", 0),
    ]
    mock_scores = {
        "doc_prior_good_0000": SentimentResult(0.80, 0.98, ["beat"], [], "old beat", reliable=True),
        "doc_crisis_0000": SentimentResult(-0.88, 0.99, [], ["fraud"], "subsequent fraud", reliable=True),
        "doc_neutral1_0000": SentimentResult(0.0, 0.98, [], [], "proxy", reliable=True),
        "doc_neutral2_0000": SentimentResult(0.0, 0.98, [], [], "8-K", reliable=True),
    }
    ext.extract_sentiment = lambda c: mock_scores[c.chunk_id]

    agg = ext.aggregate("CAUSAL", chunks, as_of=TODAY)
    assert agg.status == "ok"
    # Prior positive cannot cure subsequent fraud; floor holds long BLOCKED
    assert agg.score == pytest.approx(-0.5364, abs=1e-4)
    assert agg.score < ALPHALIVE_LONG_ENTRY_GATE_THRESHOLD
    assert agg.dilution_safeguard_active is True


def test_natural_half_life_decay_relaxes_after_time(tmp_path, monkeypatch):
    """Time decay invariant:
    An adverse shock (-0.88) from 25 days ago (> 3.5 half-lives) has naturally decayed:
    -0.88 * 0.5 ** (25 / 7) = -0.0744 > -0.30.
    The safeguard does not permanently freeze issuers in crisis; after sufficient time
    without further negative news, the gate naturally relaxes (ALLOW).
    """
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    ext = _make_extractor(tmp_path, policy="adverse_floor")

    chunks = [
        _make_chunk("doc_old_crisis", "OLD", 25),
        _make_chunk("doc_neutral", "OLD", 0),
    ]
    mock_scores = {
        "doc_old_crisis_0000": SentimentResult(-0.88, 0.99, [], ["past fine"], "old fine", reliable=True),
        "doc_neutral_0000": SentimentResult(0.0, 0.98, [], [], "routine", reliable=True),
    }
    ext.extract_sentiment = lambda c: mock_scores[c.chunk_id]

    agg = ext.aggregate("OLD", chunks, as_of=TODAY)
    assert agg.status == "ok"
    # Decayed shock is -0.0744 > -0.30; long entry is ALLOWED
    assert agg.score >= ALPHALIVE_LONG_ENTRY_GATE_THRESHOLD
    assert agg.score > -0.15


# -----------------------------------------------------------------------------
# 4. Control Scenarios & Edge Cases
# -----------------------------------------------------------------------------


def test_all_neutral_filings_unaffected(tmp_path, monkeypatch):
    """When a ticker has only neutral boilerplate filings, score is 0.0,
    safeguard is inactive, and long entry is ALLOWED.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    ext = _make_extractor(tmp_path, policy="adverse_floor")

    chunks = [
        _make_chunk("doc1", "NEUT", 0),
        _make_chunk("doc2", "NEUT", 2),
    ]
    mock_scores = {
        "doc1_0000": SentimentResult(0.0, 0.98, [], [], "routine 8-K", reliable=True),
        "doc2_0000": SentimentResult(0.0, 0.95, [], [], "routine proxy", reliable=True),
    }
    ext.extract_sentiment = lambda c: mock_scores[c.chunk_id]

    agg = ext.aggregate("NEUT", chunks, as_of=TODAY)
    assert agg.status == "ok"
    assert agg.score == pytest.approx(0.0, abs=1e-4)
    assert agg.dilution_safeguard_active is False
    assert agg.adverse_floor is None
    assert agg.score >= ALPHALIVE_LONG_ENTRY_GATE_THRESHOLD


def test_all_positive_filings_unaffected(tmp_path, monkeypatch):
    """When a ticker has only positive filings, score is positive,
    safeguard is inactive, and long entry is ALLOWED.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    ext = _make_extractor(tmp_path, policy="adverse_floor")

    chunks = [
        _make_chunk("doc1", "POS", 0),
        _make_chunk("doc2", "POS", 3),
    ]
    mock_scores = {
        "doc1_0000": SentimentResult(0.85, 0.98, ["strong demand"], [], "positive", reliable=True),
        "doc2_0000": SentimentResult(0.90, 0.95, ["record sales"], [], "positive", reliable=True),
    }
    ext.extract_sentiment = lambda c: mock_scores[c.chunk_id]

    agg = ext.aggregate("POS", chunks, as_of=TODAY)
    assert agg.status == "ok"
    assert agg.score > 0.80
    assert agg.dilution_safeguard_active is False
    assert agg.score >= ALPHALIVE_LONG_ENTRY_GATE_THRESHOLD


def test_mild_negative_filings_do_not_trigger_false_crisis(tmp_path, monkeypatch):
    """Ordinary business friction (score -0.15, above -0.30 adverse threshold)
    does not classify as a material adverse shock and does not trigger the floor.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    ext = _make_extractor(tmp_path, policy="adverse_floor")

    chunks = [
        _make_chunk("doc1", "MILD", 0),
        _make_chunk("doc2", "MILD", 1),
    ]
    mock_scores = {
        "doc1_0000": SentimentResult(-0.15, 0.95, [], ["supply delay"], "mild headwind", reliable=True),
        "doc2_0000": SentimentResult(0.0, 0.95, [], [], "routine", reliable=True),
    }
    ext.extract_sentiment = lambda c: mock_scores[c.chunk_id]

    agg = ext.aggregate("MILD", chunks, as_of=TODAY)
    assert agg.status == "ok"
    assert agg.dilution_safeguard_active is False
    assert agg.score >= ALPHALIVE_LONG_ENTRY_GATE_THRESHOLD


def test_multiple_adverse_shocks_use_worst_floor(tmp_path, monkeypatch):
    """When multiple material adverse shocks occur (-0.88 and -0.95),
    the adverse floor enforces the most severe shock.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    ext = _make_extractor(tmp_path, policy="adverse_floor")

    chunks = [
        _make_chunk("doc0", "MULTI", 0),
        _make_chunk("doc_shock1", "MULTI", 5),
        _make_chunk("doc_shock2", "MULTI", 2),
    ]
    # shock1: -0.88 at Day 5 -> -0.88 * 0.5 ** (5 / 7) = -0.5364
    # shock2: -0.95 at Day 2 -> -0.95 * 0.5 ** (2 / 7) = -0.7797
    mock_scores = {
        "doc0_0000": SentimentResult(0.0, 0.98, [], [], "routine", reliable=True),
        "doc_shock1_0000": SentimentResult(-0.88, 0.99, [], ["fine"], "shock1", reliable=True),
        "doc_shock2_0000": SentimentResult(-0.95, 0.99, [], ["fraud"], "shock2", reliable=True),
    }
    ext.extract_sentiment = lambda c: mock_scores[c.chunk_id]

    agg = ext.aggregate("MULTI", chunks, as_of=TODAY)
    assert agg.status == "ok"
    assert agg.dilution_safeguard_active is True
    # Worst decayed floor is -0.7797
    assert agg.adverse_floor == pytest.approx(-0.7797, abs=1e-3)
    assert agg.score == pytest.approx(-0.7797, abs=1e-3)
    assert agg.score < ALPHALIVE_LONG_ENTRY_GATE_THRESHOLD


# -----------------------------------------------------------------------------
# 5. Standalone Function & Offline Guarantee
# -----------------------------------------------------------------------------


def test_standalone_compute_anti_dilution_aggregate():
    """Verify compute_anti_dilution_aggregate directly with mock SentimentSignal items."""
    sig_crisis = SentimentSignal(
        ticker="SYNTH_B",
        date=TODAY - timedelta(days=5),
        score=-0.88,
        confidence=0.99,
        source="SEC EDGAR",
        doc_type="8-K",
        summary="crisis",
        key_positive=[],
        key_negative=["crisis"],
    )
    sig_neutral1 = SentimentSignal(
        ticker="SYNTH_B",
        date=TODAY,
        score=0.0,
        confidence=0.98,
        source="SEC EDGAR",
        doc_type="8-K",
        summary="neutral",
        key_positive=[],
        key_negative=[],
    )
    sig_neutral2 = SentimentSignal(
        ticker="SYNTH_B",
        date=TODAY,
        score=0.0,
        confidence=0.98,
        source="SEC EDGAR",
        doc_type="DEF 14A",
        summary="neutral proxy",
        key_positive=[],
        key_negative=[],
    )
    signals = [sig_neutral1, sig_neutral2, sig_crisis]

    # Baseline (policy=None)
    final_score, raw_score, floor, active = compute_anti_dilution_aggregate(
        signals=signals,
        as_of=TODAY,
        half_life_days=7.0,
        policy=None,
    )
    assert final_score == pytest.approx(-0.2071, abs=1e-4)
    assert raw_score == pytest.approx(-0.2071, abs=1e-4)
    assert active is False
    assert floor is None

    # Adverse floor safeguard
    final_score, raw_score, floor, active = compute_anti_dilution_aggregate(
        signals=signals,
        as_of=TODAY,
        half_life_days=7.0,
        policy="adverse_floor",
    )
    assert final_score == pytest.approx(-0.5364, abs=1e-4)
    assert raw_score == pytest.approx(-0.2071, abs=1e-4)
    assert floor == pytest.approx(-0.5364, abs=1e-4)
    assert active is True


def test_offline_zero_network_calls(tmp_path, monkeypatch):
    """Verify all tests and extraction execute strictly offline with zero network calls."""
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    # Poison OpenAI client to guarantee any unmocked call immediately crashes
    poisoned_client = MagicMock()
    poisoned_client.chat.completions.create.side_effect = AssertionError(
        "Network call attempted! Anti-dilution tests must run strictly offline."
    )

    ext = _make_extractor(tmp_path, policy="adverse_floor")
    ext.client = poisoned_client

    chunks = [
        _make_chunk("doc0", "SYNTH_B", 0),
        _make_chunk("doc5", "SYNTH_B", 5),
    ]
    # Mock extract_sentiment so zero network requests are made
    ext.extract_sentiment = lambda c: SentimentResult(
        -0.88 if "doc5" in c.chunk_id else 0.0,
        0.98,
        [],
        [],
        "mocked offline result",
        reliable=True,
    )

    agg = ext.aggregate("SYNTH_B", chunks, as_of=TODAY)
    assert agg.status == "ok"
    poisoned_client.chat.completions.create.assert_not_called()
