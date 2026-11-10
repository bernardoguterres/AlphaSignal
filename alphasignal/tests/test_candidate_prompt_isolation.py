"""Test Candidate Prompt Isolation & Recency Aggregation Invariants."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from alphasignal.generation.candidate_prompt import (
    CANDIDATE_PROMPT_VERSION,
    CANDIDATE_SYSTEM_PROMPT,
    CANDIDATE_USER_PROMPT_TEMPLATE,
    format_candidate_prompt,
)
from alphasignal.generation.sentiment import (
    PROMPT_VERSION,
    SentimentExtractor,
    SentimentResult,
)
from alphasignal.ingestion import Chunk

TODAY = datetime.now(timezone.utc).date()


def _make_chunk(doc_id: str, ticker: str, days_ago: int, chunk_index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}_{chunk_index:04d}",
        ticker=ticker,
        text=f"Sample disclosure text for {ticker}",
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


def test_production_prompt_version_isolation():
    """Ensure production SentimentExtractor configuration remains at v2."""
    assert PROMPT_VERSION == 2
    assert CANDIDATE_PROMPT_VERSION == 3
    assert "TARGET TICKER" in CANDIDATE_SYSTEM_PROMPT
    assert "{target_ticker}" in CANDIDATE_USER_PROMPT_TEMPLATE


def test_format_candidate_prompt():
    """Ensure candidate prompt injects the target ticker correctly."""
    messages = format_candidate_prompt("AAPL", "Revenue grew 10%")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "AAPL" in messages[1]["content"]
    assert "Revenue grew 10%" in messages[1]["content"]


def test_recency_time_shift_invariance(tmp_path, monkeypatch):
    """Advancing as_of with identical documents multiplies weights by the same scalar factor,
    keeping the aggregate score identical as long as documents remain in the freshness window.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    cfg = {
        "generation": {"model": "gpt-5.6-luna"},
        "sentiment": {
            "freshness_days": 30,
            "half_life_days": 7.0,
            "max_documents": 8,
            "max_chunks_per_document": 2,
            "cache_path": str(tmp_path / "cache.json"),
        },
    }
    ext = SentimentExtractor(cfg)

    # SYNTH_B setup: doc0 at Day 0 (neutral 0.0), doc5 at Day -5 (negative -0.88)
    chunks = [
        _make_chunk("doc0", "SYNTH_B", 0),
        _make_chunk("doc5", "SYNTH_B", 5),
    ]
    mock_scores = {
        "doc0_0000": SentimentResult(0.0, 0.98, [], [], "neutral", reliable=True),
        "doc5_0000": SentimentResult(-0.88, 0.99, [], ["cut"], "negative", reliable=True),
    }
    ext.extract_sentiment = lambda c: mock_scores[c.chunk_id]

    agg_today = ext.aggregate("SYNTH_B", chunks, as_of=TODAY)
    assert agg_today.status == "ok"
    assert agg_today.score == pytest.approx(-0.3354, abs=1e-4)

    # Advance as_of by 7 days: both documents age by 7 days (weights both scale by 0.5)
    agg_7d_later = ext.aggregate("SYNTH_B", chunks, as_of=TODAY + timedelta(days=7))
    assert agg_7d_later.status == "ok"
    assert agg_7d_later.score == pytest.approx(agg_today.score, abs=1e-4)


def test_recency_dilution_by_second_neutral_filing(tmp_path, monkeypatch):
    """Adding a second distinct, eligible neutral filing at Day 0 dilutes the adverse
    event weight and moves the aggregate score from -0.3354 to -0.2071 (above -0.30).
    """
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    cfg = {
        "generation": {"model": "gpt-5.6-luna"},
        "sentiment": {
            "freshness_days": 30,
            "half_life_days": 7.0,
            "max_documents": 8,
            "max_chunks_per_document": 2,
            "cache_path": str(tmp_path / "cache.json"),
        },
    }
    ext = SentimentExtractor(cfg)

    chunks_with_extra_neutral = [
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

    agg = ext.aggregate("SYNTH_B", chunks_with_extra_neutral, as_of=TODAY)
    assert agg.status == "ok"
    assert agg.documents_used == 3
    assert agg.score == pytest.approx(-0.2071, abs=1e-4)


def test_dry_run_output_is_labelled_simulated(tmp_path):
    """Dry-run metrics are true by construction; the output must say so and name no model."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    out = tmp_path / "dry.json"
    subprocess.run(
        [sys.executable, str(root / "evaluation" / "sentiment_eval_runner.py"), "--dry-run", "--output", str(out)],
        check=True, capture_output=True, cwd=root,
    )
    summary = json.loads(out.read_text())["summary"]
    assert summary["mode"] == "dry-run"
    assert summary["evidence_class"] == "simulated_by_construction_not_model_quality"
    assert summary["model"] is None
    assert summary["budget_usage"]["total_requests"] == 0


def test_candidate_prompt_not_imported_by_production_code():
    """v3 stays evaluation-only: no module in the alphasignal package (other than the
    candidate module itself and tests) may import it."""
    import ast
    from pathlib import Path

    pkg = Path(__file__).resolve().parents[1]
    offenders = []
    for path in pkg.rglob("*.py"):
        if path.name == "candidate_prompt.py" or "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            if any("candidate_prompt" in n for n in names):
                offenders.append(str(path.relative_to(pkg)))
    assert offenders == []


def test_archived_paid_claims_are_not_reconstructable():
    """The archived 'paid' rows are simulator output and the reported F1 does not
    reproduce (offline; needs the full git history, so skipped in shallow clones)."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    probe = subprocess.run(["git", "cat-file", "-e", "7688fa6^{commit}"], cwd=root, capture_output=True)
    if probe.returncode != 0:
        pytest.skip("commit 7688fa6 not available (shallow clone)")
    res = subprocess.run(
        [sys.executable, str(root / "evaluation" / "verify_archived_paid_claims.py")],
        cwd=root, capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "100/100 rows identical" in res.stdout
