"""Tests for the retrieval golden set loading/validation added 2026-08-15
(FINAL_ENGINEERING_AUDIT.md remediation item 4).

Prompt 1's audit believed alphasignal/scripts/benchmark.py and
alphasignal/evaluation/run_eval.py both read the same
evaluation/golden_set.json path with incompatible schemas. That turned out
to be wrong on inspection - benchmark.py resolves its path from
project_root (three parents up from this scripts/ dir, landing on the repo
root's evaluation/ directory) while run_eval.py resolves from its own
evaluation/ package directory, so the two always pointed at two different,
identically-named files. The retrieval file genuinely holds 50 real
question/ticker/relevant_chunk_ids records across 10 tickers, matching the
README's claim - but every relevant_chunk_ids list is empty, so it was
never actually annotated. Both files were renamed to distinct names
(retrieval_golden_set.json / sentiment_golden_set.json) to remove the
identical-filename trap that caused the original misdiagnosis, and
benchmark.py now fails fast with a clear message instead of silently
reporting meaningless all-zero metrics against an unannotated set.
"""

import json

import pytest

from alphasignal.scripts.benchmark import load_and_validate_retrieval_golden_set


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_missing_file_raises_clear_error(tmp_path):
    path = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError, match="no annotated retrieval golden"):
        load_and_validate_retrieval_golden_set(path)


def test_unannotated_golden_set_raises_clear_error_not_silent_zeros(tmp_path):
    """All entries present but every relevant_chunk_ids is empty - exactly
    the real state of evaluation/retrieval_golden_set.json today. Must
    fail loudly rather than let the benchmark loop run and report
    meaningless 0.0 metrics that look like a real finding."""
    path = _write(
        tmp_path,
        "retrieval_golden_set.json",
        [
            {"id": "1", "question": "q1", "ticker": "AAPL", "relevant_chunk_ids": []},
            {"id": "2", "question": "q2", "ticker": "MSFT", "relevant_chunk_ids": []},
        ],
    )

    with pytest.raises(ValueError, match="no annotated retrieval golden"):
        load_and_validate_retrieval_golden_set(path)


def test_malformed_schema_missing_relevant_chunk_ids_raises_clear_error(tmp_path):
    """An entry from the wrong (sentiment-quality) schema, or otherwise
    missing relevant_chunk_ids entirely, must fail with a useful message -
    not an opaque KeyError deep inside the benchmark loop."""
    path = _write(
        tmp_path,
        "retrieval_golden_set.json",
        [
            {
                "ticker": "AAPL",
                "date": "2022-10-28",
                "expected_sentiment": "negative",
                "event_description": "wrong schema entirely",
            }
        ],
    )

    with pytest.raises(ValueError, match="relevant_chunk_ids"):
        load_and_validate_retrieval_golden_set(path)


def test_partially_annotated_golden_set_loads_successfully(tmp_path):
    """Some annotated, some not: should load (main() prints a WARNING for
    the partial case, but loading itself must succeed) - only a fully
    unannotated set is a hard failure."""
    path = _write(
        tmp_path,
        "retrieval_golden_set.json",
        [
            {
                "id": "1",
                "question": "q1",
                "ticker": "AAPL",
                "relevant_chunk_ids": ["chunk_1"],
            },
            {"id": "2", "question": "q2", "ticker": "MSFT", "relevant_chunk_ids": []},
        ],
    )

    golden_set = load_and_validate_retrieval_golden_set(path)

    assert len(golden_set) == 2


def test_real_retrieval_golden_set_is_50_entries_10_tickers_but_unannotated():
    """Ground-truth check against the actual committed file: the README's
    "50 Q&A pairs across 10 tickers" claim is true (previously
    misreported as never having existed at all) - what's missing is
    annotation, not the dataset itself."""
    from pathlib import Path

    real_path = (
        Path(__file__).resolve().parents[2] / "evaluation" / "retrieval_golden_set.json"
    )
    if not real_path.exists():
        pytest.skip("retrieval_golden_set.json not present in this checkout")

    with open(real_path) as f:
        golden_set = json.load(f)

    assert len(golden_set) == 50
    assert len({entry["ticker"] for entry in golden_set}) == 10

    with pytest.raises(ValueError, match="no annotated retrieval golden"):
        load_and_validate_retrieval_golden_set(real_path)
