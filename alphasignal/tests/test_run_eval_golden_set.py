"""Regression test for FINAL_ENGINEERING_AUDIT.md remediation item 4:
evaluation/run_eval.py (sentiment-quality eval) and
scripts/benchmark.py (retrieval eval) must each resolve to their own,
distinctly-named golden set file - see test_benchmark_golden_set.py for
the retrieval side.
"""

import json

from alphasignal.evaluation.run_eval import _GOLDEN_SET_PATH


def test_golden_set_path_points_at_sentiment_file_not_retrieval_file():
    assert _GOLDEN_SET_PATH.name == "sentiment_golden_set.json"
    assert _GOLDEN_SET_PATH.parent.name == "evaluation"
    # Package-internal evaluation/ dir, not the repo-root one retrieval
    # benchmarking uses.
    assert _GOLDEN_SET_PATH.parent.parent.name == "alphasignal"


def test_bundled_sentiment_golden_set_has_sentiment_schema_not_retrieval_schema():
    if not _GOLDEN_SET_PATH.exists():
        import pytest

        pytest.skip("sentiment_golden_set.json not present in this checkout")

    with open(_GOLDEN_SET_PATH) as f:
        golden_set = json.load(f)

    assert len(golden_set) > 0
    for entry in golden_set:
        assert "ticker" in entry
        assert "date" in entry
        assert "expected_sentiment" in entry
        assert "event_description" in entry
        # Retrieval-schema fields must NOT be present - if they are, this
        # file has drifted back into holding the wrong dataset.
        assert "question" not in entry
        assert "relevant_chunk_ids" not in entry
