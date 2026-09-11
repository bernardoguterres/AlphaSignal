"""Deterministic tests for evaluation-leakage fixes in run_eval.py (objective 5).

Covers: unavailable predictions must never be scored by substituting
expected_sentiment, accuracy/Sharpe must exclude unavailable predictions,
coverage must be reported separately and explicitly, and the module must be
explicit about evaluating retrieved corpus context (not raw event text).
"""

import json
from unittest.mock import patch

from alphasignal.evaluation import run_eval
from alphasignal.evaluation.run_eval import _is_correct_prediction, run_evaluation


def test_is_correct_prediction_unavailable_is_indeterminate_not_scored():
    """An unavailable prediction must never be graded - it's neither
    correct nor incorrect, regardless of the actual return or any ground
    truth label."""
    assert _is_correct_prediction("unavailable", 5.0) is None
    assert _is_correct_prediction("unavailable", -5.0) is None
    assert _is_correct_prediction("unavailable", None) is None


def test_is_correct_prediction_neutral_excluded_from_accuracy():
    assert _is_correct_prediction("neutral", 5.0) is None


def test_is_correct_prediction_genuine_prediction_scored_normally():
    assert _is_correct_prediction("positive", 5.0) is True
    assert _is_correct_prediction("positive", -5.0) is False
    assert _is_correct_prediction("negative", -5.0) is True


def _write_golden_set(tmp_path, entries):
    path = tmp_path / "sentiment_golden_set.json"
    path.write_text(json.dumps(entries))
    return path


def _run(tmp_path, entries, returns_by_ticker, predictions_by_ticker, capsys):
    golden_path = _write_golden_set(tmp_path, entries)

    def fake_return(ticker, event_date):
        return returns_by_ticker.get(ticker)

    def fake_predict(
        ticker, event_description, event_date, base_url, timeout=60, api_key=None
    ):
        return predictions_by_ticker.get(ticker, "unavailable")

    with patch.object(
        run_eval, "_compute_5d_forward_return", side_effect=fake_return
    ), patch.object(run_eval, "_fetch_predicted_sentiment", side_effect=fake_predict):
        results = run_evaluation(golden_set_path=golden_path, save_results=False)

    captured = capsys.readouterr()
    return results, captured.out


def test_run_evaluation_unavailable_prediction_not_substituted_with_ground_truth(
    tmp_path, capsys
):
    """An entry whose prediction is unavailable must be recorded as
    'unavailable' (not silently replaced by expected_sentiment) and scored
    as indeterminate (correct=None), even though the ground truth is
    directional and the return would "confirm" it."""
    entries = [
        {
            "ticker": "AAPL",
            "date": "2024-01-05",
            "expected_sentiment": "positive",
            "event_description": "Strong earnings beat.",
        }
    ]
    results, out = _run(
        tmp_path,
        entries,
        returns_by_ticker={"AAPL": 4.2},  # return "confirms" positive
        predictions_by_ticker={},  # no prediction available
        capsys=capsys,
    )

    assert results[0]["predicted_sentiment"] == "unavailable"
    assert results[0]["correct"] is None
    assert "N/A (no genuine, non-neutral, available predictions" in out


def test_run_evaluation_zero_coverage_reports_na_not_fabricated_accuracy(
    tmp_path, capsys
):
    """When every prediction is unavailable, accuracy must be N/A, not a
    fabricated 100% (or any) accuracy derived from ground truth."""
    entries = [
        {
            "ticker": "AAPL",
            "date": "2024-01-05",
            "expected_sentiment": "positive",
            "event_description": "Event A.",
        },
        {
            "ticker": "MSFT",
            "date": "2024-02-01",
            "expected_sentiment": "negative",
            "event_description": "Event B.",
        },
    ]
    results, out = _run(
        tmp_path,
        entries,
        returns_by_ticker={"AAPL": 3.0, "MSFT": -2.0},
        predictions_by_ticker={},
        capsys=capsys,
    )

    assert all(r["correct"] is None for r in results)
    assert "Prediction coverage (actionable): 0/2" in out
    assert "Directional accuracy:            N/A" in out


def test_run_evaluation_partial_coverage_denominator_excludes_unavailable(
    tmp_path, capsys
):
    """With one genuine correct prediction and one unavailable prediction,
    accuracy must be 100% over a denominator of 1, not diluted or inflated
    by the unavailable row."""
    entries = [
        {
            "ticker": "AAPL",
            "date": "2024-01-05",
            "expected_sentiment": "positive",
            "event_description": "Event A.",
        },
        {
            "ticker": "MSFT",
            "date": "2024-02-01",
            "expected_sentiment": "negative",
            "event_description": "Event B.",
        },
    ]
    results, out = _run(
        tmp_path,
        entries,
        returns_by_ticker={"AAPL": 5.0, "MSFT": -3.0},
        predictions_by_ticker={"AAPL": "positive"},  # MSFT stays unavailable
        capsys=capsys,
    )

    aapl_result = next(r for r in results if r["ticker"] == "AAPL")
    msft_result = next(r for r in results if r["ticker"] == "MSFT")

    assert aapl_result["correct"] is True
    assert msft_result["correct"] is None
    assert "Prediction coverage (actionable): 1/2" in out
    assert "Directional accuracy:            100.0%  (1/1 correct" in out


def test_run_evaluation_sharpe_uses_predicted_not_ground_truth_sentiment(
    tmp_path, capsys
):
    """The Sharpe/return-based signal must be built from predicted_sentiment,
    not expected_sentiment - a wrong prediction against a golden-set label
    that happens to be right must not be silently "corrected" via the label."""
    entries = [
        {
            "ticker": "AAPL",
            "date": "2024-01-05",
            "expected_sentiment": "positive",  # ground truth says positive
            "event_description": "Event A.",
        },
        {
            "ticker": "MSFT",
            "date": "2024-02-01",
            "expected_sentiment": "negative",  # ground truth says negative
            "event_description": "Event B.",
        },
    ]
    # Model predicted the OPPOSITE of ground truth for both, and the market
    # agreed with the model, not the label - both should count as CORRECT
    # predictions, which is only possible if the code uses
    # predicted_sentiment, not expected_sentiment, to build the signal.
    results, out = _run(
        tmp_path,
        entries,
        returns_by_ticker={"AAPL": -6.0, "MSFT": 4.0},
        predictions_by_ticker={"AAPL": "negative", "MSFT": "positive"},
        capsys=capsys,
    )

    aapl_result = next(r for r in results if r["ticker"] == "AAPL")
    msft_result = next(r for r in results if r["ticker"] == "MSFT")
    assert aapl_result["predicted_sentiment"] == "negative"
    assert aapl_result["correct"] is True
    assert msft_result["predicted_sentiment"] == "positive"
    assert msft_result["correct"] is True
    assert "n=2 genuine non-neutral predictions" in out


def test_evaluation_design_is_explicit_about_corpus_context_not_event_text():
    """The module must document, in a way a reader can find, that it scores
    retrieved corpus context around the event date rather than the golden
    set's event_description text directly."""
    doc = run_eval.__doc__ or ""
    assert "RETRIEVED CORPUS CHUNKS" in doc
    assert "event_description" in doc
