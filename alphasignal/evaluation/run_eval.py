"""Evaluation runner for AlphaSignal quality framework.

Tests whether positive AlphaSignal sentiment predicts positive 5-day forward returns.
Works standalone without a running AlphaSignal API — if the API is unavailable,
sentiment predictions are marked 'unavailable' and only return data is shown.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.error import URLError
from urllib.request import urlopen, Request as URLRequest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_EVAL_DIR = Path(__file__).parent
_GOLDEN_SET_PATH = _EVAL_DIR / "golden_set.json"
_RESULTS_DIR = _EVAL_DIR  # save results alongside the golden set


# ---------------------------------------------------------------------------
# Price data via yfinance (optional dep) or urllib fallback
# ---------------------------------------------------------------------------

def _fetch_price_data_yfinance(ticker: str, start: date, end: date) -> dict[date, float]:
    """Return {trading_date: adj_close} using yfinance."""
    import yfinance as yf  # type: ignore

    extra_days = timedelta(days=14)  # buffer for weekends/holidays
    df = yf.download(
        ticker,
        start=(start - timedelta(days=1)).isoformat(),
        end=(end + extra_days).isoformat(),
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        return {}

    prices: dict[date, float] = {}
    for ts, row in df.iterrows():
        d = ts.date() if hasattr(ts, "date") else ts
        close_val = row["Close"]
        # Handle both scalar and Series (multi-ticker edge case)
        if hasattr(close_val, "iloc"):
            close_val = float(close_val.iloc[0])
        else:
            close_val = float(close_val)
        if not math.isnan(close_val):
            prices[d] = close_val
    return prices


def _get_nearest_trading_day(prices: dict[date, float], target: date, max_offset: int = 5) -> Optional[date]:
    """Return the nearest available trading day at or after target within max_offset days."""
    for offset in range(max_offset + 1):
        candidate = target + timedelta(days=offset)
        if candidate in prices:
            return candidate
    return None


def _compute_5d_forward_return(
    ticker: str,
    event_date: date,
) -> Optional[float]:
    """Compute 5-trading-day forward return starting from event_date.

    Returns the percentage return from event_date close to the close
    5 trading days later, or None if data is unavailable.
    """
    try:
        prices = _fetch_price_data_yfinance(ticker, event_date, event_date + timedelta(days=30))
    except ImportError:
        logger.warning("yfinance not installed. Run: pip install yfinance")
        return None
    except Exception as exc:
        logger.warning("Price fetch failed for %s on %s: %s", ticker, event_date, exc)
        return None

    if not prices:
        logger.warning("No price data returned for %s", ticker)
        return None

    sorted_dates = sorted(prices)

    # Entry: nearest trading day on or after event_date
    entry_date = _get_nearest_trading_day(prices, event_date)
    if entry_date is None:
        logger.warning("No entry trading day found for %s around %s", ticker, event_date)
        return None

    entry_idx = sorted_dates.index(entry_date)
    # Exit: 5 trading days after entry
    exit_idx = entry_idx + 5
    if exit_idx >= len(sorted_dates):
        logger.warning("Insufficient price history for 5-day forward return (%s, %s)", ticker, event_date)
        return None

    exit_date = sorted_dates[exit_idx]
    entry_price = prices[entry_date]
    exit_price = prices[exit_date]

    if entry_price == 0:
        return None

    pct_return = (exit_price - entry_price) / entry_price * 100.0
    return round(pct_return, 4)


# ---------------------------------------------------------------------------
# AlphaSignal API client
# ---------------------------------------------------------------------------

def _fetch_predicted_sentiment(
    ticker: str,
    event_description: str,
    base_url: str = "http://localhost:8000",
    timeout: int = 5,
) -> str:
    """Call the AlphaSignal sentiment endpoint and return 'positive', 'negative', or 'neutral'.

    Falls back to 'unavailable' if the API is not reachable.

    The endpoint returns a SentimentResponse with a latest_score float in [-1, 1].
    We map: score > 0.1 -> positive, score < -0.1 -> negative, else -> neutral.
    """
    from urllib.parse import quote

    url = f"{base_url}/sentiment/{ticker}?query={quote(event_description)}"
    req = URLRequest(url, headers={"Accept": "application/json"})

    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
    except Exception:
        # API not running or error — graceful fallback
        return "unavailable"

    latest_score = data.get("latest_score")
    if latest_score is None:
        return "unavailable"

    score = float(latest_score)
    if score > 0.1:
        return "positive"
    elif score < -0.1:
        return "negative"
    else:
        return "neutral"


# ---------------------------------------------------------------------------
# Results table printing
# ---------------------------------------------------------------------------

def _sentiment_to_direction(sentiment: str) -> Optional[int]:
    """Map sentiment label to +1 (positive), -1 (negative), 0 (neutral), None (unavailable)."""
    mapping = {"positive": 1, "negative": -1, "neutral": 0, "unavailable": None}
    return mapping.get(sentiment.lower())


def _return_to_direction(ret: Optional[float]) -> Optional[int]:
    """Map return to +1 (positive), -1 (negative), or None."""
    if ret is None:
        return None
    if ret > 0:
        return 1
    elif ret < 0:
        return -1
    else:
        return 0


def _is_correct_prediction(expected_sentiment: str, actual_return: Optional[float]) -> Optional[bool]:
    """True if sentiment direction matched return direction, False otherwise, None if indeterminate."""
    exp_dir = _sentiment_to_direction(expected_sentiment)
    ret_dir = _return_to_direction(actual_return)
    if exp_dir is None or ret_dir is None:
        return None
    if exp_dir == 0:
        return None  # neutral signals are excluded from accuracy
    return exp_dir == ret_dir


def _print_results_table(results: list[dict[str, Any]]) -> None:
    """Print a formatted results table to stdout."""
    col_widths = {
        "ticker": 6,
        "date": 10,
        "event": 38,
        "expected": 9,
        "predicted": 11,
        "return": 9,
        "correct": 8,
    }

    header = (
        f"{'Ticker':<{col_widths['ticker']}} "
        f"{'Date':<{col_widths['date']}} "
        f"{'Event':<{col_widths['event']}} "
        f"{'Expected':<{col_widths['expected']}} "
        f"{'Predicted':<{col_widths['predicted']}} "
        f"{'5d Ret%':<{col_widths['return']}} "
        f"{'Correct?':<{col_widths['correct']}}"
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print(header)
    print(separator)

    for r in results:
        event_short = (r["event_description"][:35] + "...") if len(r["event_description"]) > 38 else r["event_description"]
        ret_str = f"{r['actual_5d_return_pct']:+.2f}%" if r["actual_5d_return_pct"] is not None else "N/A"
        correct_str = str(r["correct"]) if r["correct"] is not None else "-"

        print(
            f"{r['ticker']:<{col_widths['ticker']}} "
            f"{r['date']:<{col_widths['date']}} "
            f"{event_short:<{col_widths['event']}} "
            f"{r['expected_sentiment']:<{col_widths['expected']}} "
            f"{r['predicted_sentiment']:<{col_widths['predicted']}} "
            f"{ret_str:<{col_widths['return']}} "
            f"{correct_str:<{col_widths['correct']}}"
        )

    print(separator + "\n")


def _print_summary(results: list[dict[str, Any]]) -> None:
    """Compute and print evaluation summary statistics."""
    # Filter entries where we have return data
    with_returns = [r for r in results if r["actual_5d_return_pct"] is not None]
    # Filter entries with actionable sentiment (positive or negative, not unavailable/neutral)
    actionable = [
        r for r in with_returns
        if r["expected_sentiment"] in ("positive", "negative")
    ]

    # Directional accuracy (using expected_sentiment as proxy when predicted unavailable)
    correctness_list = [r["correct"] for r in actionable if r["correct"] is not None]
    if correctness_list:
        accuracy = sum(correctness_list) / len(correctness_list) * 100
    else:
        accuracy = None

    # Avg returns by expected sentiment direction
    positive_returns = [
        r["actual_5d_return_pct"]
        for r in with_returns
        if r["expected_sentiment"] == "positive" and r["actual_5d_return_pct"] is not None
    ]
    negative_returns = [
        r["actual_5d_return_pct"]
        for r in with_returns
        if r["expected_sentiment"] == "negative" and r["actual_5d_return_pct"] is not None
    ]

    avg_pos = sum(positive_returns) / len(positive_returns) if positive_returns else None
    avg_neg = sum(negative_returns) / len(negative_returns) if negative_returns else None

    # Sharpe of using sentiment as a signal
    # Long on positive sentiment events, short on negative sentiment events
    signal_returns: list[float] = []
    for r in actionable:
        ret = r["actual_5d_return_pct"]
        if ret is None:
            continue
        direction = _sentiment_to_direction(r["expected_sentiment"])
        if direction is not None and direction != 0:
            signal_returns.append(direction * ret)

    if len(signal_returns) >= 2:
        mean_ret = sum(signal_returns) / len(signal_returns)
        variance = sum((x - mean_ret) ** 2 for x in signal_returns) / (len(signal_returns) - 1)
        std_ret = math.sqrt(variance) if variance > 0 else None
        sharpe = mean_ret / std_ret if std_ret else None
    else:
        sharpe = None

    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total events evaluated:          {len(results)}")
    print(f"Events with return data:         {len(with_returns)}")
    print(f"Actionable signals (pos/neg):    {len(actionable)}")

    if accuracy is not None:
        print(f"Directional accuracy:            {accuracy:.1f}%  ({sum(correctness_list)}/{len(correctness_list)} correct)")
    else:
        print("Directional accuracy:            N/A (predicted sentiment unavailable)")

    if avg_pos is not None:
        print(f"Avg 5d return (positive signals): {avg_pos:+.2f}%  (n={len(positive_returns)})")
    else:
        print("Avg 5d return (positive signals): N/A")

    if avg_neg is not None:
        print(f"Avg 5d return (negative signals): {avg_neg:+.2f}%  (n={len(negative_returns)})")
    else:
        print("Avg 5d return (negative signals): N/A")

    if sharpe is not None:
        print(f"Sentiment signal Sharpe ratio:    {sharpe:.3f}")
    else:
        print("Sentiment signal Sharpe ratio:    N/A")

    print("=" * 60)

    # Note about predicted sentiment
    unavailable_count = sum(1 for r in results if r["predicted_sentiment"] == "unavailable")
    if unavailable_count > 0:
        print(
            f"\nNote: {unavailable_count}/{len(results)} predicted sentiments are 'unavailable'.\n"
            "Start the AlphaSignal API (uvicorn alphasignal.api.app:app --reload) and\n"
            "re-run to get live sentiment predictions."
        )
    print()


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def run_evaluation(
    golden_set_path: Optional[Path] = None,
    api_base_url: str = "http://localhost:8000",
    save_results: bool = True,
) -> list[dict[str, Any]]:
    """Run the AlphaSignal quality evaluation.

    Loads the golden set, fetches actual 5-day returns via yfinance,
    attempts to call the sentiment API, then prints and saves results.

    Args:
        golden_set_path: Path to golden_set.json. Defaults to the bundled file.
        api_base_url: Base URL for the AlphaSignal API.
        save_results: Whether to save results to eval_results_{date}.json.

    Returns:
        List of result dicts, one per golden set entry.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    golden_path = golden_set_path or _GOLDEN_SET_PATH
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden set not found: {golden_path}")

    with open(golden_path) as fh:
        golden_set: list[dict[str, Any]] = json.load(fh)

    if not golden_set:
        raise ValueError("Golden set is empty.")

    print(f"\nLoaded {len(golden_set)} events from {golden_path.name}")
    print("Fetching price data and sentiment predictions...\n")

    results: list[dict[str, Any]] = []

    for i, entry in enumerate(golden_set, start=1):
        ticker: str = entry["ticker"]
        event_date_str: str = entry["date"]
        expected_sentiment: str = entry["expected_sentiment"]
        event_description: str = entry["event_description"]

        event_date = date.fromisoformat(event_date_str)

        print(f"[{i:2d}/{len(golden_set)}] {ticker:5s} {event_date_str}  Fetching returns...", end="", flush=True)

        # Compute actual 5-day forward return
        actual_return = _compute_5d_forward_return(ticker, event_date)

        if actual_return is not None:
            print(f"  return={actual_return:+.2f}%", end="", flush=True)
        else:
            print("  return=N/A", end="", flush=True)

        # Fetch predicted sentiment from API
        predicted_sentiment = _fetch_predicted_sentiment(
            ticker=ticker,
            event_description=event_description,
            base_url=api_base_url,
        )
        print(f"  sentiment={predicted_sentiment}")

        # Determine correctness
        # When predicted is unavailable, fall back to expected_sentiment for correctness check
        sentiment_for_correctness = (
            predicted_sentiment if predicted_sentiment != "unavailable" else expected_sentiment
        )
        correct = _is_correct_prediction(sentiment_for_correctness, actual_return)

        result: dict[str, Any] = {
            "ticker": ticker,
            "date": event_date_str,
            "event_description": event_description,
            "expected_sentiment": expected_sentiment,
            "predicted_sentiment": predicted_sentiment,
            "actual_5d_return_pct": actual_return,
            "correct": correct,
        }
        results.append(result)

    # Print results table
    _print_results_table(results)

    # Print summary
    _print_summary(results)

    # Save results
    if save_results:
        today = date.today().isoformat()
        results_path = _RESULTS_DIR / f"eval_results_{today}.json"
        with open(results_path, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"Results saved to: {results_path}\n")

    return results


if __name__ == "__main__":
    run_evaluation()
