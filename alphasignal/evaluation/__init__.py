"""AlphaSignal evaluation framework.

Tests whether AlphaSignal sentiment predictions correlate with
actual 5-day forward returns (the core quality question).
"""

from alphasignal.evaluation.run_eval import run_evaluation

__all__ = ["run_evaluation"]
