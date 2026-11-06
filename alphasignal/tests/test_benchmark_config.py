"""Benchmark rows must use independent settings on the same stored corpus."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from alphasignal.scripts import benchmark


@pytest.mark.parametrize("reverse", [False, True])
def test_benchmark_weights_are_isolated_and_order_independent(monkeypatch, reverse):
    original = {"retrieval": {"hybrid_weights": {"bm25": 0.4, "dense": 0.6}}}
    config = deepcopy(original)
    observed = []

    def make_retriever(settings, *args):
        observed.append(deepcopy(settings["retrieval"]["hybrid_weights"]))
        return MagicMock()

    monkeypatch.setattr(benchmark, "HybridRetriever", make_retriever)
    monkeypatch.setattr(
        benchmark, "build_storage_components", lambda settings: (None, None, None)
    )
    evaluator = MagicMock()
    evaluator.evaluate.return_value = SimpleNamespace(
        mrr=0.5, ndcg_at_5=0.5, hit_at_3=0.5, num_queries=1
    )
    monkeypatch.setattr(benchmark, "RetrievalEvaluator", lambda path: evaluator)
    monkeypatch.setattr(benchmark, "CrossEncoderReranker", MagicMock())
    specs = list(benchmark.CONFIGS_TO_BENCHMARK)
    if reverse:
        specs.reverse()
    for spec in specs:
        benchmark.run_benchmark_config(spec, config, "unused.json")
        assert config == original

    expected = {
        "dense_only": {"bm25": 0.0, "dense": 1.0},
        "sparse_only": {"bm25": 1.0, "dense": 0.0},
        "hybrid": original["retrieval"]["hybrid_weights"],
    }
    assert observed == [expected[spec["retrieval"]] for spec in specs]
    assert [
        call.kwargs["reranker"] is not None
        for call in evaluator.evaluate.call_args_list
    ] == [spec["reranking"] for spec in specs]
