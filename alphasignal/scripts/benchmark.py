"""Benchmark script for evaluating retrieval configurations."""

import json
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from alphasignal.retrieval.evaluator import RetrievalEvaluator
from alphasignal.retrieval.reranker import CrossEncoderReranker
from alphasignal.retrieval.retriever import HybridRetriever
from alphasignal.scripts._common import build_storage_components, load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# All rows evaluate the same already-ingested corpus. Only retrieval weights
# and reranking vary; no row claims to benchmark a different chunker.
CONFIGS_TO_BENCHMARK = [
    {
        "name": "Dense only",
        "retrieval": "dense_only",
        "reranking": False,
    },
    {
        "name": "Sparse only",
        "retrieval": "sparse_only",
        "reranking": False,
    },
    {
        "name": "Hybrid",
        "retrieval": "hybrid",
        "reranking": False,
    },
    {
        "name": "Hybrid + reranker",
        "retrieval": "hybrid",
        "reranking": True,
    },
]


def run_benchmark_config(config_spec, base_config, golden_set_path):
    """Run benchmark for a specific configuration.

    Args:
        config_spec: Configuration specification dict
        base_config: Base system configuration
        golden_set_path: Path to golden set JSON

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'=' * 80}")
    print(f"Benchmarking: {config_spec['name']}")
    print(f"{'=' * 80}")

    logger.info("Evaluating the existing corpus without changing chunking")

    # Isolate nested settings before passing them to any component.
    modified_config = deepcopy(base_config)

    # Adjust retrieval weights based on retrieval mode
    mode = config_spec["retrieval"]
    if mode in {"dense_only", "sparse_only"}:
        retrieval = modified_config.setdefault("retrieval", {})
        retrieval["hybrid_weights"] = {
            "bm25": 1.0 if mode == "sparse_only" else 0.0,
            "dense": 1.0 if mode == "dense_only" else 0.0,
        }
    elif mode != "hybrid":
        raise ValueError(f"Unknown retrieval mode: {mode}")

    vector_store, metadata_store, embedder = build_storage_components(modified_config)

    # Initialize retriever
    retriever = HybridRetriever(modified_config, embedder, vector_store, metadata_store)
    retriever.build_bm25_index()

    # Initialize evaluator
    evaluator = RetrievalEvaluator(str(golden_set_path))

    # Initialize reranker if this config calls for it
    reranker = CrossEncoderReranker() if config_spec["reranking"] else None

    # Run evaluation
    logger.info("Running evaluation on golden set...")
    start_time = time.time()

    eval_results = evaluator.evaluate(retriever, top_k=10, reranker=reranker)

    elapsed_ms = int((time.time() - start_time) * 1000)
    avg_latency_ms = (
        elapsed_ms // eval_results.num_queries if eval_results.num_queries > 0 else 0
    )

    logger.info(f"Evaluation complete in {elapsed_ms}ms")

    return {
        "config_name": config_spec["name"],
        "mrr_at_10": round(eval_results.mrr, 3),
        "ndcg_at_5": round(eval_results.ndcg_at_5, 3),
        "hit_at_3": round(eval_results.hit_at_3, 3),
        "avg_latency_ms": avg_latency_ms,
        "num_queries": eval_results.num_queries,
    }


def load_and_validate_retrieval_golden_set(golden_set_path: Path) -> list[dict]:
    """Load the retrieval golden set and validate it's actually usable.

    Deliberately a plain file-in/list-out function (no printing, no config
    loading) so it's unit-testable in isolation - see
    tests/test_benchmark_golden_set.py. Distinct from
    alphasignal/evaluation/sentiment_golden_set.json (the sentiment quality
    eval's own dataset, read only by evaluation/run_eval.py) - files
    renamed 2026-08-15 (FINAL_ENGINEERING_AUDIT.md remediation item 4) so
    the two unrelated evaluation concepts can't be confused for one shared
    file again, even though the two scripts already resolved to different
    paths (project_root vs run_eval.py's own directory).

    Raises:
        FileNotFoundError: golden_set_path doesn't exist.
        ValueError: the file exists but is malformed (an entry missing
            'relevant_chunk_ids') or unannotated (every entry's
            relevant_chunk_ids is empty, so no metric could ever be
            anything but a meaningless 0.0 - reporting that as a real
            benchmark result would be exactly the kind of fabricated
            finding this project must not produce).
    """
    if not golden_set_path.exists():
        raise FileNotFoundError(
            "Retrieval benchmark cannot run: no annotated retrieval golden "
            f"set is available (expected at {golden_set_path})."
        )

    with open(golden_set_path) as f:
        golden_set = json.load(f)

    for entry in golden_set:
        if "relevant_chunk_ids" not in entry:
            raise ValueError(
                "Retrieval benchmark cannot run: golden set entry "
                f"{entry.get('id', entry)!r} is missing 'relevant_chunk_ids' "
                "- malformed retrieval golden set schema."
            )

    annotated = [q for q in golden_set if q.get("relevant_chunk_ids")]
    if not annotated:
        raise ValueError(
            "Retrieval benchmark cannot run: no annotated retrieval golden "
            f"set is available - all {len(golden_set)} questions in "
            f"{golden_set_path.name} have empty relevant_chunk_ids. Run "
            "annotate_golden_set.py first; until then, any MRR/NDCG/Hit@k "
            "numbers from this script would be meaningless zeros, not a "
            "real measurement of retrieval quality."
        )

    return golden_set


def main():
    """Run benchmarks across all configurations."""
    print("=" * 80)
    print("AlphaSignal Retrieval Benchmark")
    print("=" * 80)
    print()

    # Load configuration
    config = load_config(project_root)

    golden_set_path = project_root / "evaluation" / "retrieval_golden_set.json"
    try:
        golden_set = load_and_validate_retrieval_golden_set(golden_set_path)
    except (FileNotFoundError, ValueError) as e:
        print(str(e))
        return

    print(f"Loaded golden set with {len(golden_set)} questions")

    annotated = [q for q in golden_set if q.get("relevant_chunk_ids")]
    if len(annotated) < len(golden_set):
        print(
            f"WARNING: Only {len(annotated)}/{len(golden_set)} questions are annotated!"
        )
        print("Run annotate_golden_set.py first for accurate results.")
        print()

    # Run benchmarks
    results = []

    for config_spec in CONFIGS_TO_BENCHMARK:
        try:
            result = run_benchmark_config(config_spec, config, golden_set_path)
            results.append(result)
        except Exception as e:
            logger.error(
                f"Error benchmarking {config_spec['name']}: {e}", exc_info=True
            )
            results.append(
                {
                    "config_name": config_spec["name"],
                    "mrr_at_10": 0.0,
                    "ndcg_at_5": 0.0,
                    "hit_at_3": 0.0,
                    "avg_latency_ms": 0,
                    "num_queries": 0,
                    "error": str(e),
                }
            )

    # Print results table
    print("\n\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print()
    print(
        f"{'Config':<45} | {'MRR@10':>7} | {'NDCG@5':>7} | {'Hit@3':>6} | {'Avg Latency':>12}"
    )
    print("-" * 80)

    for result in results:
        print(
            f"{result['config_name']:<45} | "
            f"{result['mrr_at_10']:>7.3f} | "
            f"{result['ndcg_at_5']:>7.3f} | "
            f"{result['hit_at_3']:>6.3f} | "
            f"{result['avg_latency_ms']:>10}ms"
        )

    print()

    # Save results
    results_path = project_root / "data" / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_questions": len(golden_set),
                "num_annotated": len(annotated),
                "results": results,
            },
            f,
            indent=2,
        )

    logger.info(f"Saved benchmark results to {results_path}")

    # Check if best config meets threshold
    best_result = max(results, key=lambda x: x["mrr_at_10"])
    print(f"\nBest configuration: {best_result['config_name']}")
    print(f"  MRR@10: {best_result['mrr_at_10']:.3f}")

    if best_result["mrr_at_10"] < 0.5:
        print()
        print("WARNING: Best MRR@10 < 0.5 threshold!")
        print("Consider:")
        print("  - Reviewing golden set annotations")
        print("  - Improving chunking strategy")
        print("  - Tuning retrieval weights")
        print("  - Adding more diverse training data")

    print()


if __name__ == "__main__":
    main()
