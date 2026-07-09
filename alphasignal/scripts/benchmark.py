"""Benchmark script for evaluating retrieval configurations."""

import json
import logging
import sys
import time
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


# Benchmark configurations
#
# NOTE: "chunking" is not actually varied between rows. Only SemanticChunker
# is implemented, and these benchmarks evaluate retrieval against the corpus
# as it was already ingested on disk - they don't re-ingest with a different
# chunking strategy per row. The "chunking" field is kept for future use once
# a second chunker (e.g. naive fixed-window) exists, but a warning is logged
# below so results aren't misread as a real chunking comparison.
CONFIGS_TO_BENCHMARK = [
    {
        "name": "Baseline: naive chunks + dense only",
        "chunking": "naive_512",
        "retrieval": "dense_only",
        "reranking": False,
    },
    {
        "name": "Semantic chunks + dense only",
        "chunking": "semantic",
        "retrieval": "dense_only",
        "reranking": False,
    },
    {
        "name": "Semantic chunks + hybrid",
        "chunking": "semantic",
        "retrieval": "hybrid",
        "reranking": False,
    },
    {
        "name": "Semantic chunks + hybrid + reranker",
        "chunking": "semantic",
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

    logger.warning(
        f"chunking={config_spec['chunking']!r} is not actually applied - only "
        "SemanticChunker is implemented, so this row evaluates the corpus as "
        "already ingested on disk, not a re-chunked variant."
    )

    # Initialize components
    vector_store, metadata_store, embedder = build_storage_components(base_config)

    # Modify config based on benchmark spec
    modified_config = base_config.copy()

    # Adjust retrieval weights based on retrieval mode
    if config_spec["retrieval"] == "dense_only":
        # Disable BM25 by setting weight to 0
        if "retrieval" not in modified_config:
            modified_config["retrieval"] = {}
        if "hybrid_weights" not in modified_config["retrieval"]:
            modified_config["retrieval"]["hybrid_weights"] = {}

        modified_config["retrieval"]["hybrid_weights"]["bm25"] = 0.0
        modified_config["retrieval"]["hybrid_weights"]["dense"] = 1.0
        logger.info("Using dense-only retrieval (BM25 weight = 0)")

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


def main():
    """Run benchmarks across all configurations."""
    print("=" * 80)
    print("AlphaSignal Retrieval Benchmark")
    print("=" * 80)
    print()

    # Load configuration
    config = load_config(project_root)

    # Check for golden set
    golden_set_path = project_root / "evaluation" / "golden_set.json"
    if not golden_set_path.exists():
        print(f"ERROR: Golden set not found at {golden_set_path}")
        print("Please create the golden set first.")
        return

    with open(golden_set_path) as f:
        golden_set = json.load(f)

    print(f"Loaded golden set with {len(golden_set)} questions")

    # Check annotations
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
