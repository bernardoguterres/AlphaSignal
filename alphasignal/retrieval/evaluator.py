"""Retrieval evaluation module."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalResults:
    """Results from retrieval evaluation."""

    mrr: float  # Mean Reciprocal Rank
    ndcg_at_5: float  # Normalized Discounted Cumulative Gain @ 5
    ndcg_at_10: float  # NDCG @ 10
    hit_at_1: float  # Hit rate @ 1
    hit_at_5: float  # Hit rate @ 5
    hit_at_10: float  # Hit rate @ 10
    num_queries: int  # Number of queries evaluated


class RetrievalEvaluator:
    """Evaluates retrieval quality using a golden dataset."""

    def __init__(self, golden_set_path: str):
        """Initialize evaluator with golden set.

        Args:
            golden_set_path: Path to JSON file with golden query-chunk pairs

        Golden set format:
        [
            {
                "query": "What were Apple's revenue drivers in Q4?",
                "relevant_chunk_ids": ["aapl_10k_abc_0001", "aapl_10q_xyz_0005"],
                "ticker": "AAPL"
            },
            ...
        ]
        """
        self.golden_set_path = Path(golden_set_path)
        self.golden_set = self.load_golden_set()

    def load_golden_set(self) -> list[dict]:
        """Load golden set from JSON file.

        Returns:
            List of golden query-chunk pairs
        """
        if not self.golden_set_path.exists():
            logger.warning(f"Golden set not found at {self.golden_set_path}")
            return []

        with open(self.golden_set_path) as f:
            golden_set = json.load(f)

        logger.info(
            f"Loaded {len(golden_set)} golden queries from {self.golden_set_path}"
        )
        return golden_set

    def evaluate(self, retriever, top_k: int = 10, reranker=None) -> EvalResults:
        """Evaluate retrieval performance on golden set.

        Args:
            retriever: HybridRetriever instance
            top_k: Number of results to retrieve per query
            reranker: Optional CrossEncoderReranker. When provided, retrieved
                chunks are reranked before metrics are computed against them.

        Returns:
            EvalResults with computed metrics
        """
        if not self.golden_set:
            logger.warning("No golden set available for evaluation")
            return EvalResults(
                mrr=0.0,
                ndcg_at_5=0.0,
                ndcg_at_10=0.0,
                hit_at_1=0.0,
                hit_at_5=0.0,
                hit_at_10=0.0,
                num_queries=0,
            )

        logger.info(f"Evaluating retrieval on {len(self.golden_set)} queries")

        # Track results for each query
        reciprocal_ranks = []
        ndcg_5_scores = []
        ndcg_10_scores = []
        hit_1_scores = []
        hit_5_scores = []
        hit_10_scores = []

        for item in self.golden_set:
            query = item["query"]
            relevant_chunk_ids = set(item["relevant_chunk_ids"])
            ticker = item.get("ticker")

            # Retrieve chunks
            retrieved_chunks = retriever.retrieve(
                query=query, ticker=ticker, top_k=top_k
            )

            # Rerank if a reranker was provided
            if reranker is not None:
                retrieved_chunks = reranker.rerank(query, retrieved_chunks, top_k=top_k)

            # Get retrieved chunk IDs in order
            retrieved_ids = [chunk.chunk_id for chunk in retrieved_chunks]

            # Compute metrics for this query
            reciprocal_ranks.append(self.compute_mrr(retrieved_ids, relevant_chunk_ids))
            ndcg_5_scores.append(
                self.compute_ndcg(retrieved_ids[:5], relevant_chunk_ids)
            )
            ndcg_10_scores.append(
                self.compute_ndcg(retrieved_ids[:10], relevant_chunk_ids)
            )
            hit_1_scores.append(
                self.compute_hit_at_k(retrieved_ids[:1], relevant_chunk_ids)
            )
            hit_5_scores.append(
                self.compute_hit_at_k(retrieved_ids[:5], relevant_chunk_ids)
            )
            hit_10_scores.append(
                self.compute_hit_at_k(retrieved_ids[:10], relevant_chunk_ids)
            )

        # Compute mean metrics
        results = EvalResults(
            mrr=float(np.mean(reciprocal_ranks)),
            ndcg_at_5=float(np.mean(ndcg_5_scores)),
            ndcg_at_10=float(np.mean(ndcg_10_scores)),
            hit_at_1=float(np.mean(hit_1_scores)),
            hit_at_5=float(np.mean(hit_5_scores)),
            hit_at_10=float(np.mean(hit_10_scores)),
            num_queries=len(self.golden_set),
        )

        logger.info(
            f"Evaluation complete: MRR={results.mrr:.3f}, NDCG@5={results.ndcg_at_5:.3f}"
        )
        return results

    @staticmethod
    def compute_mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
        """Compute Mean Reciprocal Rank for a single query.

        Args:
            retrieved_ids: List of retrieved chunk IDs in order
            relevant_ids: Set of relevant chunk IDs

        Returns:
            Reciprocal rank (1/rank of first relevant item, or 0 if none found)
        """
        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def compute_ndcg(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
        """Compute Normalized Discounted Cumulative Gain.

        Args:
            retrieved_ids: List of retrieved chunk IDs in order
            relevant_ids: Set of relevant chunk IDs

        Returns:
            NDCG score in [0, 1]
        """
        if not relevant_ids or not retrieved_ids:
            return 0.0

        # Compute DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            relevance = 1.0 if chunk_id in relevant_ids else 0.0
            dcg += relevance / np.log2(rank + 1)

        # Compute IDCG (Ideal DCG)
        # Best case: all relevant items ranked first
        ideal_ranks = min(len(retrieved_ids), len(relevant_ids))
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_ranks + 1))

        # Normalize
        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def compute_hit_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
        """Compute Hit@k metric.

        Args:
            retrieved_ids: List of retrieved chunk IDs (top k)
            relevant_ids: Set of relevant chunk IDs

        Returns:
            1.0 if any relevant item is in top k, else 0.0
        """
        for chunk_id in retrieved_ids:
            if chunk_id in relevant_ids:
                return 1.0
        return 0.0
