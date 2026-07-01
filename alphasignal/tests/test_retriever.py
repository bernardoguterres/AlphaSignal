"""Tests for hybrid retrieval pipeline."""

import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from alphasignal.embeddings.cache import EmbeddingCache
from alphasignal.embeddings.embedder import Embedder
from alphasignal.ingestion import Chunk
from alphasignal.retrieval import RetrievedChunk
from alphasignal.retrieval.evaluator import EvalResults, RetrievalEvaluator
from alphasignal.retrieval.reranker import CrossEncoderReranker
from alphasignal.retrieval.retriever import HybridRetriever
from alphasignal.store.metadata_store import MetadataStore
from alphasignal.store.vector_store import VectorStore


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "embeddings": {
            "model": "text-embedding-ada-002",
            "batch_size": 100,
        },
        "retrieval": {
            "dense_candidates": 10,
            "sparse_candidates": 10,
            "rerank_candidates": 5,
            "hybrid_weights": {"bm25": 0.4, "dense": 0.6},
        },
    }


@pytest.fixture
def test_chunks():
    """Create test chunks with varied content."""
    today = date.today()
    return [
        Chunk(
            chunk_id=f"aapl_10k_test_{i:04d}",
            ticker="AAPL",
            text=f"Apple Inc revenue increased by {i*10}% due to strong iPhone sales and services growth in fiscal year 2024.",
            token_count=50,
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_7",
            date=today - timedelta(days=i),
            url=None,
            chunk_index=i,
            total_chunks=5,
        )
        for i in range(5)
    ] + [
        Chunk(
            chunk_id=f"msft_10k_test_{i:04d}",
            ticker="MSFT",
            text=f"Microsoft Corporation cloud revenue grew by {i*15}% driven by Azure and Office 365 adoption in enterprise segment.",
            token_count=50,
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_7",
            date=today - timedelta(days=i + 10),
            url=None,
            chunk_index=i,
            total_chunks=3,
        )
        for i in range(3)
    ]


@pytest.fixture
def hybrid_retriever(test_config, test_chunks, tmp_path):
    """Create HybridRetriever with test data."""
    # Create stores
    vector_store = VectorStore(str(tmp_path / "index"), dim=1536)
    vector_store.load()

    metadata_store = MetadataStore(str(tmp_path / "test.db"))
    metadata_store.add_chunks(test_chunks)

    # Create embedding cache
    cache = EmbeddingCache(str(tmp_path / "cache.pkl"))

    # Mock embedder
    with patch("alphasignal.embeddings.embedder.OpenAI"):
        embedder = Embedder(test_config, cache)

        # Mock embed_texts to return random embeddings
        def mock_embed_texts(texts):
            return [np.random.rand(1536).astype(np.float32) for _ in texts]

        embedder.embed_texts = mock_embed_texts

    # Generate embeddings for all chunks and add to vector store
    embeddings = np.random.rand(len(test_chunks), 1536).astype(np.float32)
    chunk_ids = [chunk.chunk_id for chunk in test_chunks]
    vector_store.add(embeddings, chunk_ids)

    # Create retriever
    retriever = HybridRetriever(test_config, embedder, vector_store, metadata_store)

    return retriever


def test_hybrid_retriever_returns_results(hybrid_retriever):
    """Test that hybrid retriever returns relevant results."""
    query = "What was Apple's revenue growth?"

    results = hybrid_retriever.retrieve(query, top_k=5)

    # Should return results
    assert len(results) > 0
    assert len(results) <= 5

    # Results should be RetrievedChunk objects
    assert all(isinstance(r, RetrievedChunk) for r in results)

    # Should have scores
    assert all(r.dense_score is not None for r in results)
    assert all(r.sparse_score is not None for r in results)
    assert all(r.hybrid_score is not None for r in results)

    # Should be sorted by hybrid_score
    scores = [r.hybrid_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_ticker_filter_limits_results(hybrid_retriever):
    """Test that ticker filter only returns chunks from specified ticker."""
    query = "revenue growth"

    # Filter for AAPL only
    results = hybrid_retriever.retrieve(query, ticker="AAPL", top_k=10)

    # All results should be from AAPL
    assert all(r.ticker == "AAPL" for r in results)
    assert len(results) > 0


def test_date_filter_limits_results(hybrid_retriever):
    """Test that date filter limits results to date range."""
    query = "revenue growth"
    today = date.today()
    date_from = today - timedelta(days=5)
    date_to = today

    results = hybrid_retriever.retrieve(
        query, date_from=date_from, date_to=date_to, top_k=10
    )

    # All results should be within date range
    assert all(date_from <= r.date <= date_to for r in results)


def test_merge_results_combines_scores_correctly(hybrid_retriever):
    """Test that merge_results correctly combines dense and sparse scores."""
    dense_results = [("chunk_1", 0.9), ("chunk_2", 0.8), ("chunk_3", 0.7)]
    sparse_results = [("chunk_2", 0.9), ("chunk_3", 0.8), ("chunk_4", 0.7)]

    merged = hybrid_retriever._merge_results(dense_results, sparse_results)

    # Should have 4 unique chunks
    assert len(merged) == 4

    # Each entry should have (chunk_id, dense_score, sparse_score, hybrid_score)
    assert all(len(entry) == 4 for entry in merged)

    # chunk_2 should appear with both non-zero scores (it's not min in either list)
    chunk_2_entry = [e for e in merged if e[0] == "chunk_2"][0]
    assert chunk_2_entry[1] >= 0  # Has dense score (normalized)
    assert chunk_2_entry[2] >= 0  # Has sparse score (normalized)
    assert chunk_2_entry[3] > 0  # Has hybrid score

    # chunk_1 should only have dense score
    chunk_1_entry = [e for e in merged if e[0] == "chunk_1"][0]
    assert chunk_1_entry[1] == 1.0  # Max dense score
    assert chunk_1_entry[2] == 0.0  # No sparse score

    # Should be sorted by hybrid score (descending)
    hybrid_scores = [e[3] for e in merged]
    assert hybrid_scores == sorted(hybrid_scores, reverse=True)


def test_bm25_index_builds_successfully(hybrid_retriever):
    """Test that BM25 index builds from metadata store."""
    # Index should build on first sparse search
    assert hybrid_retriever.bm25_index is None

    # Trigger index build
    hybrid_retriever.build_bm25_index()

    # Index should now exist
    assert hybrid_retriever.bm25_index is not None
    assert len(hybrid_retriever.bm25_chunk_ids) > 0
    assert len(hybrid_retriever.bm25_chunks) > 0


def test_reranker_sorts_by_final_score():
    """Test that reranker computes final_score and sorts correctly."""
    # Create mock retrieved chunks
    chunks = [
        RetrievedChunk(
            chunk_id=f"chunk_{i}",
            ticker="AAPL",
            text=f"This is test chunk number {i} about Apple revenue and growth.",
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_7",
            date=date.today(),
            url=None,
            dense_score=0.5,
            sparse_score=0.5,
            hybrid_score=0.5,
            final_score=None,
        )
        for i in range(3)
    ]

    # Mock the cross-encoder model
    with patch("alphasignal.retrieval.reranker.CrossEncoder") as MockEncoder:
        mock_model = MagicMock()
        # Return decreasing scores so reranking changes order
        mock_model.predict.return_value = [0.9, 0.5, 0.7]
        MockEncoder.return_value = mock_model

        reranker = CrossEncoderReranker()
        query = "Apple revenue growth"

        reranked = reranker.rerank(query, chunks, top_k=3)

        # Should have called predict with query-text pairs
        assert mock_model.predict.called

        # All chunks should have final_score set
        assert all(r.final_score is not None for r in reranked)

        # Should be sorted by final_score (descending)
        final_scores = [r.final_score for r in reranked]
        assert final_scores == sorted(final_scores, reverse=True)

        # First result should be chunk_0 (score 0.9)
        assert reranked[0].chunk_id == "chunk_0"


def test_evaluator_mrr_calculation():
    """Test MRR (Mean Reciprocal Rank) calculation."""
    # Test case 1: First result is relevant
    retrieved_ids = ["chunk_1", "chunk_2", "chunk_3"]
    relevant_ids = {"chunk_1"}
    mrr = RetrievalEvaluator.compute_mrr(retrieved_ids, relevant_ids)
    assert mrr == 1.0  # 1/1

    # Test case 2: Second result is relevant
    relevant_ids = {"chunk_2"}
    mrr = RetrievalEvaluator.compute_mrr(retrieved_ids, relevant_ids)
    assert mrr == 0.5  # 1/2

    # Test case 3: Third result is relevant
    relevant_ids = {"chunk_3"}
    mrr = RetrievalEvaluator.compute_mrr(retrieved_ids, relevant_ids)
    assert abs(mrr - 0.333) < 0.01  # 1/3

    # Test case 4: No relevant results
    relevant_ids = {"chunk_999"}
    mrr = RetrievalEvaluator.compute_mrr(retrieved_ids, relevant_ids)
    assert mrr == 0.0


def test_evaluator_ndcg_calculation():
    """Test NDCG (Normalized Discounted Cumulative Gain) calculation."""
    # Test case 1: Perfect ranking (all relevant items first)
    retrieved_ids = ["rel_1", "rel_2", "irrel_1", "irrel_2"]
    relevant_ids = {"rel_1", "rel_2"}
    ndcg = RetrievalEvaluator.compute_ndcg(retrieved_ids, relevant_ids)
    assert ndcg == 1.0  # Perfect score

    # Test case 2: Worst ranking (relevant items last)
    retrieved_ids = ["irrel_1", "irrel_2", "rel_1", "rel_2"]
    ndcg = RetrievalEvaluator.compute_ndcg(retrieved_ids, relevant_ids)
    assert 0.0 < ndcg < 1.0  # Not perfect but > 0

    # Test case 3: No relevant items
    retrieved_ids = ["irrel_1", "irrel_2"]
    relevant_ids = {"rel_1", "rel_2"}
    ndcg = RetrievalEvaluator.compute_ndcg(retrieved_ids, relevant_ids)
    assert ndcg == 0.0


def test_evaluator_hit_at_k():
    """Test Hit@k metric calculation."""
    # Test case 1: Hit at k=1
    retrieved_ids = ["rel_1"]
    relevant_ids = {"rel_1", "rel_2"}
    hit = RetrievalEvaluator.compute_hit_at_k(retrieved_ids, relevant_ids)
    assert hit == 1.0

    # Test case 2: Miss at k=1
    retrieved_ids = ["irrel_1"]
    hit = RetrievalEvaluator.compute_hit_at_k(retrieved_ids, relevant_ids)
    assert hit == 0.0

    # Test case 3: Hit at k=3 (relevant in position 2)
    retrieved_ids = ["irrel_1", "rel_1", "irrel_2"]
    hit = RetrievalEvaluator.compute_hit_at_k(retrieved_ids, relevant_ids)
    assert hit == 1.0


def test_evaluator_with_golden_set(tmp_path, hybrid_retriever):
    """Test evaluator with a small golden set."""
    # Create golden set file
    golden_set = [
        {
            "query": "Apple revenue growth",
            "relevant_chunk_ids": ["aapl_10k_test_0000", "aapl_10k_test_0001"],
            "ticker": "AAPL",
        },
        {
            "query": "Microsoft cloud performance",
            "relevant_chunk_ids": ["msft_10k_test_0000"],
            "ticker": "MSFT",
        },
    ]

    golden_path = tmp_path / "golden.json"
    import json

    with open(golden_path, "w") as f:
        json.dump(golden_set, f)

    # Create evaluator
    evaluator = RetrievalEvaluator(str(golden_path))

    # Run evaluation
    results = evaluator.evaluate(hybrid_retriever, top_k=5)

    # Should have results
    assert isinstance(results, EvalResults)
    assert results.num_queries == 2

    # Metrics should be in valid ranges
    assert 0.0 <= results.mrr <= 1.0
    assert 0.0 <= results.ndcg_at_5 <= 1.0
    assert 0.0 <= results.ndcg_at_10 <= 1.0
    assert 0.0 <= results.hit_at_1 <= 1.0
    assert 0.0 <= results.hit_at_5 <= 1.0
    assert 0.0 <= results.hit_at_10 <= 1.0
