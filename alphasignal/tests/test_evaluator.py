"""Tests for retrieval evaluation."""

import json
from pathlib import Path

import pytest

from alphasignal.retrieval.evaluator import RetrievalEvaluator


@pytest.fixture
def golden_set_path():
    """Get path to golden set."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / "evaluation" / "golden_set.json"


def test_evaluator_loads_golden_set(golden_set_path):
    """Test that evaluator loads golden set correctly."""
    if not golden_set_path.exists():
        pytest.skip("Golden set not yet created")

    # Load golden set
    with open(golden_set_path) as f:
        golden_set = json.load(f)

    # Should have at least 50 entries
    assert len(golden_set) >= 50, f"Expected >=50 entries, got {len(golden_set)}"

    # Check required fields
    for entry in golden_set:
        assert "id" in entry, "Missing 'id' field"
        assert "question" in entry, "Missing 'question' field"
        assert "ticker" in entry, "Missing 'ticker' field"
        assert "relevant_chunk_ids" in entry, "Missing 'relevant_chunk_ids' field"
        assert "answer_summary" in entry, "Missing 'answer_summary' field"
        assert "question_type" in entry, "Missing 'question_type' field"


def test_evaluator_mrr_perfect_score():
    """Test MRR computation with perfect score."""
    ranked_ids = ["a", "b", "c"]
    relevant_ids = {"a"}

    mrr = RetrievalEvaluator.compute_mrr(ranked_ids, relevant_ids)

    # First result is relevant, so MRR = 1/1 = 1.0
    assert mrr == 1.0


def test_evaluator_mrr_second_place():
    """Test MRR computation with relevant doc in second place."""
    ranked_ids = ["a", "b", "c"]
    relevant_ids = {"b"}

    mrr = RetrievalEvaluator.compute_mrr(ranked_ids, relevant_ids)

    # Second result is relevant, so MRR = 1/2 = 0.5
    assert mrr == 0.5


def test_evaluator_ndcg_at_5_with_two_relevant():
    """Test NDCG@5 with two relevant documents at different positions."""
    # Case 1: Relevant docs at positions 1 and 3
    ranked_ids_1 = ["rel_1", "irrel_1", "rel_2", "irrel_2", "irrel_3"]
    relevant_ids = {"rel_1", "rel_2"}

    ndcg_1 = RetrievalEvaluator.compute_ndcg(ranked_ids_1, relevant_ids)

    # Case 2: Relevant docs at positions 2 and 5 (worse ranking)
    ranked_ids_2 = ["irrel_1", "rel_1", "irrel_2", "irrel_3", "rel_2"]

    ndcg_2 = RetrievalEvaluator.compute_ndcg(ranked_ids_2, relevant_ids)

    # Better ranking should have higher NDCG
    assert ndcg_1 > ndcg_2
    assert 0.0 < ndcg_1 <= 1.0
    assert 0.0 < ndcg_2 <= 1.0


def test_evaluator_hit_at_3_true():
    """Test Hit@3 when relevant doc is within top 3."""
    ranked_ids = ["irrel_1", "rel_1", "irrel_2", "irrel_3", "irrel_4"]
    relevant_ids = {"rel_1"}

    # Relevant doc at position 2 (within top 3)
    hit = RetrievalEvaluator.compute_hit_at_k(ranked_ids[:3], relevant_ids)

    assert hit == 1.0


def test_evaluator_hit_at_3_false():
    """Test Hit@3 when relevant doc is outside top 3."""
    ranked_ids = ["irrel_1", "irrel_2", "irrel_3", "irrel_4", "rel_1"]
    relevant_ids = {"rel_1"}

    # Relevant doc at position 5 (outside top 3)
    hit = RetrievalEvaluator.compute_hit_at_k(ranked_ids[:3], relevant_ids)

    assert hit == 0.0


def test_golden_set_has_variety_of_question_types(golden_set_path):
    """Test that golden set has variety of question types."""
    if not golden_set_path.exists():
        pytest.skip("Golden set not yet created")

    with open(golden_set_path) as f:
        golden_set = json.load(f)

    # Collect all question types
    question_types = set(entry["question_type"] for entry in golden_set)

    # Should have at least 3 distinct question types
    assert len(question_types) >= 3, (
        f"Expected at least 3 question types, got {len(question_types)}: {question_types}"
    )

    # Check for expected types
    expected_types = {"factual", "trend", "comparative", "sentiment", "specific"}
    assert question_types.issubset(expected_types), (
        f"Unexpected question types: {question_types - expected_types}"
    )
