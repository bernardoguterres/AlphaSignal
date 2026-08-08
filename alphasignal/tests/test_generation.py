"""Tests for RAG generation module."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from alphasignal.generation import GenerationResult
from alphasignal.generation.generator import RAGGenerator
from alphasignal.retrieval import RetrievedChunk


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "generation": {"model": "gpt-4o-mini", "max_tokens": 1000, "temperature": 0.1}
    }


@pytest.fixture
def test_chunks():
    """Create test retrieved chunks."""
    return [
        RetrievedChunk(
            chunk_id="aapl_10k_test_0001",
            ticker="AAPL",
            text="Apple Inc. reported revenue of $394.3 billion for fiscal year 2024, representing a 16% increase year-over-year driven primarily by iPhone sales.",
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_7",
            date=date(2024, 10, 31),
            url=None,
            dense_score=0.95,
            sparse_score=0.87,
            hybrid_score=0.92,
            final_score=0.94,
        ),
        RetrievedChunk(
            chunk_id="aapl_10k_test_0002",
            ticker="AAPL",
            text="iPhone revenue reached $200.6 billion, up 12% from the prior year, benefiting from strong demand for iPhone 15 Pro models.",
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_7",
            date=date(2024, 10, 31),
            url=None,
            dense_score=0.89,
            sparse_score=0.82,
            hybrid_score=0.86,
            final_score=0.88,
        ),
        RetrievedChunk(
            chunk_id="aapl_10k_test_0003",
            ticker="AAPL",
            text="Services revenue grew to $85.2 billion, reflecting continued growth in App Store, Apple Music, and iCloud subscriptions.",
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_7",
            date=date(2024, 10, 31),
            url=None,
            dense_score=0.78,
            sparse_score=0.75,
            hybrid_score=0.77,
            final_score=0.80,
        ),
    ]


def test_rag_generator_builds_prompt_with_context(test_config, test_chunks):
    """Test that build_prompt includes all context chunks."""
    with patch("alphasignal.generation.generator.OpenAI"):
        generator = RAGGenerator(test_config)

        system_msg, user_msg = generator.build_prompt(
            "What is Apple's revenue?", test_chunks
        )

        # System message should contain guidance
        assert "financial research assistant" in system_msg.lower()
        assert "[Source N]" in system_msg or "cite" in system_msg.lower()

        # User message should contain all chunk texts
        assert "394.3 billion" in user_msg
        assert "200.6 billion" in user_msg
        assert "85.2 billion" in user_msg

        # Should reference source numbering
        assert "[Source 1]" in user_msg
        assert "[Source 2]" in user_msg
        assert "[Source 3]" in user_msg

        # Should contain the query
        assert "What is Apple's revenue?" in user_msg


def test_rag_generator_parses_citations(test_config, test_chunks):
    """Test that _parse_citations correctly extracts cited chunks."""
    with patch("alphasignal.generation.generator.OpenAI"):
        generator = RAGGenerator(test_config)

        # Mock answer with citations
        answer = "Revenue grew to $394.3 billion [Source 1] due to iPhone sales [Source 2] and services growth."

        parsed_answer, cited_chunks = generator._parse_citations(answer, test_chunks)

        # Should return original answer
        assert parsed_answer == answer

        # Should have extracted 2 cited chunks
        assert len(cited_chunks) == 2
        assert cited_chunks[0].chunk_id == "aapl_10k_test_0001"  # Source 1
        assert cited_chunks[1].chunk_id == "aapl_10k_test_0002"  # Source 2


def test_rag_generator_handles_no_citations(test_config, test_chunks):
    """Test that generate handles answers with no citations."""
    with patch("alphasignal.generation.generator.OpenAI") as MockOpenAI:
        # Mock OpenAI response with no citations
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "I don't have enough information to answer this question."
        )
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 20

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        generator = RAGGenerator(test_config)
        result = generator.generate("What is the stock price?", test_chunks)

        # Should have result with no citations
        assert isinstance(result, GenerationResult)
        assert len(result.cited_chunks) == 0
        assert (
            result.answer == "I don't have enough information to answer this question."
        )
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 20


def test_rag_generator_handles_multiple_citations_same_source(test_config, test_chunks):
    """Test that duplicate citations to same source only appear once."""
    with patch("alphasignal.generation.generator.OpenAI"):
        generator = RAGGenerator(test_config)

        # Answer citing Source 1 twice
        answer = "Revenue was $394.3B [Source 1] which is significant [Source 1]."

        _, cited_chunks = generator._parse_citations(answer, test_chunks)

        # Should only include chunk once
        assert len(cited_chunks) == 1
        assert cited_chunks[0].chunk_id == "aapl_10k_test_0001"


def test_rag_generator_handles_invalid_source_numbers(test_config, test_chunks):
    """Test that invalid source numbers are ignored."""
    with patch("alphasignal.generation.generator.OpenAI"):
        generator = RAGGenerator(test_config)

        # Answer with invalid source numbers
        answer = "Revenue grew [Source 0] significantly [Source 99]."

        _, cited_chunks = generator._parse_citations(answer, test_chunks)

        # Should not include any chunks (both indices out of range)
        assert len(cited_chunks) == 0


def test_rag_generator_handles_empty_chunks(test_config):
    """Test that generator handles empty chunk list gracefully."""
    with patch("alphasignal.generation.generator.OpenAI"):
        generator = RAGGenerator(test_config)

        result = generator.generate("What is Apple's revenue?", [])

        # Should return default response
        assert isinstance(result, GenerationResult)
        assert "don't have enough information" in result.answer.lower()
        assert len(result.cited_chunks) == 0
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0


def test_rag_generator_handles_api_error(test_config, test_chunks):
    """Test that generator handles OpenAI API errors gracefully."""
    with patch("alphasignal.generation.generator.OpenAI") as MockOpenAI:
        # Mock OpenAI to raise an exception
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        MockOpenAI.return_value = mock_client

        generator = RAGGenerator(test_config)
        result = generator.generate("What is Apple's revenue?", test_chunks)

        # Should return error response without raising
        assert isinstance(result, GenerationResult)
        assert "error occurred" in result.answer.lower()
        assert len(result.cited_chunks) == 0
