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
    """Test that invalid source numbers are ignored - and, since the
    2026-08-15 citation-integrity fix, actually removed from the returned
    answer text too (not just dropped from cited_chunks while the dangling
    marker stays in the text - the original defect)."""
    with patch("alphasignal.generation.generator.OpenAI"):
        generator = RAGGenerator(test_config)

        # Answer with invalid source numbers
        answer = "Revenue grew [Source 0] significantly [Source 99]."

        cleaned_answer, cited_chunks = generator._parse_citations(answer, test_chunks)

        # Should not include any chunks (both indices out of range)
        assert len(cited_chunks) == 0
        # The dangling, unresolvable markers must not survive into the
        # returned answer text - every [Source N] in the text must resolve.
        assert "[Source 0]" not in cleaned_answer
        assert "[Source 99]" not in cleaned_answer


def test_rag_generator_single_citation(test_config, test_chunks):
    """One cited source: the simplest case."""
    with patch("alphasignal.generation.generator.OpenAI"):
        generator = RAGGenerator(test_config)
        answer = "Revenue grew to $394.3 billion [Source 1]."

        cleaned_answer, cited_chunks = generator._parse_citations(answer, test_chunks)

        assert cleaned_answer == answer
        assert len(cited_chunks) == 1
        assert cited_chunks[0].chunk_id == test_chunks[0].chunk_id


def test_rag_generator_several_citations_out_of_order_are_renumbered(
    test_config, test_chunks
):
    """Several cited sources, referenced OUT OF ORDER in the text (e.g. the
    model cites Source 3 before Source 1) - must be renumbered so
    cited_chunks[i] always corresponds to the text's rewritten [Source i+1],
    a real invariant rather than incidentally true only when the model
    happens to cite sources in ascending order."""
    with patch("alphasignal.generation.generator.OpenAI"):
        generator = RAGGenerator(test_config)
        answer = "Litigation risk is noted [Source 3], driven by growth [Source 1]."

        cleaned_answer, cited_chunks = generator._parse_citations(answer, test_chunks)

        assert len(cited_chunks) == 2
        # First-appearance order in the (renumbered) text: Source 3 -> new 1, Source 1 -> new 2.
        assert cited_chunks[0].chunk_id == test_chunks[2].chunk_id
        assert cited_chunks[1].chunk_id == test_chunks[0].chunk_id
        assert "[Source 1]" in cleaned_answer
        assert "[Source 2]" in cleaned_answer
        assert "[Source 3]" not in cleaned_answer  # renumbered away, not left stale

        # The core invariant: every [Source N] in the cleaned text resolves.
        import re

        for n in re.findall(r"\[Source (\d+)\]", cleaned_answer):
            assert int(n) <= len(cited_chunks)


def test_rag_generator_mixed_valid_and_out_of_range_citations(test_config, test_chunks):
    """A generated reference beyond the available source count, mixed with
    valid ones: the reproduction of the original bug report (text like
    "[Source 1] ... [Source 5]" with fewer than 5 real sources) - the
    out-of-range marker must be removed, valid ones kept and renumbered."""
    with patch("alphasignal.generation.generator.OpenAI"):
        generator = RAGGenerator(test_config)
        answer = (
            "Revenue grew [Source 1], driven by services [Source 2], "
            "per the analyst note [Source 5]."
        )

        cleaned_answer, cited_chunks = generator._parse_citations(answer, test_chunks)

        assert len(cited_chunks) == 2
        assert cited_chunks[0].chunk_id == test_chunks[0].chunk_id
        assert cited_chunks[1].chunk_id == test_chunks[1].chunk_id
        assert "[Source 5]" not in cleaned_answer
        assert "[Source 1]" in cleaned_answer and "[Source 2]" in cleaned_answer


def test_rag_generator_malformed_source_marker_left_untouched(test_config, test_chunks):
    """A malformed marker (non-numeric) doesn't match the citation regex at
    all - it's plain prose to this parser, left alone rather than crashing
    or being misinterpreted."""
    with patch("alphasignal.generation.generator.OpenAI"):
        generator = RAGGenerator(test_config)
        answer = "Revenue grew [Source A] according to the filing [Source 1]."

        cleaned_answer, cited_chunks = generator._parse_citations(answer, test_chunks)

        assert "[Source A]" in cleaned_answer  # untouched, not a valid pattern match
        assert len(cited_chunks) == 1
        assert cited_chunks[0].chunk_id == test_chunks[0].chunk_id


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
