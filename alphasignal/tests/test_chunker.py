"""Tests for document chunking."""

from datetime import date

import pytest

from alphasignal.ingestion import RawArticle, RawDocument
from alphasignal.ingestion.chunker import SemanticChunker


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "chunking": {
            "target_tokens": 300,
            "min_tokens": 100,
            "max_tokens": 400,
            "overlap_tokens": 50,
        }
    }


@pytest.fixture
def chunker(test_config):
    """Create a SemanticChunker instance."""
    return SemanticChunker(test_config)


def test_chunk_text_respects_max_tokens(chunker):
    """Test that all chunks have token_count <= max_tokens."""
    # Create text of approximately 1200 tokens (about 4800 characters)
    # Each sentence is about 30 tokens
    sentences = [
        f"This is sentence number {i} which contains information about financial markets "
        f"and various aspects of corporate performance including revenue, earnings, and growth metrics. "
        for i in range(40)  # 40 sentences * ~30 tokens = ~1200 tokens
    ]
    text = " ".join(sentences)

    # Chunk the text
    chunks = chunker.chunk_text(text)

    # Verify all chunks respect max_tokens
    assert len(chunks) > 0, "Should produce at least one chunk"
    for chunk_text in chunks:
        token_count = chunker.count_tokens(chunk_text)
        assert token_count <= chunker.max_tokens, \
            f"Chunk has {token_count} tokens, exceeds max {chunker.max_tokens}"


def test_chunk_text_respects_min_tokens(chunker):
    """Test that all chunks except possibly the last have token_count >= min_tokens."""
    # Create text of approximately 1000 tokens
    sentences = [
        f"Sentence {i} discusses important business developments and strategic initiatives "
        f"that impact shareholder value and market positioning. "
        for i in range(35)  # ~1000 tokens
    ]
    text = " ".join(sentences)

    # Chunk the text
    chunks = chunker.chunk_text(text)

    # Verify min_tokens constraint
    assert len(chunks) > 0
    for i, chunk_text in enumerate(chunks):
        token_count = chunker.count_tokens(chunk_text)
        # All chunks except the last should meet min_tokens
        if i < len(chunks) - 1:
            assert token_count >= chunker.min_tokens, \
                f"Non-final chunk {i} has {token_count} tokens, below min {chunker.min_tokens}"


def test_chunk_text_overlap_preserves_context(chunker):
    """Test that consecutive chunks share content from overlap."""
    # Create text with 10 clear, distinct sentences (make them longer to exceed token limit)
    sentences = [
        "The first sentence discusses market conditions and macroeconomic factors that influence corporate performance.",
        "The second sentence analyzes revenue growth trends across multiple business segments and geographic regions.",
        "The third sentence examines profit margin expansion opportunities through operational improvements.",
        "The fourth sentence evaluates operational efficiency gains from technology investments and process optimization.",
        "The fifth sentence explores competitive positioning strategies in rapidly evolving market landscapes.",
        "The sixth sentence reviews customer acquisition metrics and retention rates across different channels.",
        "The seventh sentence assesses product development initiatives and innovation pipeline progress.",
        "The eighth sentence considers regulatory compliance requirements and their impact on business operations.",
        "The ninth sentence investigates supply chain optimization efforts to reduce costs and improve reliability.",
        "The tenth sentence concludes with forward-looking strategic guidance for stakeholders and investors."
    ]
    text = " ".join(sentences)

    # Use smaller max_tokens to force multiple chunks
    chunker.max_tokens = 50
    chunker.overlap_tokens = 20

    chunks = chunker.chunk_text(text)

    # Should have multiple chunks
    assert len(chunks) >= 2, "Should produce multiple chunks for this text"

    # Check that consecutive chunks share some content
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]

        # Find common words between chunks
        current_words = set(current_chunk.split())
        next_words = set(next_chunk.split())
        common_words = current_words & next_words

        # Should have some overlap
        assert len(common_words) > 0, \
            f"Chunks {i} and {i+1} should share some content from overlap"


def test_chunk_text_does_not_split_sentences(chunker):
    """Test that chunks don't split in the middle of sentences."""
    # Create text with sentences of known length
    sentences = [
        f"This is a carefully constructed sentence number {i} that contains exactly thirty tokens "
        f"including numbers and punctuation to ensure consistent length for testing purposes always. "
        for i in range(10)
    ]
    text = " ".join(sentences)

    chunks = chunker.chunk_text(text)

    # Each chunk should start and end at sentence boundaries
    for chunk_text in chunks:
        # Should start with capital letter or digit (sentence start)
        assert chunk_text[0].isupper() or chunk_text[0].isdigit(), \
            "Chunk should start at sentence boundary"

        # Should end with sentence-ending punctuation
        assert chunk_text.rstrip()[-1] in ['.', '!', '?'], \
            "Chunk should end at sentence boundary"


def test_split_sentences_handles_abbreviations(chunker):
    """Test that split_sentences doesn't split on abbreviations."""
    text = "The U.S. economy grew 3.5% in Q3. Analysts were surprised."

    sentences = chunker.split_into_sentences(text)

    # Should split into exactly 2 sentences, not more
    assert len(sentences) == 2, f"Expected 2 sentences, got {len(sentences)}: {sentences}"
    assert "U.S." in sentences[0], "First sentence should contain 'U.S.'"
    assert "Analysts" in sentences[1], "Second sentence should start with 'Analysts'"


def test_chunk_document_assigns_correct_metadata(chunker):
    """Test that chunk_document assigns correct metadata to chunks."""
    # Create a test RawDocument with two sections
    doc = RawDocument(
        ticker="AAPL",
        doc_type="10-K",
        filing_date=date(2024, 1, 15),
        period_of_report=date(2024, 1, 15),
        source="SEC EDGAR",
        sections={
            "item_1": "Apple Inc. designs and manufactures consumer electronics. " * 20,
            "item_7": "Management discusses financial performance and outlook. " * 20,
        },
        file_path="/path/to/filing",
        accession_number="0001234567-24-000001"
    )

    chunks = chunker.chunk_document(doc)

    # Should produce multiple chunks
    assert len(chunks) > 0

    # Verify metadata
    for chunk in chunks:
        assert chunk.ticker == "AAPL"
        assert chunk.doc_type == "10-K"
        assert chunk.source == "SEC EDGAR"
        assert chunk.date == date(2024, 1, 15)
        assert chunk.section in ["item_1", "item_7"]

    # Verify chunk_ids are unique
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "All chunk_ids should be unique"

    # Verify chunk_index is sequential
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i, f"Chunk {i} has incorrect index {chunk.chunk_index}"

    # Verify total_chunks is consistent
    total = len(chunks)
    for chunk in chunks:
        assert chunk.total_chunks == total


def test_chunk_document_deterministic_ids(chunker):
    """Test that chunking the same document produces identical chunk_ids."""
    # Create a test RawDocument
    doc = RawDocument(
        ticker="MSFT",
        doc_type="10-Q",
        filing_date=date(2024, 6, 30),
        period_of_report=date(2024, 6, 30),
        source="SEC EDGAR",
        sections={
            "item_1": "Microsoft Corporation provides software and cloud services. " * 15,
        },
        file_path="/path/to/filing",
        accession_number="0001234567-24-000002"
    )

    # Chunk twice
    chunks1 = chunker.chunk_document(doc)
    chunks2 = chunker.chunk_document(doc)

    # Should produce same number of chunks
    assert len(chunks1) == len(chunks2)

    # chunk_ids should be identical
    ids1 = [chunk.chunk_id for chunk in chunks1]
    ids2 = [chunk.chunk_id for chunk in chunks2]
    assert ids1 == ids2, "chunk_ids should be deterministic"


def test_chunk_article_single_chunk_for_short_article(chunker):
    """Test that short articles are returned as a single chunk."""
    # Create a short article (~200 tokens)
    article = RawArticle(
        ticker="GOOGL",
        title="Google Announces New AI Features",
        content="Google announced today a suite of new artificial intelligence features " * 20,
        published_date=date(2024, 3, 15),
        url="https://example.com/google-ai",
        source="Reuters"
    )

    chunks = chunker.chunk_article(article)

    # Should be exactly one chunk
    assert len(chunks) == 1, f"Expected 1 chunk for short article, got {len(chunks)}"

    chunk = chunks[0]
    assert chunk.ticker == "GOOGL"
    assert chunk.doc_type == "news"
    assert chunk.source == "Reuters"
    assert chunk.url == "https://example.com/google-ai"
    assert chunk.chunk_index == 0
    assert chunk.total_chunks == 1


def test_split_into_sentences_empty_text_returns_empty_list(chunker):
    """Test that split_into_sentences handles empty input gracefully."""
    assert chunker.split_into_sentences("") == []


def test_chunk_text_empty_text_returns_empty_list(chunker):
    """Test that chunk_text returns [] for empty input rather than raising."""
    assert chunker.chunk_text("") == []


def test_chunk_text_hard_splits_oversized_single_sentence(chunker):
    """Test that a single sentence exceeding max_tokens gets hard-split by token count."""
    chunker.max_tokens = 50
    chunker.min_tokens = 10
    chunker.overlap_tokens = 5

    # One giant "sentence" (no sentence-ending punctuation) that is way over max_tokens
    huge_sentence = ("financial market analysis word " * 100).strip() + "."

    chunks = chunker.chunk_text(huge_sentence)

    # Should have been forcibly split into multiple pieces
    assert len(chunks) > 1
    for c in chunks:
        assert chunker.count_tokens(c) <= chunker.max_tokens


def test_chunk_document_skips_empty_sections(chunker):
    """Test that chunk_document skips sections with empty or whitespace-only text."""
    doc = RawDocument(
        ticker="AAPL",
        doc_type="10-K",
        filing_date=date(2024, 1, 15),
        period_of_report=date(2024, 1, 15),
        source="SEC EDGAR",
        sections={
            "item_1": "Apple Inc. designs and manufactures consumer electronics. " * 20,
            "item_2": "",
            "item_3": "   ",
        },
        file_path="/path/to/filing",
        accession_number="0001234567-24-000003",
    )

    chunks = chunker.chunk_document(doc)

    # Only item_1 should have produced chunks
    sections_present = {c.section for c in chunks}
    assert sections_present == {"item_1"}


def test_chunk_article_long_article_produces_multiple_chunks(chunker):
    """Test that a long article is split into multiple sequentially-indexed chunks."""
    long_content = "Apple reported strong quarterly results across all segments. " * 60
    article = RawArticle(
        ticker="AAPL",
        title="Apple Reports Record Quarter",
        content=long_content,
        published_date=date(2024, 3, 15),
        url="https://example.com/apple-earnings",
        source="Reuters",
    )

    chunks = chunker.chunk_article(article)

    assert len(chunks) > 1
    assert all(c.doc_type == "news" for c in chunks)
    assert all(c.total_chunks == len(chunks) for c in chunks)
    # chunk_index should be sequential starting at 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # chunk_ids should be unique
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_count_tokens_consistent_with_tiktoken(chunker):
    """Test that count_tokens returns expected values."""
    # Test known token counts
    assert chunker.count_tokens("hello world") == 2
    assert chunker.count_tokens("") == 0

    # Test longer text
    text = "The quick brown fox jumps over the lazy dog."
    token_count = chunker.count_tokens(text)
    assert token_count > 0
    assert token_count < 20  # Should be around 10 tokens
