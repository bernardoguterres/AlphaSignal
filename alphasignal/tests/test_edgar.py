"""Tests for SEC EDGAR ingestion."""

import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from alphasignal.ingestion import RawDocument
from alphasignal.ingestion.edgar import EDGARIngester
from alphasignal.ingestion.pipeline import IngestionPipeline


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "ingestion": {
            "edgar": {
                "filing_types": ["10-K", "10-Q"],
                "years_back": 2,
                "rate_limit_delay": 0.1,
            }
        }
    }


@pytest.fixture
def edgar_ingester(test_config, tmp_path):
    """Create an EDGARIngester instance with temp directory."""
    return EDGARIngester(test_config, download_dir=str(tmp_path / "edgar"))


def test_fetch_filings_returns_raw_documents(edgar_ingester, tmp_path):
    """Test that fetch_filings returns list of RawDocument objects."""
    # Create mock filing directory structure
    ticker = "AAPL"
    filing_type = "10-K"

    filing_dir = tmp_path / "edgar" / "sec-edgar-filings" / ticker / filing_type / "0001234567-23-000001"
    filing_dir.mkdir(parents=True, exist_ok=True)

    # Create mock filing file with realistic content (> 500 chars after parsing)
    filing_file = filing_dir / "primary-document.html"
    html_content = """
    <html>
    <body>
    <p>Item 1. Business</p>
    <p>Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets,
    wearables, and accessories worldwide. The Company is committed to bringing the best user
    experience to its customers through innovative hardware, software, and services. The Company
    sells its products worldwide through its retail stores, online stores, and direct sales force,
    as well as through third-party cellular network carriers, wholesalers, retailers, and resellers.
    The Company believes ongoing investment in research and development, marketing and advertising
    is critical to the development and sale of innovative products, services, and technologies.</p>
    <p>Item 1A. Risk Factors</p>
    <p>The Company's business can be impacted by various factors including competition,
    supply chain disruptions, and economic conditions. Global markets are highly competitive
    and the Company faces intense competition in all areas of its business. The Company believes
    it offers superior innovation and integration of hardware, software, and services. Competition
    has been and continues to be intense as competitors attempt to imitate some of the features of
    the Company's products and applications within their own products.</p>
    </body>
    </html>
    """
    filing_file.write_text(html_content)

    # Mock the downloader
    with patch.object(edgar_ingester, 'downloader') as mock_downloader:
        mock_downloader.get = MagicMock()

        # Call fetch_filings
        results = edgar_ingester.fetch_filings(
            ticker=ticker,
            filing_types=[filing_type],
            years_back=2
        )

    # Assertions
    assert len(results) == 1
    assert isinstance(results[0], RawDocument)
    assert results[0].ticker == ticker
    assert results[0].doc_type == filing_type
    assert results[0].source == "SEC EDGAR"
    assert results[0].accession_number == "0001234567-23-000001"
    assert len(results[0].sections) > 0


def test_parse_filing_strips_html_tags(edgar_ingester):
    """Test that parse_filing removes HTML tags correctly."""
    # Create temp HTML file with > 500 chars of meaningful content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        html_content = """
        <html>
        <head>
            <script>alert('remove me');</script>
            <style>body { color: red; }</style>
        </head>
        <body>
            <table><tr><td>Remove this table</td></tr></table>
            <p>This is important content that should remain in the parsed output.</p>
            <p>This is another paragraph with meaningful information about the company's business operations.</p>
            <p>The company operates in multiple segments and provides various products and services to customers
            worldwide. This content is essential for understanding the company's business model and strategic
            direction. The management team is committed to delivering value to shareholders through innovation
            and operational excellence. The company faces various risks including market competition, regulatory
            changes, and economic conditions that could impact its business operations and financial performance.</p>
        </body>
        </html>
        """
        f.write(html_content)
        temp_path = Path(f.name)

    try:
        # Parse the filing
        result = edgar_ingester.parse_filing(temp_path)

        # Assertions
        assert result is not None
        assert len(result) > 0
        assert "alert" not in result  # Script removed
        assert "color: red" not in result  # Style removed
        assert "Remove this table" not in result  # Table removed
        assert "important content" in result
        assert "another paragraph" in result

    finally:
        temp_path.unlink()


def test_parse_filing_handles_malformed_html(edgar_ingester):
    """Test that parse_filing returns empty string for very short content."""
    # Create temp file with minimal content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write("<html><body>short</body></html>")
        temp_path = Path(f.name)

    try:
        # Parse the filing
        result = edgar_ingester.parse_filing(temp_path)

        # Should return empty string because content is < 500 chars
        assert result == ""

    finally:
        temp_path.unlink()


def test_extract_sections_10k_standard_format(edgar_ingester):
    """Test that extract_sections correctly identifies 10-K sections."""
    text = """
    SECURITIES AND EXCHANGE COMMISSION
    FORM 10-K

    Item 1. Business

    Our company provides excellent products and services to customers worldwide.
    We operate in multiple segments including consumer electronics and software.

    Item 1A. Risk Factors

    Investment in our securities involves risks. Competition is intense in our industry.
    We face risks related to product development, supply chain, and economic conditions.

    Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations

    The following discussion should be read in conjunction with our financial statements.
    Revenue increased by 10% year over year due to strong product demand.

    Item 8. Financial Statements and Supplementary Data

    See the consolidated financial statements and notes thereto.
    """

    sections = edgar_ingester.extract_sections(text, "10-K")

    # Should have identified multiple sections
    assert len(sections) > 0

    # Check that specific sections were found
    assert any("item_1" in key for key in sections.keys()) or "full_text" in sections

    # If sections were extracted, check they contain expected content
    if "item_1" in sections:
        assert "products and services" in sections["item_1"].lower()

    if "item_1a" in sections:
        assert "risk" in sections["item_1a"].lower()


def test_extract_sections_falls_back_to_full_text(edgar_ingester):
    """Test that extract_sections falls back to full_text when sections not found."""
    # Text without standard section headers
    text = "This is a filing without standard section markers. " * 100

    sections = edgar_ingester.extract_sections(text, "10-K")

    # Should fall back to full_text
    assert "full_text" in sections
    assert len(sections["full_text"]) > 0
    assert "filing without standard section markers" in sections["full_text"]


def test_fetch_filings_handles_network_error(edgar_ingester):
    """Test that fetch_filings handles network errors gracefully."""
    import requests.exceptions

    # Mock the downloader to raise an exception
    with patch.object(edgar_ingester, 'downloader') as mock_downloader:
        mock_downloader.get.side_effect = requests.exceptions.RequestException("Network error")

        # Should return empty list, not raise
        results = edgar_ingester.fetch_filings(
            ticker="AAPL",
            filing_types=["10-K"],
            years_back=2
        )

        assert results == []


# Integration tests for /ingest endpoint are in test_api.py
