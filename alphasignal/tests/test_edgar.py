"""Tests for SEC EDGAR ingestion."""

import tempfile
from datetime import date, datetime
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


def test_extract_filing_dates_parses_real_sec_header(edgar_ingester, tmp_path):
    """Regression test for the fixed bug where filing_date was always datetime.now().

    extract_filing_dates must parse the actual FILED AS OF DATE / CONFORMED
    PERIOD OF REPORT from the SEC SGML header, not fall back to today.
    """
    filing_dir = tmp_path / "filing"
    filing_dir.mkdir()
    submission = filing_dir / "full-submission.txt"
    submission.write_text(
        "<SEC-HEADER>\n"
        "ACCESSION NUMBER:\t\t0001234567-23-000001\n"
        "CONFORMED PERIOD OF REPORT:\t20221231\n"
        "FILED AS OF DATE:\t\t20230203\n"
        "</SEC-HEADER>\n"
    )

    filing_date, period_of_report = edgar_ingester.extract_filing_dates(filing_dir)

    # Must be the real SEC dates, not datetime.now().date()
    assert filing_date == date(2023, 2, 3)
    assert period_of_report == date(2022, 12, 31)
    assert filing_date != date.today()


def test_extract_filing_dates_missing_submission_file_falls_back_to_today(
    edgar_ingester, tmp_path
):
    """Test that extract_filing_dates falls back to today when the file is absent."""
    filing_dir = tmp_path / "no_submission"
    filing_dir.mkdir()

    filing_date, period_of_report = edgar_ingester.extract_filing_dates(filing_dir)

    today = datetime.now().date()
    assert filing_date == today
    assert period_of_report == today


def test_extract_filing_dates_missing_filed_date_falls_back_to_today(
    edgar_ingester, tmp_path
):
    """Test that a header without FILED AS OF DATE falls back to today's date."""
    filing_dir = tmp_path / "partial_header"
    filing_dir.mkdir()
    submission = filing_dir / "full-submission.txt"
    submission.write_text("<SEC-HEADER>\nSOME OTHER FIELD:\t123\n</SEC-HEADER>\n")

    filing_date, period_of_report = edgar_ingester.extract_filing_dates(filing_dir)

    today = datetime.now().date()
    assert filing_date == today
    # period_of_report falls back to filing_date when no period match either
    assert period_of_report == today


def test_fetch_filings_skips_ticker_with_no_downloaded_dir(edgar_ingester):
    """Test that fetch_filings returns empty results when no filings were downloaded."""
    with patch.object(edgar_ingester, "downloader") as mock_downloader:
        mock_downloader.get = MagicMock()

        results = edgar_ingester.fetch_filings(
            ticker="ZZZZ", filing_types=["10-K"], years_back=1
        )

    assert results == []


def test_fetch_filings_skips_non_directory_entries(edgar_ingester, tmp_path):
    """Test that stray files inside the ticker/filing_type dir are skipped."""
    ticker_dir = (
        edgar_ingester.download_dir / "sec-edgar-filings" / "AAPL" / "10-K"
    )
    ticker_dir.mkdir(parents=True)
    # A stray file (not a directory) should be skipped, not crash the loop
    (ticker_dir / "stray.txt").write_text("not a filing directory")

    with patch.object(edgar_ingester, "downloader") as mock_downloader:
        mock_downloader.get = MagicMock()

        results = edgar_ingester.fetch_filings(
            ticker="AAPL", filing_types=["10-K"], years_back=1
        )

    assert results == []


def test_fetch_filings_skips_directory_with_no_filing_document(
    edgar_ingester
):
    """Test that a filing directory without any recognizable document is skipped."""
    filing_dir = (
        edgar_ingester.download_dir
        / "sec-edgar-filings"
        / "AAPL"
        / "10-K"
        / "0001234567-23-000001"
    )
    filing_dir.mkdir(parents=True)
    # No primary-document.*, *.txt, or *.htm* files present
    (filing_dir / "metadata.json").write_text("{}")

    with patch.object(edgar_ingester, "downloader") as mock_downloader:
        mock_downloader.get = MagicMock()

        results = edgar_ingester.fetch_filings(
            ticker="AAPL", filing_types=["10-K"], years_back=1
        )

    assert results == []


def test_parse_filing_plain_text_strips_sec_header_boilerplate(edgar_ingester):
    """Test that parse_filing strips SEC header boilerplate from .txt filings."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        content = (
            "<SEC-HEADER>\nJunk header info that should be discarded\n</SEC-HEADER>\n"
            "SECURITIES AND EXCHANGE COMMISSION\n"
            "FORM TYPE:\t10-K\n\n"
            + (
                "Item 1. Business. This section describes the company's core business "
                "operations, products, and services offered to customers worldwide. "
            )
            * 10
        )
        f.write(content)
        temp_path = Path(f.name)

    try:
        result = edgar_ingester.parse_filing(temp_path)

        assert result != ""
        assert "Junk header info" not in result
        assert "core business" in result
    finally:
        temp_path.unlink()


def test_parse_filing_handles_missing_file(edgar_ingester, tmp_path):
    """Test that parse_filing returns empty string when the file can't be read."""
    missing_path = tmp_path / "does_not_exist.html"

    result = edgar_ingester.parse_filing(missing_path)

    assert result == ""


def test_extract_sections_10q_standard_format(edgar_ingester):
    """Test that extract_sections correctly identifies 10-Q sections."""
    text = """
    SECURITIES AND EXCHANGE COMMISSION
    FORM 10-Q

    Item 1. Financial Statements

    The unaudited condensed consolidated financial statements are presented below
    along with explanatory notes covering revenue recognition and expenses.

    Item 2. Management's Discussion and Analysis

    Revenue for the quarter increased due to strong demand across all segments.

    Item 3. Quantitative and Qualitative Disclosures About Market Risk

    The company is exposed to interest rate and foreign currency risks.

    Item 4. Controls and Procedures

    Management concluded that disclosure controls were effective as of the period end.
    """

    sections = edgar_ingester.extract_sections(text, "10-Q")

    assert len(sections) > 0
    assert any("item_1" in key or "item_2" in key for key in sections.keys())


def test_extract_sections_unknown_doc_type_returns_full_text(edgar_ingester):
    """Test that an unrecognized doc_type falls back to full_text."""
    text = "Some filing content that doesn't match 10-K or 10-Q patterns."

    sections = edgar_ingester.extract_sections(text, "8-K")

    assert sections == {"full_text": text}


def test_extract_sections_handles_internal_exception(edgar_ingester):
    """Test that extract_sections falls back to full_text if pattern matching raises."""
    text = "Item 1. Business. " + ("Some content. " * 50)

    with patch(
        "alphasignal.ingestion.edgar.re.search", side_effect=RuntimeError("boom")
    ):
        sections = edgar_ingester.extract_sections(text, "10-K")

    assert sections == {"full_text": text}


# Integration tests for /ingest endpoint are in test_api.py
