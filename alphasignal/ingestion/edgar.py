"""SEC EDGAR filing ingestion module."""

import logging
import re
import time
from datetime import date, datetime
from pathlib import Path

from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader

from alphasignal.ingestion import RawDocument
from alphasignal.ingestion.tables import ExtractedTable, extract_tables

logger = logging.getLogger(__name__)


def edgar_filing_url(cik: str | None, accession_number: str | None) -> str | None:
    """EDGAR filing index page for a filing, e.g.
    https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/0000320193-25-000079-index.htm
    """
    if not cik or not accession_number:
        return None
    nodash = accession_number.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{nodash}/"
        f"{accession_number}-index.htm"
    )


class EDGARIngester:
    """Downloads and parses SEC EDGAR filings for specified tickers."""

    def __init__(self, config: dict, download_dir: str = "data/edgar_raw"):
        """Initialize EDGAR ingester.

        Args:
            config: Configuration dictionary containing ingestion.edgar settings
            download_dir: Directory to store downloaded filings
        """
        self.config = config
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Get rate limit delay from config
        edgar_config = config.get("ingestion", {}).get("edgar", {})
        self.rate_limit_delay = edgar_config.get("rate_limit_delay", 0.5)

        # Initialize SEC EDGAR downloader
        self.downloader = Downloader(
            company_name="AlphaSignal",
            email_address="research@alphasignal.com",
            download_folder=str(self.download_dir),
        )

    def fetch_filings(
        self,
        ticker: str,
        filing_types: list[str],
        years_back: int | None = None,
        after: str | None = None,
        before: str | None = None,
    ) -> list[RawDocument]:
        """Download SEC filings for a ticker.

        Two mutually-usable modes: the default relative window (years_back,
        computed from today - what every regular ingestion run uses), or an
        explicit absolute window (after/before, e.g. "2015-01-01"/"2019-12-31"
        for a historical backfill). SEC EDGAR is a permanent public archive
        back to the 1990s and the underlying sec-edgar-downloader library
        already supports both bounds - after/before just wire that through,
        nothing about EDGAR itself limits how far back this can go.

        Args:
            ticker: Stock ticker symbol
            filing_types: List of filing types (e.g., ["10-K", "10-Q"])
            years_back: Number of years back from today to fetch. Ignored if
                `after` is given.
            after: Explicit absolute start date (YYYY-MM-DD). Overrides
                years_back when set.
            before: Explicit absolute end date (YYYY-MM-DD). Open-ended
                (fetches up to today) if omitted.

        Returns:
            List of RawDocument objects
        """
        if after is None and years_back is None:
            raise ValueError("fetch_filings requires either years_back or after")
        resolved_after = after or f"{datetime.now().year - years_back}-01-01"

        raw_documents = []

        for filing_type in filing_types:
            try:
                window_desc = (
                    f"{resolved_after} to {before}"
                    if before
                    else f"{resolved_after} to present"
                )
                logger.info(
                    f"Downloading {filing_type} filings for {ticker}, {window_desc}"
                )

                # Download filings
                self.downloader.get(
                    filing_type,
                    ticker,
                    after=resolved_after,
                    before=before,
                    download_details=True,
                )

                # Respect rate limit
                time.sleep(self.rate_limit_delay)

                # Find downloaded filing files
                ticker_dir = (
                    self.download_dir / "sec-edgar-filings" / ticker / filing_type
                )
                if not ticker_dir.exists():
                    logger.warning(f"No filings found for {ticker} {filing_type}")
                    continue

                # Process each filing
                for filing_dir in sorted(ticker_dir.iterdir()):
                    if not filing_dir.is_dir():
                        continue

                    # Find the primary filing document
                    filing_files = (
                        list(filing_dir.glob("primary-document.*"))
                        + list(filing_dir.glob("*.txt"))
                        + list(filing_dir.glob("*.htm*"))
                    )

                    if not filing_files:
                        logger.warning(f"No filing document found in {filing_dir}")
                        continue

                    filing_path = filing_files[0]

                    # Parse the filing
                    text = self.parse_filing(filing_path)
                    if not text:
                        logger.warning(f"Could not parse filing: {filing_path}")
                        continue

                    # Extract sections
                    sections = self.extract_sections(text, filing_type)
                    tables = self.parse_tables(filing_path, filing_type)

                    # Extract metadata from directory name (accession number)
                    accession_number = filing_dir.name

                    # Extract real filing date and period of report from the SEC header
                    filing_date, period_of_report = self.extract_filing_dates(
                        filing_dir
                    )
                    if filing_date is None:
                        # Never invent a date (it used to fall back to today,
                        # silently misdating the filing).
                        logger.warning(
                            f"Skipping {filing_dir}: no FILED AS OF DATE in SEC header"
                        )
                        continue
                    cik = self.extract_cik(filing_dir)

                    # Create RawDocument
                    raw_doc = RawDocument(
                        ticker=ticker,
                        doc_type=filing_type,
                        filing_date=filing_date,
                        period_of_report=period_of_report,
                        source="SEC EDGAR",
                        sections=sections,
                        file_path=str(filing_path),
                        accession_number=accession_number,
                        cik=cik,
                        url=edgar_filing_url(cik, accession_number),
                        tables=tables,
                    )
                    raw_documents.append(raw_doc)

                logger.info(
                    f"Fetched {len(raw_documents)} {filing_type} filings for {ticker}"
                )

            except Exception as e:
                logger.error(f"Error fetching {filing_type} filings for {ticker}: {e}")
                # Don't raise, just continue with next filing type
                continue

        return raw_documents

    def extract_filing_dates(self, filing_dir: Path) -> tuple[date, date]:
        """Extract the real filing date and period of report from the SEC SGML header.

        sec-edgar-downloader saves a full-submission.txt alongside the primary
        document, whose header contains lines like:
            CONFORMED PERIOD OF REPORT:    20221231
            FILED AS OF DATE:               20230203

        Args:
            filing_dir: Directory containing the downloaded filing

        Returns:
            Tuple of (filing_date, period_of_report); either is None when the
            header does not state it. A missing period is never replaced by
            the filing date - retrieval uses it to tell fiscal periods apart.
        """
        submission_path = filing_dir / "full-submission.txt"

        if not submission_path.exists():
            logger.warning(f"No full-submission.txt in {filing_dir}")
            return None, None

        try:
            with open(submission_path, "r", encoding="utf-8", errors="ignore") as f:
                header = f.read(4000)  # header appears within the first few KB
        except Exception as e:
            logger.warning(f"Could not read {submission_path}: {e}")
            return None, None

        filed_match = re.search(r"FILED AS OF DATE:\s*(\d{8})", header)
        period_match = re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})", header)

        if not filed_match:
            logger.warning(f"Could not find FILED AS OF DATE in {submission_path}")
            return None, None
        filing_date = datetime.strptime(filed_match.group(1), "%Y%m%d").date()
        period_of_report = (
            datetime.strptime(period_match.group(1), "%Y%m%d").date()
            if period_match
            else None
        )

        return filing_date, period_of_report

    def extract_cik(self, filing_dir: Path) -> str | None:
        """CENTRAL INDEX KEY of the filer from the SEC header, or None."""
        try:
            with open(
                filing_dir / "full-submission.txt",
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as f:
                header = f.read(6000)
        except Exception:
            return None
        m = re.search(r"CENTRAL INDEX KEY:\s*(\d+)", header)
        return str(int(m.group(1))) if m else None

    def parse_filing(self, filing_path: Path) -> str:
        """Parse a SEC filing and extract clean text.

        Args:
            filing_path: Path to filing file (HTML or TXT)

        Returns:
            Cleaned text content, or empty string if parsing fails
        """
        try:
            with open(filing_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Determine file type
            if filing_path.suffix.lower() in [".htm", ".html"]:
                # Parse HTML
                soup = BeautifulSoup(content, "lxml")

                # Remove unwanted tags
                for tag in soup.find_all(["script", "style", "table"]):
                    tag.decompose()

                # Extract text
                text = soup.get_text()

            else:
                # Plain text file - strip SEC header boilerplate
                text = content

                # Remove SEC header (everything before "FORM TYPE:" or "SECURITIES AND EXCHANGE")
                lines = text.split("\n")
                start_idx = 0
                for i, line in enumerate(lines):
                    if re.search(
                        r"(FORM TYPE:|SECURITIES AND EXCHANGE COMMISSION)",
                        line,
                        re.IGNORECASE,
                    ):
                        start_idx = i
                        break
                text = "\n".join(lines[start_idx:])

            # Normalize whitespace
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"\n\s*\n", "\n\n", text)
            text = text.strip()

            # Check minimum length
            if len(text) < 500:
                logger.warning(f"Filing too short after cleaning: {len(text)} chars")
                return ""

            return text

        except Exception as e:
            logger.error(f"Error parsing filing {filing_path}: {e}")
            return ""

    def parse_tables(
        self, filing_path: Path, doc_type: str
    ) -> list[tuple[str, ExtractedTable]]:
        """Numeric tables of an HTML filing with the section each sits in.

        Separate from parse_filing so the prose text (and therefore every
        existing prose chunk id and content hash) is unchanged. A failure
        here never blocks the prose ingest - the filing is kept without
        tables and the error logged.
        """
        if filing_path.suffix.lower() not in (".htm", ".html"):
            return []
        try:
            html = filing_path.read_text(encoding="utf-8", errors="ignore")
            tables, marked = extract_tables(html)
            sections = self.extract_sections(marked, doc_type)
            where: dict[int, str] = {}
            for name, body in sections.items():
                for m in re.finditer(r"\[\[TABLE (\d+)\]\]", body):
                    where.setdefault(int(m.group(1)), name)
            return [(where.get(t.index, "front_matter"), t) for t in tables]
        except Exception as e:
            logger.error(f"Table extraction failed for {filing_path}: {e}")
            return []

    def extract_sections(self, text: str, doc_type: str) -> dict[str, str]:
        """Extract specific sections from a SEC filing.

        Args:
            text: Full text of the filing
            doc_type: Filing type ("10-K" or "10-Q")

        Returns:
            Dictionary mapping section names to section text
        """
        sections = {}

        try:
            if doc_type == "10-K":
                # Define section patterns for 10-K
                section_patterns = {
                    "item_1": r"item\s+1[\.\s:]+(?:business|description)",
                    "item_1a": r"item\s+1a[\.\s:]+risk\s+factors",
                    "item_7": r"item\s+7[\.\s:]+management",
                    "item_7a": r"item\s+7a[\.\s:]+quantitative",
                    "item_8": r"item\s+8[\.\s:]+financial\s+statements",
                }

            elif doc_type == "10-Q":
                # Define section patterns for 10-Q
                section_patterns = {
                    "item_1": r"item\s+1[\.\s:]+financial\s+statements",
                    "item_2": r"item\s+2[\.\s:]+management",
                    "item_3": r"item\s+3[\.\s:]+quantitative",
                    "item_4": r"item\s+4[\.\s:]+controls",
                }

            else:
                # Unknown doc type
                return {"full_text": text}

            # Try to find sections
            text_lower = text.lower()
            matches = []

            for section_name, pattern in section_patterns.items():
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    matches.append((match.start(), section_name))

            # If we found sections, extract them
            if len(matches) >= 2:  # Need at least 2 sections to split
                # Sort by position
                matches.sort()

                for i in range(len(matches)):
                    start_pos = matches[i][0]
                    section_name = matches[i][1]

                    # End position is start of next section, or end of text
                    end_pos = matches[i + 1][0] if i + 1 < len(matches) else len(text)

                    # Extract section text
                    section_text = text[start_pos:end_pos].strip()
                    sections[section_name] = section_text

            else:
                # Couldn't find standard sections - use full text
                logger.info(
                    f"Could not extract standard sections from {doc_type}, using full text"
                )
                sections = {"full_text": text}

        except Exception as e:
            logger.error(f"Error extracting sections: {e}")
            sections = {"full_text": text}

        return sections
