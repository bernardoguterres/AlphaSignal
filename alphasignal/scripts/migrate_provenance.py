"""Backfill provenance on an existing corpus - only where it can be proven.

For every downloaded filing under <data>/edgar_raw/sec-edgar-filings/
{TICKER}/{FORM}/{ACCESSION}/, the chunker's deterministic document prefix
is recomputed exactly as SemanticChunker.document_source_prefix() does
(md5 of ticker + SEC-header filing date + accession). Chunks whose
chunk_id carries that prefix get accession_number, the EDGAR filing URL,
source_id and attribution="filer". Filing chunks with no matching raw
filing are left untouched and reported - nothing is guessed.

News chunks are re-attributed per article from its first chunk's stored text
("Title. body"): "news_title" if the company is named as a whole word in the
headline, else "news_body". Every chunk of an article gets the same value.

Usage:
    python -m alphasignal.scripts.migrate_provenance --db data/metadata.db \
        --edgar-raw data/edgar_raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sqlalchemy import text

from alphasignal.ingestion.edgar import EDGARIngester, edgar_filing_url
from alphasignal.ingestion.news import mentions_company
from alphasignal.store.metadata_store import MetadataStore


def filing_prefix(ticker: str, form: str, filing_date, accession: str) -> str:
    h = hashlib.md5(f"{ticker}{filing_date}{accession}".encode()).hexdigest()[:8]
    return f"{ticker.lower()}_{form.lower().replace('-', '')}_{h}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--edgar-raw", required=True)
    args = ap.parse_args()

    store = MetadataStore(args.db)  # also applies the schema upgrade
    reader = EDGARIngester.__new__(EDGARIngester)  # header helpers only, no network
    root = Path(args.edgar_raw) / "sec-edgar-filings"

    report = {
        "filings_seen": 0,
        "filings_matched": 0,
        "chunks_updated": 0,
        "filings_without_chunks": [],
        "header_unreadable": [],
    }
    with store.engine.begin() as conn:
        for filing_dir in sorted(root.glob("*/*/*")):
            if not filing_dir.is_dir():
                continue
            ticker, form, accession = (
                filing_dir.parts[-3],
                filing_dir.parts[-2],
                filing_dir.name,
            )
            report["filings_seen"] += 1
            filing_date, _ = reader.extract_filing_dates(filing_dir)
            if filing_date is None:
                report["header_unreadable"].append(str(filing_dir))
                continue
            cik = reader.extract_cik(filing_dir)
            prefix = filing_prefix(ticker, form, filing_date, accession)
            res = conn.execute(
                text(
                    "UPDATE chunks SET accession_number=:acc, url=:url, source_id=:sid, "
                    "attribution='filer' WHERE substr(chunk_id, 1, :n) = :pfx "
                    "AND doc_type != 'news'"
                ),
                {
                    "acc": accession,
                    "url": edgar_filing_url(cik, accession),
                    "sid": prefix,
                    "n": len(prefix) + 1,
                    "pfx": prefix + "_",
                },
            )
            if res.rowcount:
                report["filings_matched"] += 1
                report["chunks_updated"] += res.rowcount
            else:
                report["filings_without_chunks"].append(f"{ticker}/{form}/{accession}")

        news = conn.execute(
            text(
                "SELECT chunk_id, ticker, text FROM chunks WHERE doc_type='news' "
                "ORDER BY chunk_id"
            )
        ).fetchall()
        # Only an article's first chunk starts with its headline; chunk ids
        # are "{article prefix}_{index:04d}", so later chunks inherit it.
        first_text = {}
        for cid, _, body in news:
            first_text.setdefault(cid.rsplit("_", 1)[0], body)
        attrib = {"news_title": 0, "news_body": 0}
        for cid, ticker, _ in news:
            title = first_text[cid.rsplit("_", 1)[0]].split(". ", 1)[0]
            a = "news_title" if mentions_company(title, ticker) else "news_body"
            attrib[a] += 1
            conn.execute(
                text("UPDATE chunks SET attribution=:a WHERE chunk_id=:c"),
                {"a": a, "c": cid},
            )
        report["news_attribution"] = attrib
        left = conn.execute(
            text(
                "SELECT count(*) FROM chunks WHERE doc_type!='news' AND accession_number IS NULL"
            )
        ).scalar()
        report["filing_chunks_without_provenance"] = int(left)

    report["filings_without_chunks"] = report["filings_without_chunks"][:20]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
