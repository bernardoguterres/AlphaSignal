"""Backfill each filing chunk's reporting period from its SEC header.

For every downloaded filing under <edgar-raw>/sec-edgar-filings/{TICKER}/
{FORM}/{ACCESSION}/, the "CONFORMED PERIOD OF REPORT" in full-submission.txt
is written to `period_end` on the chunks carrying that accession number.
Filings without a raw copy, or whose header states no period, are left NULL
and reported - nothing is inferred from the filing date. News is untouched.

The database is never edited in place: the update runs on a copy next to
it, is verified, the original is kept as <db>.pre-period-<date>, and the
copy replaces it with one atomic rename. Restart the service afterwards.

Usage:
    python -m alphasignal.scripts.migrate_reporting_period --db data/metadata.db \
        --edgar-raw data/edgar_raw [--edgar-raw other/edgar_raw] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import date
from pathlib import Path

from sqlalchemy import text

from alphasignal.ingestion.edgar import EDGARIngester
from alphasignal.store.metadata_store import MetadataStore


def header_periods(raw_roots: list[Path]) -> tuple[dict[str, date], list[str]]:
    reader = EDGARIngester.__new__(EDGARIngester)  # header helpers only
    periods: dict[str, date] = {}
    no_period: list[str] = []
    for root in raw_roots:
        for filing_dir in sorted((root / "sec-edgar-filings").glob("*/*/*")):
            if not filing_dir.is_dir():
                continue
            _, period = reader.extract_filing_dates(filing_dir)
            if period is None:
                no_period.append(filing_dir.name)
            else:
                periods.setdefault(filing_dir.name, period)
    return periods, no_period


def backfill(db: Path, periods: dict[str, date]) -> dict:
    store = MetadataStore(str(db))  # adds the period_end column if missing
    report = {"filings_updated": 0, "chunks_updated": 0}
    with store.engine.begin() as conn:
        for acc, period in sorted(periods.items()):
            res = conn.execute(
                text(
                    "UPDATE chunks SET period_end = :p "
                    "WHERE accession_number = :a AND doc_type != 'news'"
                ),
                {"p": period.isoformat(), "a": acc},
            )
            if res.rowcount:
                report["filings_updated"] += 1
                report["chunks_updated"] += res.rowcount
        missing = conn.execute(
            text(
                "SELECT ticker, doc_type, accession_number, count(*) FROM chunks "
                "WHERE doc_type != 'news' AND period_end IS NULL "
                "GROUP BY accession_number"
            )
        ).fetchall()
        report["filings_without_period"] = [list(r) for r in missing]
        report["filing_chunks_total"] = conn.execute(
            text("SELECT count(*) FROM chunks WHERE doc_type != 'news'")
        ).scalar()
    store.engine.dispose()
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--edgar-raw", required=True, action="append", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="report only, no swap")
    args = ap.parse_args()

    db = args.db.resolve()
    periods, no_period = header_periods(args.edgar_raw)
    work = db.with_name(db.name + ".period-tmp")
    shutil.copy2(db, work)
    try:
        report = backfill(work, periods)
        report["raw_filings_with_period"] = len(periods)
        report["raw_filings_without_period"] = no_period
        if args.dry_run:
            report["published"] = False
        else:
            backup = db.with_name(f"{db.name}.pre-period-{date.today().isoformat()}")
            if not backup.exists():
                shutil.copy2(db, backup)
            os.replace(work, db)  # atomic on one filesystem
            report["published"] = True
            report["backup"] = str(backup)
    finally:
        if work.exists():
            work.unlink()
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
