"""Backfill SEC filings for an explicit historical window (e.g. pre-COVID).

EDGAR-only, deliberately - see IngestionPipeline.ingest_historical_filings()
for why news is skipped here (RSS feeds have no historical query capability
at all, so this script wouldn't be able to add real historical news even if
it tried). SEC EDGAR is a permanent public archive back to the 1990s, so any
window is fetchable; the default here is 2015-01-01 to 2019-12-31 (the
pre-COVID "grind" period AlphaLab's research scripts already test strategies
against - see wf_common.py's STANDARD_REGIME_WINDOWS).

Safe to re-run: the embedding cache, MetadataStore, and VectorStore are all
idempotent on the content-derived chunk_id (see ingest_historical_filings's
docstring), so re-running this - even on an overlapping or identical window -
costs nothing extra.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from alphasignal.ingestion.pipeline import IngestionPipeline
from alphasignal.scripts._common import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Backfill historical SEC filings for all configured tickers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--after",
        default="2015-01-01",
        help="Absolute start date, YYYY-MM-DD (default: 2015-01-01)",
    )
    parser.add_argument(
        "--before",
        default="2019-12-31",
        help="Absolute end date, YYYY-MM-DD (default: 2019-12-31, pre-COVID)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("AlphaSignal Historical Filings Backfill")
    print(f"Window: {args.after} to {args.before}")
    print("=" * 80)

    config = load_config(project_root)

    tickers = config.get("tickers", [])
    print(f"\nBackfilling {len(tickers)} tickers: {', '.join(tickers)}")
    print()

    pipeline = IngestionPipeline(config)

    corpus_stats = {
        "timestamp": datetime.now().isoformat(),
        "window": {"after": args.after, "before": args.before},
        "tickers": {},
    }

    results_table = []
    total_filings = 0
    total_chunks = 0

    for ticker in tickers:
        print(f"\n{'=' * 80}")
        print(f"Backfilling {ticker}...")
        print(f"{'=' * 80}")

        start_time = time.time()

        try:
            result = pipeline.ingest_historical_filings(
                ticker, after=args.after, before=args.before
            )

            elapsed = time.time() - start_time

            chunks = pipeline.metadata_store.get_chunks_by_ticker(ticker)
            filings = [
                c
                for c in chunks
                if c.doc_type in ("10-K", "10-Q")
                and args.after <= str(c.date) <= args.before
            ]

            if filings:
                dates = [c.date for c in filings]
                date_range = f"{min(dates)} to {max(dates)}"
            else:
                date_range = "N/A"

            num_filings = (
                len(set(c.chunk_id.split("_")[2] for c in filings)) if filings else 0
            )

            logger.info(
                f"Backfilled {ticker}: {result.chunks_created} chunks from "
                f"{num_filings} filings in {elapsed:.1f}s"
            )

            corpus_stats["tickers"][ticker] = {
                "chunks_created": result.chunks_created,
                "chunks_embedded": result.chunks_embedded,
                "num_filings": num_filings,
                "date_range": date_range,
                "backfill_time_seconds": round(elapsed, 2),
            }

            results_table.append(
                {
                    "ticker": ticker,
                    "filings": num_filings,
                    "chunks": result.chunks_created,
                    "date_range": date_range,
                }
            )

            total_filings += num_filings
            total_chunks += result.chunks_created

        except Exception as e:
            logger.error(f"Error backfilling {ticker}: {e}", exc_info=True)
            corpus_stats["tickers"][ticker] = {"error": str(e)}
            results_table.append(
                {"ticker": ticker, "filings": 0, "chunks": 0, "date_range": "ERROR"}
            )

    print("\n\n" + "=" * 80)
    print("HISTORICAL BACKFILL STATISTICS")
    print("=" * 80)
    print()
    print(f"{'Ticker':<8} | {'Filings':>8} | {'Chunks':>8} | Date Range")
    print("-" * 80)

    for row in results_table:
        print(
            f"{row['ticker']:<8} | "
            f"{row['filings']:>8} | "
            f"{row['chunks']:>8} | "
            f"{row['date_range']}"
        )

    print("-" * 80)
    print(f"{'TOTAL':<8} | {total_filings:>8} | {total_chunks:>8} |")
    print()

    stats_path = (
        project_root
        / "data"
        / f"corpus_stats_historical_{args.after}_{args.before}.json"
    )
    stats_path.parent.mkdir(exist_ok=True)

    with open(stats_path, "w") as f:
        json.dump(corpus_stats, f, indent=2)

    logger.info(f"Saved backfill statistics to {stats_path}")

    print("\nHistorical backfill complete:")
    print(f"  - {len(tickers)} tickers")
    print(f"  - {total_filings} SEC filings")
    print(f"  - {total_chunks} total chunks")
    print(f"  - Statistics saved to: {stats_path}")
    print()


if __name__ == "__main__":
    main()
