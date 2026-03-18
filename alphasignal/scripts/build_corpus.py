"""Build the complete corpus by ingesting all configured tickers."""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from alphasignal.ingestion.pipeline import IngestionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Build the complete corpus by ingesting all tickers."""
    print("=" * 80)
    print("AlphaSignal Corpus Builder")
    print("=" * 80)

    # Load configuration
    config_path = project_root / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    tickers = config.get("tickers", [])
    print(f"\nIngesting {len(tickers)} tickers: {', '.join(tickers)}")
    print()

    # Initialize pipeline
    pipeline = IngestionPipeline(config)

    # Track statistics
    corpus_stats = {
        "timestamp": datetime.now().isoformat(),
        "tickers": {}
    }

    results_table = []
    total_filings = 0
    total_articles = 0
    total_chunks = 0

    # Ingest each ticker
    for ticker in tickers:
        print(f"\n{'=' * 80}")
        print(f"Ingesting {ticker}...")
        print(f"{'=' * 80}")

        start_time = time.time()

        try:
            # Run full ingestion
            result = pipeline.full_ingest(ticker)

            elapsed = time.time() - start_time

            # Get chunk details
            chunks = pipeline.metadata_store.get_chunks_by_ticker(ticker)

            # Count filings vs articles
            filings = [c for c in chunks if c.doc_type in ["10-K", "10-Q"]]
            articles = [c for c in chunks if c.doc_type == "news"]

            # Get date range
            if chunks:
                dates = [c.date for c in chunks]
                date_range = f"{min(dates)} to {max(dates)}"
            else:
                date_range = "N/A"

            num_filings = len(set(c.chunk_id.split("_")[2] for c in filings)) if filings else 0
            num_articles = len(set(c.url for c in articles if c.url)) if articles else 0

            # Log progress
            logger.info(
                f"Ingested {ticker}: "
                f"{result.chunks_created} chunks from "
                f"{num_filings} filings, "
                f"{num_articles} articles "
                f"in {elapsed:.1f}s"
            )

            # Store statistics
            corpus_stats["tickers"][ticker] = {
                "chunks_created": result.chunks_created,
                "chunks_embedded": result.chunks_embedded,
                "chunks_stored": result.chunks_stored,
                "num_filings": num_filings,
                "num_articles": num_articles,
                "date_range": date_range,
                "ingestion_time_seconds": round(elapsed, 2)
            }

            # Add to results table
            results_table.append({
                "ticker": ticker,
                "filings": num_filings,
                "articles": num_articles,
                "chunks": result.chunks_created,
                "date_range": date_range
            })

            total_filings += num_filings
            total_articles += num_articles
            total_chunks += result.chunks_created

        except Exception as e:
            logger.error(f"Error ingesting {ticker}: {e}", exc_info=True)
            corpus_stats["tickers"][ticker] = {
                "error": str(e)
            }
            results_table.append({
                "ticker": ticker,
                "filings": 0,
                "articles": 0,
                "chunks": 0,
                "date_range": "ERROR"
            })

    # Print summary table
    print("\n\n" + "=" * 80)
    print("CORPUS STATISTICS")
    print("=" * 80)
    print()
    print(f"{'Ticker':<8} | {'Filings':>8} | {'Articles':>8} | {'Chunks':>8} | Date Range")
    print("-" * 80)

    for row in results_table:
        print(
            f"{row['ticker']:<8} | "
            f"{row['filings']:>8} | "
            f"{row['articles']:>8} | "
            f"{row['chunks']:>8} | "
            f"{row['date_range']}"
        )

    print("-" * 80)
    print(
        f"{'TOTAL':<8} | "
        f"{total_filings:>8} | "
        f"{total_articles:>8} | "
        f"{total_chunks:>8} |"
    )
    print()

    # Save corpus statistics
    stats_path = project_root / "data" / "corpus_stats.json"
    stats_path.parent.mkdir(exist_ok=True)

    with open(stats_path, "w") as f:
        json.dump(corpus_stats, f, indent=2)

    logger.info(f"Saved corpus statistics to {stats_path}")

    # Print summary
    print(f"\nCorpus built successfully:")
    print(f"  - {len(tickers)} tickers")
    print(f"  - {total_filings} SEC filings")
    print(f"  - {total_articles} news articles")
    print(f"  - {total_chunks} total chunks")
    print(f"  - Statistics saved to: {stats_path}")
    print()


if __name__ == "__main__":
    main()
