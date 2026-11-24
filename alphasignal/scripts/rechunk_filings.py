"""Re-process the SEC filings already downloaded under data/edgar_raw.

Parses, sections and chunks every local filing with the CURRENT code
(``CHUNKER_VERSION``), embeds the chunks the embedding cache does not hold
(OpenAI, paid) and stores them; each filing's previous chunks are removed
by ``IngestionPipeline.store_chunks`` (orphan clean-up by source_id). Nothing
is downloaded from the SEC and news chunks are untouched.

Take a snapshot first (``corpus_version snapshot --with-cache``) and stop the
API while this runs (single writer).

    python -m alphasignal.scripts.rechunk_filings --dry-run      # counts + cost estimate, writes nothing
    python -m alphasignal.scripts.rechunk_filings [--tickers JPM GS] [--include-tables]

Tables are left out unless ``--include-tables`` is given, so a run only
changes what the parser/chunker change touches (the stored corpus has no
table chunks yet).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv  # noqa: E402

from alphasignal.ingestion.chunker import CHUNKER_VERSION  # noqa: E402
from alphasignal.ingestion.pipeline import IngestionPipeline  # noqa: E402
from alphasignal.scripts._common import load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rechunk_filings")

# text-embedding-3-small list price, USD per 1M input tokens
PRICE_PER_M_TOKENS = 0.02


def _local_docs(pipeline: IngestionPipeline, ticker: str, include_tables: bool):
    """RawDocuments for every local filing of `ticker`, via the normal
    fetch_filings() path with the SEC download switched off."""
    ing = pipeline.edgar_ingester
    ing.downloader.get = lambda *a, **k: None  # local files only
    ing.rate_limit_delay = 0
    types = (
        pipeline.config.get("ingestion", {})
        .get("edgar", {})
        .get("filing_types", ["10-K", "10-Q"])
    )
    docs = ing.fetch_filings(
        ticker=ticker, filing_types=types, after="1990-01-01", before="2100-01-01"
    )
    if not include_tables:
        for d in docs:
            d.tables = []
    return docs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tickers", nargs="*", help="default: config tickers")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-tables", action="store_true")
    args = ap.parse_args()

    load_dotenv(project_root / ".env")  # OPENAI_API_KEY for the embedder
    config = load_config(project_root)
    tickers = args.tickers or config.get("tickers", [])
    pipeline = IngestionPipeline(config)

    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    totals = {"filings": 0, "chunks": 0, "to_embed": 0, "tokens": 0}
    for ticker in tickers:
        t0 = time.time()
        docs = _local_docs(pipeline, ticker, args.include_tables)
        chunks = pipeline.chunk_documents(docs)
        hashes = {c.chunk_id: c.content_hash for c in chunks}
        _, uncached = pipeline.embedder.cache.get_many(hashes)
        tokens = sum(len(enc.encode(c.text)) for c in chunks if c.chunk_id in uncached)
        totals["filings"] += len(docs)
        totals["chunks"] += len(chunks)
        totals["to_embed"] += len(uncached)
        totals["tokens"] += tokens
        line = (
            f"{ticker}: {len(docs)} filings, {len(chunks)} chunks ({CHUNKER_VERSION}), "
            f"{len(uncached)} to embed (~{tokens:,} tokens)"
        )
        if args.dry_run:
            logger.info(line)
            continue
        sources = {pipeline.chunker.document_source_prefix(d) for d in docs}
        embeddings = pipeline.embedder.embed_chunks(chunks)
        pipeline.store_chunks(chunks, embeddings, source_prefixes=sources)
        logger.info(f"{line} - stored in {time.time() - t0:.0f}s")

    cost = totals["tokens"] / 1e6 * PRICE_PER_M_TOKENS
    logger.info(
        f"TOTAL: {totals['filings']} filings, {totals['chunks']} chunks, "
        f"{totals['to_embed']} to embed, ~{totals['tokens']:,} tokens "
        f"(~${cost:.2f} at ${PRICE_PER_M_TOKENS}/1M){' [dry run]' if args.dry_run else ''}"
    )


if __name__ == "__main__":
    main()
