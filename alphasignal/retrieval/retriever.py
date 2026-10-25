"""Hybrid retrieval module combining dense and sparse search."""

import logging
import re
import threading
from datetime import date

import numpy as np
from rank_bm25 import BM25Okapi

from alphasignal.embeddings.embedder import Embedder
from alphasignal.ingestion.tables import fold_dashes
from alphasignal.retrieval import RetrievedChunk
from alphasignal.retrieval.periods import (
    column_matches,
    expected_period_end,
    fiscal_label,
    requested_periods,
    source_header,
)
from alphasignal.store.metadata_store import MetadataStore
from alphasignal.store.vector_store import VectorStore

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")

# Table bases that are not the company's reported results (see
# ingestion.tables.table_basis); such tables only enter the table lane when
# the question asks for that basis.
_REPORTED_TABLES_CAP = 15  # reported statement tables always offered to the lane
_NON_REPORTED_BASES = {"PRO FORMA", "NON-GAAP", "MANAGED BASIS", "CONSTANT CURRENCY"}
# Words by which a question asks for a non-reported basis (dashes folded
# first). Deliberately narrow: "managed care revenue" or "adjusted for the
# stock split" are not requests for a managed or non-GAAP basis.
_BASIS_WORDS = {
    "PRO FORMA": r"pro[\s-]*forma",
    "NON-GAAP": r"non[\s-]*gaap|\badjusted\s+(?:ebitda|ebit|eps|earnings|net\s+income|operating\s+income|operating\s+margin|diluted)",
    "MANAGED BASIS": r"managed[\s-]+basis",
    "CONSTANT CURRENCY": r"constant[\s-]+currency",
}


def _basis_parts(basis: str) -> set[str]:
    """{"PRO FORMA", "NON-GAAP"} for "PRO FORMA / NON-GAAP"."""
    return {b.strip() for b in basis.split(" / ") if b.strip()}


def _basis_of(text: str) -> str:
    """The table's basis name from its "Basis:" line ("" if absent)."""
    line = next((ln for ln in text.split("\n") if ln.startswith("Basis: ")), "")
    return line[len("Basis: ") :].split(" (")[0].strip()


def requested_bases(query: str) -> set[str]:
    """Bases the question explicitly asks for; {"reported"} when none."""
    query = fold_dashes(query)
    asked = {b for b, rx in _BASIS_WORDS.items() if re.search(rx, query, re.IGNORECASE)}
    return asked or {"reported"}


def _asks_for_basis(query: str, basis: str) -> bool:
    return bool(_basis_parts(basis) & requested_bases(query))


class HybridRetriever:
    """Combines BM25 sparse retrieval and FAISS dense retrieval."""

    def __init__(
        self,
        config: dict,
        embedder: Embedder,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
    ):
        """Initialize hybrid retriever.

        Args:
            config: Configuration dictionary
            embedder: Embedder instance for query embedding
            vector_store: FAISS vector store
            metadata_store: SQLite metadata store
        """
        self.config = config
        self.embedder = embedder
        self.vector_store = vector_store
        self.metadata_store = metadata_store

        # Retrieval parameters
        retrieval_config = config.get("retrieval", {})
        self.dense_candidates = retrieval_config.get("dense_candidates", 50)
        self.sparse_candidates = retrieval_config.get("sparse_candidates", 50)
        # Extra candidates drawn only from the filing(s) whose reporting
        # period the question names (see _period_scope).
        self.period_candidates = retrieval_config.get("period_scoped_candidates", 10)
        # Separate table lane (see _table_lane); False reproduces the
        # single-pool retrieval of 2026-10-20 for before/after comparisons.
        self.table_lane = retrieval_config.get("table_lane", False)
        self._fye_cache: dict[str, tuple[int, int] | None] = {}

        # Hybrid weights
        weights = retrieval_config.get("hybrid_weights", {})
        self.bm25_weight = weights.get("bm25", 0.4)
        self.dense_weight = weights.get("dense", 0.6)

        # BM25 index (built on demand)
        self.bm25_index: BM25Okapi | None = None
        self.bm25_chunk_ids: list[str] = []
        self.bm25_chunks: dict[str, any] = {}
        self._diag = threading.local()
        self.last_diagnostics = {}

    # One retriever serves every request; /query runs in FastAPI's
    # threadpool, so per-call diagnostics are thread-local - concurrent
    # requests must not report each other's mode or counts.
    @property
    def last_diagnostics(self) -> dict:
        return getattr(self._diag, "value", {})

    @last_diagnostics.setter
    def last_diagnostics(self, value: dict) -> None:
        self._diag.value = value

    @property
    def last_trace(self) -> dict:
        """Ranked candidate ids per stage of the last retrieve() call on
        this thread (dense, sparse, merged pool)."""
        return getattr(self._diag, "trace", {})

    def fiscal_year_end(self, ticker: str | None) -> tuple[int, int] | None:
        """(month, day) of the ticker's fiscal year end, from its most recent
        10-K period of report; None when no 10-K period is stored."""
        if not ticker:
            return None
        if ticker not in self._fye_cache:
            ends = [
                pe
                for _, dt, pe in self.metadata_store.get_filing_periods(ticker)
                if dt == "10-K"
            ]
            latest = max(ends) if ends else None
            self._fye_cache[ticker] = (latest.month, latest.day) if latest else None
        return self._fye_cache[ticker]

    def _period_scope(
        self, query: str, ticker: str | None
    ) -> tuple[list[str], set[str]] | None:
        """Filing chunks matching the fiscal period(s) the query names.

        An annual period selects that fiscal year's 10-K, a quarter its 10-Q
        (a fourth quarter, which has no 10-Q, the 10-K). Needs a ticker
        filter (the fiscal calendar is per company) and stored periods;
        otherwise None and retrieval is unscoped.
        """
        fye = self.fiscal_year_end(ticker)
        wanted = requested_periods(query, fye)
        if not wanted:
            return None
        rows = self.metadata_store.get_filing_periods(ticker)
        present = {fiscal_label(dt, pe, fye) for _, dt, pe in rows}
        # A period whose own filing is not in the corpus (e.g. the 2020-2023
        # gap) is reported as a prior-period comparative in the next year's
        # filing for the same period, so that filing is searched instead.
        want = set()
        labels = []
        for year, part in wanted:
            if (year, part) in present:
                want.add((year, part))
                labels.append(f"{part}{year}")
            elif (year + 1, part) in present:
                want.add((year + 1, part))
                labels.append(f"{part}{year} (via {part}{year + 1} comparatives)")
            else:
                labels.append(f"{part}{year}")
        ids = {cid for cid, dt, pe in rows if fiscal_label(dt, pe, fye) in want}
        return labels, ids

    def _is_table(self, chunk_id: str) -> bool:
        chunk = self.metadata_store.get_chunk(chunk_id)
        return bool(chunk and (chunk.section or "").endswith("_table"))

    def _targets(self, query: str, ticker: str | None) -> list[tuple]:
        """(fiscal_year, part, expected period end) for each period named."""
        fye = self.fiscal_year_end(ticker)
        return [
            (y, p, expected_period_end(y, p, fye))
            for y, p in requested_periods(query, fye)
        ]

    def _table_ids(self, ticker: str | None, query: str) -> set[str]:
        """Table chunks of the filings that can report the requested period:
        its own filing and the next one for the same period (comparative
        columns)."""
        fye = self.fiscal_year_end(ticker)
        want = set()
        for y, p, _ in self._targets(query, ticker):
            want |= {(y, p), (y + 1, p)}
        return {
            cid
            for cid, dt, pe, sec in self.metadata_store.get_filing_periods(
                ticker, with_section=True
            )
            if (sec or "").endswith("_table") and fiscal_label(dt, pe, fye) in want
        }

    def _table_lane(self, query, query_embedding, ticker, table_ids):
        """Best table chunks that have a column for the requested period.

        Tables are searched apart from prose (dense + BM25 restricted to the
        table chunks of the relevant filings), then kept only when one of
        their column headings denotes the requested period - explicit
        company (ticker filter), document (filing) and period matching.
        Tables whose columns could not be aligned assert no period and are
        left out of this lane.
        """
        if not table_ids:
            return [], {}
        k = self.period_candidates * 3
        dense = self._exclude_stale_dense_hits(
            self.vector_store.search(query_embedding, k=k, filter_ids=table_ids)
        )
        sparse = self._sparse_search(
            query, k=k, ticker_filter=ticker, restrict_ids=table_ids
        )
        targets = self._targets(query, ticker)

        def matched_columns(chunk) -> list[str] | None:
            header = next(
                (ln for ln in chunk.text.split("\n") if ln.startswith("Columns: ")), ""
            )
            if "NOT ALIGNED" in header:
                return None
            basis = _basis_of(chunk.text)
            if _basis_parts(basis) & _NON_REPORTED_BASES and not _asks_for_basis(
                query, basis
            ):
                return None  # e.g. pro forma figures for a question about results
            labels = [c.strip() for c in header[len("Columns: ") :].split(" | ")]
            return [
                c
                for c in labels
                if any(column_matches(c, y, p, end) for y, p, end in targets)
            ] or None

        kept, cols = [], {}
        for row in self._merge_results(dense, sparse):
            match = matched_columns(self.metadata_store.get_chunk(row[0]))
            if match:
                kept.append(row)
                cols[row[0]] = match
            if len(kept) >= self.period_candidates:
                break
        # The requested filings' explicitly reported tables (financial
        # statements) always compete, searched or not: a 30-candidate search
        # over a filing's tables can miss the income statement entirely and
        # leave only tax or segment tables (development runs, 2026-10-21).
        # Bounded; the row-wise reranker and the reported-basis preference
        # choose among them.
        fye = self.fiscal_year_end(ticker)
        own = {(y, p) for y, p, _ in targets}
        reported = []
        for cid in table_ids - set(cols):
            chunk = self.metadata_store.get_chunk(cid)
            if chunk is None or _basis_of(chunk.text) != "reported":
                continue
            match = matched_columns(chunk)
            if match:
                is_own = fiscal_label(chunk.doc_type, chunk.period_end, fye) in own
                reported.append((not is_own, chunk.chunk_index, cid, match))
        for _, _, cid, match in sorted(reported)[:_REPORTED_TABLES_CAP]:
            kept.append((cid, 0.0, 0.0, 0.0))
            cols[cid] = match
        return kept, cols

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lower-cased word/number tokens. Punctuation is not part of a token,
        so "2025?" in a question matches "2025" in a filing; decimal and
        thousands separators inside numbers are kept ("130.5", "36,000")."""
        return _TOKEN.findall(text.lower())

    def build_bm25_index(self):
        """Build BM25 index from all chunks in metadata store."""
        logger.info("Building BM25 index from metadata store...")
        self._fye_cache.clear()

        # Get all chunk IDs
        all_chunk_ids = self.metadata_store.get_all_chunk_ids()

        if not all_chunk_ids:
            logger.warning("No chunks found in metadata store")
            return

        # Retrieve all chunks
        chunks = []
        for chunk_id in all_chunk_ids:
            chunk = self.metadata_store.get_chunk(chunk_id)
            if chunk:
                chunks.append(chunk)

        tokenized_texts = [self._tokenize(chunk.text) for chunk in chunks]

        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_texts)
        self.bm25_chunk_ids = [chunk.chunk_id for chunk in chunks]

        # Store chunks for later retrieval
        self.bm25_chunks = {chunk.chunk_id: chunk for chunk in chunks}

        logger.info(f"Built BM25 index with {len(chunks)} chunks")

    def _dense_search(
        self,
        query_embedding: np.ndarray,
        k: int,
        ticker_filter: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> list[tuple[str, float]]:
        """Perform dense vector search using FAISS.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to retrieve
            ticker_filter: Optional ticker to filter by
            date_from: Optional start date filter
            date_to: Optional end date filter

        Returns:
            List of (chunk_id, score) tuples
        """
        # Build filter_ids set if filters are provided
        filter_ids = None
        if ticker_filter or date_from or date_to:
            # Query metadata store for matching chunks
            if ticker_filter and not date_from and not date_to:
                # Just ticker filter
                matching_chunks = self.metadata_store.get_chunks_by_ticker(
                    ticker_filter
                )
            elif date_from or date_to:
                # Date filter (with optional ticker)
                start_date = date_from if date_from else date.min
                end_date = date_to if date_to else date.max
                matching_chunks = self.metadata_store.get_chunks_by_date_range(
                    start=start_date, end=end_date, ticker=ticker_filter
                )
            else:
                matching_chunks = []

            filter_ids = {chunk.chunk_id for chunk in matching_chunks}

            # If no matches, return empty
            if not filter_ids:
                return []

        # Search vector store
        results = self.vector_store.search(query_embedding, k=k, filter_ids=filter_ids)
        kept = self._exclude_stale_dense_hits(results)
        self.last_diagnostics["dense_raw_hits"] = len(results)
        self.last_diagnostics["dense_excluded_unverified"] = len(results) - len(kept)
        return kept

    def _exclude_stale_dense_hits(
        self, results: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """Exclude a dense hit whose FAISS vector is not verifiably current.

        Two distinct cases, both excluded (audit correction, 2026-09-11
        follow-up pass):

        1. KNOWN MISMATCH - the vector's recorded content_hash disagrees
           with the chunk's current SQLite content_hash. FAISS holds only
           vectors, SQLite holds only text/metadata - if an interrupted
           content-replacement (or a rolled-back ingestion) leaves them
           representing different generations of the same chunk_id, the
           retriever would otherwise rank a candidate by a vector computed
           from one text and return a completely different (current) text
           as if it were the match.

        2. UNKNOWN PROVENANCE - the vector has no recorded content_hash at
           all (vector_hash == ""). This is not "no evidence of a
           problem," it's "no evidence of anything" - most commonly a
           legacy FAISS index migrated into the generation+manifest layout
           (VectorStore._migrate_legacy_if_present) whose entries predate
           content_hash tracking entirely, or a vector added via the
           identity-only VectorStore.add() fallback (no content_hashes
           argument - never used by the production ingestion path, see
           IngestionPipeline.store_chunks). An earlier version of this
           method only excluded case 1, silently trusting hashless vectors
           as if their similarity to the query were verified evidence.
           They are not: nothing has ever confirmed they were computed
           from the chunk's current (or any specific) text.

        BM25/sparse search is unaffected - it always re-reads current
        SQLite text directly (see build_bm25_index), so a chunk excluded
        here can still surface via BM25 on its own merits, just without
        borrowing similarity evidence from an unverified dense vector
        (_merge_results treats a chunk_id missing from dense_results as
        dense_score=0.0, not as a missing/failed lookup).

        Idempotent re-ingestion (a real content_hash + a freshly generated
        embedding, via IngestionPipeline.store_chunks) is the repair path
        for both cases: once the vector is replaced and its hash recorded,
        the chunk stops being excluded on its own. No data is deleted here
        - exclusion is a query-time filter, not a mutation.
        """
        filtered = []
        for chunk_id, score in results:
            vector_hash = self.vector_store.content_hashes.get(chunk_id, "")
            if not vector_hash:
                # Never log document text/content - chunk_id and the fact
                # of unknown provenance are safe, actionable diagnostics;
                # the text itself is not.
                logger.warning(
                    "Excluding chunk_id=%s from dense results: FAISS vector "
                    "has no known content_hash (unverified/legacy "
                    "provenance) - re-run ingestion for this source to "
                    "restore dense retrieval.",
                    chunk_id,
                )
                continue
            chunk = self.metadata_store.get_chunk(chunk_id)
            if (
                chunk is not None
                and chunk.content_hash
                and chunk.content_hash != vector_hash
            ):
                logger.warning(
                    "Excluding chunk_id=%s from dense results: FAISS "
                    "vector content_hash does not match current SQLite "
                    "content_hash (stale or interrupted replacement) - "
                    "re-run ingestion for this source to repair.",
                    chunk_id,
                )
                continue
            filtered.append((chunk_id, score))
        return filtered

    def _sparse_search(
        self,
        query: str,
        k: int,
        ticker_filter: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        restrict_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Perform sparse BM25 search.

        Args:
            query: Query text
            k: Number of results to retrieve
            ticker_filter: Optional ticker to filter by
            date_from: Optional start date filter
            date_to: Optional end date filter
            restrict_ids: Optional set of chunk ids to search within

        Returns:
            List of (chunk_id, score) tuples
        """
        # Build index if not already built
        if self.bm25_index is None:
            self.build_bm25_index()

        if self.bm25_index is None or not self.bm25_chunk_ids:
            return []

        tokenized_query = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)

        # Apply filters
        filtered_results = []
        for idx, score in enumerate(scores):
            chunk_id = self.bm25_chunk_ids[idx]
            chunk = self.bm25_chunks.get(chunk_id)

            if not chunk:
                continue
            if restrict_ids is not None and chunk_id not in restrict_ids:
                continue

            # Apply ticker filter
            if ticker_filter and chunk.ticker != ticker_filter:
                continue

            # Apply date filters
            if date_from and chunk.date < date_from:
                continue
            if date_to and chunk.date > date_to:
                continue

            filtered_results.append((chunk_id, float(score)))

        # Sort by score and take top k
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        return filtered_results[:k]

    def _merge_results(
        self,
        dense_results: list[tuple[str, float]],
        sparse_results: list[tuple[str, float]],
    ) -> list[tuple[str, float, float, float]]:
        """Merge dense and sparse results with weighted scoring.

        Args:
            dense_results: List of (chunk_id, dense_score) tuples
            sparse_results: List of (chunk_id, sparse_score) tuples

        Returns:
            List of (chunk_id, dense_score, sparse_score, hybrid_score) tuples
        """

        # Normalize scores to [0, 1] range
        def normalize_scores(results: list[tuple[str, float]]) -> dict[str, float]:
            """Min-max normalise retrieval scores to the [0, 1] range.

            Args:
                results: List of (chunk_id, raw_score) tuples from a retriever.

            Returns:
                Dict mapping chunk_id to its normalised score. Returns an empty
                dict for empty input; returns all-1.0 when all scores are equal
                (degenerate case, avoids division by zero).
            """
            if not results:
                return {}

            scores = [score for _, score in results]
            min_score = min(scores)
            max_score = max(scores)

            # Avoid division by zero
            if max_score == min_score:
                return {chunk_id: 1.0 for chunk_id, _ in results}

            return {
                chunk_id: (score - min_score) / (max_score - min_score)
                for chunk_id, score in results
            }

        # Normalize both sets of scores
        dense_normalized = normalize_scores(dense_results)
        sparse_normalized = normalize_scores(sparse_results)

        # Combine all unique chunk IDs
        all_chunk_ids = set(dense_normalized.keys()) | set(sparse_normalized.keys())

        # Calculate hybrid scores
        merged = []
        for chunk_id in all_chunk_ids:
            dense_score = dense_normalized.get(chunk_id, 0.0)
            sparse_score = sparse_normalized.get(chunk_id, 0.0)
            hybrid_score = (
                self.dense_weight * dense_score + self.bm25_weight * sparse_score
            )
            merged.append((chunk_id, dense_score, sparse_score, hybrid_score))

        # Sort by hybrid score
        merged.sort(key=lambda x: x[3], reverse=True)
        return merged

    def retrieve(
        self,
        query: str,
        ticker: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks using hybrid search.

        Args:
            query: Query text
            ticker: Optional ticker filter
            date_from: Optional start date filter
            date_to: Optional end date filter
            top_k: Number of results to return (default from config)

        Returns:
            List of RetrievedChunk objects sorted by hybrid score
        """
        if top_k is None:
            top_k = self.config.get("retrieval", {}).get("rerank_candidates", 20)

        logger.info(f"Retrieving chunks for query: '{query[:50]}...'")
        # Per-call diagnostics (2026-10): whether dense retrieval actually
        # contributed, so a degraded (sparse-only) run is visible instead of
        # silently presented as "hybrid".
        self.last_diagnostics = {"dense_raw_hits": 0, "dense_excluded_unverified": 0}

        # Embed query
        query_embedding = self.embedder.embed_texts([query])[0]

        # Perform dense search
        dense_results = self._dense_search(
            query_embedding,
            k=self.dense_candidates,
            ticker_filter=ticker,
            date_from=date_from,
            date_to=date_to,
        )
        logger.debug(f"Dense search returned {len(dense_results)} results")

        # Perform sparse search
        sparse_results = self._sparse_search(
            query,
            k=self.sparse_candidates,
            ticker_filter=ticker,
            date_from=date_from,
            date_to=date_to,
        )
        logger.debug(f"Sparse search returned {len(sparse_results)} results")

        # Merge results
        merged_results = self._merge_results(dense_results, sparse_results)
        logger.debug(f"Merged to {len(merged_results)} unique chunks")

        # Period-scoped pass (2026-10): when the question names a fiscal
        # period, also search only the filing(s) for that period and put
        # their best chunks into the reranking pool. Without it, a
        # neighbouring year's near-identical passage (same wording, other
        # numbers) could fill every slot - the period was never consulted.
        scoped: list[tuple[str, float, float, float]] = []
        scoped_dense_ids: set[str] = set()
        scope = None if date_from or date_to else self._period_scope(query, ticker)
        table_ids: set[str] = set()
        if self.table_lane and scope:
            table_ids = self._table_ids(ticker, query)
        if scope and scope[1]:
            # with the table lane on, the scoped pass is the prose lane
            ids = scope[1]
            if self.table_lane:
                ids = ids - {
                    cid
                    for cid, _, _, sec in self.metadata_store.get_filing_periods(
                        ticker, with_section=True
                    )
                    if (sec or "").endswith("_table")
                }
            s_dense = self._exclude_stale_dense_hits(
                self.vector_store.search(
                    query_embedding, k=self.period_candidates, filter_ids=ids
                )
            )
            s_sparse = self._sparse_search(
                query, k=self.period_candidates, ticker_filter=ticker, restrict_ids=ids
            )
            scoped = self._merge_results(s_dense, s_sparse)[: self.period_candidates]
            scoped_dense_ids = {cid for cid, _ in s_dense}

        lane_tables: list[tuple[str, float, float, float]] = []
        match_cols: dict[str, list[str]] = {}
        if self.table_lane and scope:
            # Prose lane: the scoped pass above searched prose only; tables
            # compete in their own lane (they used to take the scoped slots
            # useful prose needed) and leave the unscoped pool.
            merged_results = [x for x in merged_results if not self._is_table(x[0])]
            lane_tables, match_cols = self._table_lane(
                query, query_embedding, ticker, table_ids
            )

        # Convert to RetrievedChunk objects, collapsing exact-duplicate text
        # (the corpus holds ~11% repeated boilerplate across filings - e.g.
        # identical risk-factor paragraphs - which otherwise fills several
        # result slots with the same passage). The highest-ranked copy wins;
        # period-scoped candidates come first, so a duplicated passage is
        # kept as the copy from the requested period's filing.
        retrieved_chunks = []
        seen_ids: set[str] = set()
        seen_hashes: set[str] = set()
        duplicates = 0
        limit = top_k + len(scoped) + len(lane_tables)
        for chunk_id, dense_score, sparse_score, hybrid_score in (
            scoped + lane_tables + merged_results
        ):
            if len(retrieved_chunks) >= limit:
                break
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            chunk = self.metadata_store.get_chunk(chunk_id)
            if not chunk:
                logger.warning(f"Chunk {chunk_id} not found in metadata store")
                continue
            if chunk.content_hash and chunk.content_hash in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(chunk.content_hash)

            retrieved_chunk = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                ticker=chunk.ticker,
                text=chunk.text,
                doc_type=chunk.doc_type,
                source=chunk.source,
                section=chunk.section,
                date=chunk.date,
                url=chunk.url,
                dense_score=dense_score,
                sparse_score=sparse_score,
                hybrid_score=hybrid_score,
                final_score=None,  # Will be set by reranker
                accession_number=chunk.accession_number,
                attribution=chunk.attribution,
                content_hash=chunk.content_hash,
                period_end=chunk.period_end,
            )
            retrieved_chunk.header = source_header(
                retrieved_chunk, self.fiscal_year_end(chunk.ticker)
            )
            if self.table_lane and (chunk.section or "").endswith("_table"):
                retrieved_chunk.lane = "table"
                retrieved_chunk.basis = _basis_of(chunk.text)
                retrieved_chunk.match_columns = match_cols.get(chunk.chunk_id, [])
                if retrieved_chunk.match_columns:
                    retrieved_chunk.header += (
                        "; column(s) for the requested period: "
                        + " | ".join(retrieved_chunk.match_columns)
                    )
                    own = fiscal_label(
                        chunk.doc_type,
                        chunk.period_end,
                        self.fiscal_year_end(chunk.ticker),
                    )
                    if own and own not in {
                        (y, p) for y, p, _ in self._targets(query, ticker)
                    }:
                        retrieved_chunk.header += (
                            " (a comparative column: the figure describes the "
                            "requested period but comes from this later filing)"
                        )
            retrieved_chunks.append(retrieved_chunk)

        dense_ids = {cid for cid, _ in dense_results}
        self._diag.trace = {
            "dense": [cid for cid, _ in dense_results],
            "sparse": [cid for cid, _ in sparse_results],
            "candidates": [c.chunk_id for c in retrieved_chunks],
            "period_scoped": [cid for cid, *_ in scoped],
            "table_lane": [cid for cid, *_ in lane_tables],
        }
        self.last_diagnostics.update(
            {
                "dense_hits": len(dense_results),
                "sparse_hits": len(sparse_results),
                "overlap": len(dense_ids & {cid for cid, _ in sparse_results}),
                "results_from_dense": sum(
                    1
                    for c in retrieved_chunks
                    if c.chunk_id in dense_ids or c.chunk_id in scoped_dense_ids
                ),
                "duplicates_collapsed": duplicates,
                "mode": "hybrid" if dense_results else "sparse_only",
                "period_scope": (
                    {"periods": scope[0], "filing_chunks": len(scope[1])}
                    if scope
                    else None
                ),
            }
        )

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        return retrieved_chunks
