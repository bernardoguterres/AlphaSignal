"""Hybrid retrieval module combining dense and sparse search."""

import logging
from datetime import date

import numpy as np
from rank_bm25 import BM25Okapi

from alphasignal.embeddings.embedder import Embedder
from alphasignal.retrieval import RetrievedChunk
from alphasignal.store.metadata_store import MetadataStore
from alphasignal.store.vector_store import VectorStore

logger = logging.getLogger(__name__)


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

        # Hybrid weights
        weights = retrieval_config.get("hybrid_weights", {})
        self.bm25_weight = weights.get("bm25", 0.4)
        self.dense_weight = weights.get("dense", 0.6)

        # BM25 index (built on demand)
        self.bm25_index: BM25Okapi | None = None
        self.bm25_chunk_ids: list[str] = []
        self.bm25_chunks: dict[str, any] = {}

    def build_bm25_index(self):
        """Build BM25 index from all chunks in metadata store."""
        logger.info("Building BM25 index from metadata store...")

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

        # Tokenize texts (simple whitespace tokenization for BM25)
        tokenized_texts = [chunk.text.lower().split() for chunk in chunks]

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

        return self._exclude_stale_dense_hits(results)

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
    ) -> list[tuple[str, float]]:
        """Perform sparse BM25 search.

        Args:
            query: Query text
            k: Number of results to retrieve
            ticker_filter: Optional ticker to filter by
            date_from: Optional start date filter
            date_to: Optional end date filter

        Returns:
            List of (chunk_id, score) tuples
        """
        # Build index if not already built
        if self.bm25_index is None:
            self.build_bm25_index()

        if self.bm25_index is None or not self.bm25_chunk_ids:
            return []

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)

        # Apply filters
        filtered_results = []
        for idx, score in enumerate(scores):
            chunk_id = self.bm25_chunk_ids[idx]
            chunk = self.bm25_chunks.get(chunk_id)

            if not chunk:
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

        # Take top k
        top_results = merged_results[:top_k]

        # Convert to RetrievedChunk objects
        retrieved_chunks = []
        for chunk_id, dense_score, sparse_score, hybrid_score in top_results:
            # Get chunk metadata
            chunk = self.metadata_store.get_chunk(chunk_id)
            if not chunk:
                logger.warning(f"Chunk {chunk_id} not found in metadata store")
                continue

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
            )
            retrieved_chunks.append(retrieved_chunk)

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        return retrieved_chunks
