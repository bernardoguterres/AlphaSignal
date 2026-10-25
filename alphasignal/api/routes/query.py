"""Query endpoint for AlphaSignal RAG API."""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from openai import OpenAIError

from alphasignal.api.dependencies import (
    get_config,
    get_generator,
    get_metrics_collector,
    get_reranker,
    get_retriever,
)
from alphasignal.generation.support import assess_support
from alphasignal.api.schemas import Citation, QueryRequest, QueryResponse
from alphasignal.generation.generator import RAGGenerator
from alphasignal.monitoring.metrics import MetricsCollector
from alphasignal.retrieval.premise import retrieval_query
from alphasignal.retrieval.reranker import CrossEncoderReranker, compose_context
from alphasignal.retrieval.retriever import requested_bases
from alphasignal.retrieval.retriever import HybridRetriever

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=QueryResponse)
def query(
    request: QueryRequest,
    retriever: HybridRetriever = Depends(get_retriever),
    reranker: CrossEncoderReranker = Depends(get_reranker),
    generator: RAGGenerator = Depends(get_generator),
    config: dict = Depends(get_config),
    metrics_collector: MetricsCollector = Depends(get_metrics_collector),
) -> QueryResponse:
    """Execute a RAG query against the financial corpus.

    Args:
        request: Query request with query text and optional filters
        retriever: Hybrid retriever instance
        reranker: Cross-encoder reranker instance
        generator: RAG generator instance
        config: Configuration dictionary
        metrics_collector: Metrics collector instance

    Returns:
        QueryResponse with answer and citations
    """
    start_time = time.time()

    try:
        logger.info(f"Processing query: '{request.query[:50]}...'")

        # Deliberate decision (2026-09-11): request.ticker_filter is NOT run
        # through validate_ticker_in_config()/the 404 allowlist, unlike
        # /sentiment/{ticker} and /ingest/{ticker}. Those paths identify a
        # resource by ticker - an unknown one is a 404 ("no such
        # resource"). ticker_filter is a search *filter* over a shared
        # /query resource, not the resource identifier itself - an unknown
        # or misspelled value is not an error, it just narrows the search
        # to zero matches, exactly like any other filter (date_from/to)
        # that happens to exclude everything. This keeps /query's error
        # semantics uniform: only a 5xx for a genuine failure, otherwise
        # always 200, with an empty citations list signaling "no matches"
        # (see the no-results branch just below).
        # Step 1: Retrieve candidates
        # Search with the premise-neutral form of the question (see
        # retrieval/premise.py); the model answers the question as asked.
        search_query = retrieval_query(request.query)
        retrieved_chunks = retriever.retrieve(
            query=search_query,
            ticker=request.ticker_filter,
            date_from=request.date_from,
            date_to=request.date_to,
            top_k=20,  # Get more candidates for reranking
        )

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")

        # Handle no results
        if not retrieved_chunks:
            latency_ms = int((time.time() - start_time) * 1000)
            metrics_collector.record_query(latency_ms)
            return QueryResponse(
                query=request.query,
                answer="No relevant information found in the knowledge base for this query.",
                citations=[],
                latency_ms=latency_ms,
                retrieval_scores=[],
                model_used=config.get("generation", {}).get("model", "gpt-5.6-luna"),
                retrieval=dict(getattr(retriever, "last_diagnostics", {}) or {}),
            )

        # Step 2: Rerank
        # Rerank the whole pool (sorted), then keep top_k - the full order is
        # only reported when include_trace is set.
        reranked_all = reranker.rerank(
            query=search_query, chunks=retrieved_chunks, top_k=None
        )
        reranked_chunks = compose_context(
            reranked_all,
            request.top_k,
            table_slots=config.get("retrieval", {}).get("table_slots", 1),
            prefer_bases=requested_bases(request.query),
        )

        logger.info(f"Reranked to top {len(reranked_chunks)} chunks")

        # Step 3: Generate answer
        generation_result = generator.generate(
            query=request.query, chunks=reranked_chunks
        )

        logger.info(
            f"Generated answer with {len(generation_result.cited_chunks)} citations"
        )

        # Step 4: Build response
        # Convert cited chunks to Citation objects
        citations = []
        for chunk in generation_result.cited_chunks:
            citation = Citation(
                chunk_id=chunk.chunk_id,
                ticker=chunk.ticker,
                source=chunk.source,
                date=chunk.date,
                excerpt=chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
                relevance_score=(
                    chunk.final_score
                    if chunk.final_score is not None
                    else chunk.hybrid_score
                ),
                doc_type=chunk.doc_type,
                url=chunk.url,
                accession_number=getattr(chunk, "accession_number", None),
                attribution=getattr(chunk, "attribution", None) or None,
            )
            citations.append(citation)

        # Get retrieval scores from reranked chunks
        retrieval_scores = [
            chunk.final_score if chunk.final_score is not None else chunk.hybrid_score
            for chunk in reranked_chunks
        ]

        latency_ms = int((time.time() - start_time) * 1000)
        metrics_collector.record_query(latency_ms)

        return QueryResponse(
            query=request.query,
            answer=generation_result.answer,
            citations=citations,
            latency_ms=latency_ms,
            retrieval_scores=retrieval_scores,
            model_used=config.get("generation", {}).get("model", "gpt-5.6-luna"),
            retrieval=dict(getattr(retriever, "last_diagnostics", {}) or {}),
            retrieved_chunk_ids=[chunk.chunk_id for chunk in reranked_chunks],
            trace=(
                {
                    **getattr(retriever, "last_trace", {}),
                    "reranked": [chunk.chunk_id for chunk in reranked_all],
                }
                if request.include_trace
                else None
            ),
            citation_support=assess_support(
                generation_result.answer,
                [c.text for c in generation_result.cited_chunks],
            ),
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        metrics_collector.record_error()
        if isinstance(e, OpenAIError):
            raise HTTPException(
                status_code=503, detail="AI service temporarily unavailable"
            )
        raise
