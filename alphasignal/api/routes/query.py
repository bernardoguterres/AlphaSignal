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
from alphasignal.api.schemas import Citation, QueryRequest, QueryResponse
from alphasignal.generation.generator import RAGGenerator
from alphasignal.monitoring.metrics import MetricsCollector
from alphasignal.retrieval.reranker import CrossEncoderReranker
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

        # Step 1: Retrieve candidates
        ticker_filter = request.ticker_filter[0] if request.ticker_filter else None
        retrieved_chunks = retriever.retrieve(
            query=request.query,
            ticker=ticker_filter,
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
                model_used=config.get("generation", {}).get("model", "gpt-4o-mini"),
            )

        # Step 2: Rerank
        reranked_chunks = reranker.rerank(
            query=request.query, chunks=retrieved_chunks, top_k=request.top_k
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
            model_used=config.get("generation", {}).get("model", "gpt-4o-mini"),
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        metrics_collector.record_error()
        if isinstance(e, OpenAIError):
            raise HTTPException(
                status_code=503, detail="AI service temporarily unavailable"
            )
        raise
