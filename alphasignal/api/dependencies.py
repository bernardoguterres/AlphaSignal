"""Dependency injection for FastAPI routes."""

from fastapi import Request

from alphasignal.api.state import AppState
from alphasignal.generation.generator import RAGGenerator
from alphasignal.generation.sentiment import SentimentExtractor
from alphasignal.ingestion.pipeline import IngestionPipeline
from alphasignal.monitoring.metrics import MetricsCollector
from alphasignal.retrieval.evaluator import RetrievalEvaluator
from alphasignal.retrieval.reranker import CrossEncoderReranker
from alphasignal.retrieval.retriever import HybridRetriever
from alphasignal.store.metadata_store import MetadataStore
from alphasignal.store.vector_store import VectorStore


def get_app_state(request: Request) -> AppState:
    """Get application state from request.

    Args:
        request: FastAPI request object

    Returns:
        AppState instance
    """
    return request.app.state.app_state


def get_config(request: Request) -> dict:
    """Get configuration dictionary.

    Args:
        request: FastAPI request object

    Returns:
        Configuration dictionary
    """
    state = get_app_state(request)
    return state.config


def get_pipeline(request: Request) -> IngestionPipeline:
    """Get ingestion pipeline.

    Args:
        request: FastAPI request object

    Returns:
        IngestionPipeline instance
    """
    state = get_app_state(request)
    return state.pipeline


def get_retriever(request: Request) -> HybridRetriever:
    """Get hybrid retriever.

    Args:
        request: FastAPI request object

    Returns:
        HybridRetriever instance
    """
    state = get_app_state(request)
    return state.retriever


def get_reranker(request: Request) -> CrossEncoderReranker:
    """Get cross-encoder reranker.

    Args:
        request: FastAPI request object

    Returns:
        CrossEncoderReranker instance
    """
    state = get_app_state(request)
    return state.reranker


def get_generator(request: Request) -> RAGGenerator:
    """Get RAG generator.

    Args:
        request: FastAPI request object

    Returns:
        RAGGenerator instance
    """
    state = get_app_state(request)
    return state.generator


def get_sentiment_extractor(request: Request) -> SentimentExtractor:
    """Get sentiment extractor.

    Args:
        request: FastAPI request object

    Returns:
        SentimentExtractor instance
    """
    state = get_app_state(request)
    return state.sentiment_extractor


def get_metadata_store(request: Request) -> MetadataStore:
    """Get metadata store.

    Args:
        request: FastAPI request object

    Returns:
        MetadataStore instance
    """
    state = get_app_state(request)
    return state.pipeline.metadata_store


def get_vector_store(request: Request) -> VectorStore:
    """Get vector store.

    Args:
        request: FastAPI request object

    Returns:
        VectorStore instance
    """
    state = get_app_state(request)
    return state.pipeline.vector_store


def get_evaluator(request: Request) -> RetrievalEvaluator | None:
    """Get retrieval evaluator (optional).

    Args:
        request: FastAPI request object

    Returns:
        RetrievalEvaluator instance or None if not available
    """
    state = get_app_state(request)
    return state.evaluator


def get_metrics_collector(request: Request) -> MetricsCollector:
    """Get metrics collector.

    Args:
        request: FastAPI request object

    Returns:
        MetricsCollector instance
    """
    state = get_app_state(request)
    return state.metrics_collector
