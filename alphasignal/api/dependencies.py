"""Dependency injection for FastAPI routes."""

import os
import secrets

from fastapi import Header, HTTPException, Request

from alphasignal.api.state import AppState
from alphasignal.generation.generator import RAGGenerator
from alphasignal.generation.sentiment import SentimentExtractor
from alphasignal.ingestion.pipeline import IngestionPipeline
from alphasignal.monitoring.metrics import MetricsCollector
from alphasignal.retrieval.reranker import CrossEncoderReranker
from alphasignal.retrieval.retriever import HybridRetriever
from alphasignal.store.metadata_store import MetadataStore


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


def get_metrics_collector(request: Request) -> MetricsCollector:
    """Get metrics collector.

    Args:
        request: FastAPI request object

    Returns:
        MetricsCollector instance
    """
    state = get_app_state(request)
    return state.metrics_collector


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Reject requests without a valid X-API-Key when auth is configured.

    Auth is enabled by setting the ALPHASIGNAL_API_KEY env var on the server;
    unset = open (local dev). AlphaLive's client already sends this header
    when its own ALPHASIGNAL_API_KEY env var is set - use the same value on
    both services. Applied to every router except /health (Railway's
    healthcheck can't send headers). Without this, a public deploy lets
    anyone spend the OpenAI budget via /query and /ingest.
    """
    expected = os.getenv("ALPHASIGNAL_API_KEY", "")
    if not expected:
        return  # auth not configured - open access
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
