"""Application state dataclass for dependency injection."""

from dataclasses import dataclass

from alphasignal.generation.generator import RAGGenerator
from alphasignal.generation.sentiment import SentimentExtractor
from alphasignal.ingestion.pipeline import IngestionPipeline
from alphasignal.monitoring.metrics import MetricsCollector
from alphasignal.retrieval.evaluator import RetrievalEvaluator
from alphasignal.retrieval.reranker import CrossEncoderReranker
from alphasignal.retrieval.retriever import HybridRetriever


@dataclass
class AppState:
    """Application state container for dependency injection."""

    config: dict
    pipeline: IngestionPipeline
    retriever: HybridRetriever
    reranker: CrossEncoderReranker
    generator: RAGGenerator
    sentiment_extractor: SentimentExtractor
    evaluator: RetrievalEvaluator | None
    metrics_collector: MetricsCollector
    start_time: float
