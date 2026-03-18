"""Generation module for RAG answer synthesis and sentiment extraction."""

from dataclasses import dataclass

from alphasignal.retrieval import RetrievedChunk


@dataclass
class GenerationResult:
    """Result from RAG answer generation."""

    answer: str
    cited_chunks: list[RetrievedChunk]
    prompt_tokens: int
    completion_tokens: int
    model: str


@dataclass
class SentimentResult:
    """Result from sentiment extraction."""

    score: float  # -1.0 (very negative) to 1.0 (very positive)
    confidence: float  # 0.0 to 1.0
    key_positive: list[str]
    key_negative: list[str]
    summary: str


__all__ = ["GenerationResult", "SentimentResult"]
