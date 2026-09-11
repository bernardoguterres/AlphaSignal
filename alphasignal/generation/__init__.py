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
    # False when this result came from a provider/parsing fallback (JSON
    # parse failure, non-finite score/confidence, or an API exception)
    # rather than a genuine model prediction. A reliable score of 0.0 is a
    # real neutral prediction; an unreliable score of 0.0 is a placeholder
    # and must never be presented as the same thing (audit finding,
    # 2026-09-11: both cases previously produced an identical
    # SentimentResult(score=0.0, confidence=0.0), making degraded
    # extraction indistinguishable from genuine neutral sentiment).
    reliable: bool = True


__all__ = ["GenerationResult", "SentimentResult"]
