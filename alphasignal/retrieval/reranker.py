"""Reranking module using cross-encoder models."""

import logging

from sentence_transformers import CrossEncoder

from alphasignal.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks retrieved chunks using cross-encoder for better relevance."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace cross-encoder model name
        """
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Cross-encoder model loaded successfully")

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Rerank chunks by computing cross-encoder relevance scores.

        Args:
            query: Query text
            chunks: List of retrieved chunks to rerank
            top_k: Number of top results to return (None = return all)

        Returns:
            Reranked list of chunks sorted by final_score
        """
        if not chunks:
            return []

        logger.info(f"Reranking {len(chunks)} chunks")

        # Prepare query-chunk pairs for cross-encoder
        pairs = [(query, chunk.text) for chunk in chunks]

        # Compute relevance scores
        scores = self.model.predict(pairs)

        # Set final_score on each chunk
        for chunk, score in zip(chunks, scores):
            chunk.final_score = float(score)

        # Sort by final_score
        reranked = sorted(chunks, key=lambda x: x.final_score or 0.0, reverse=True)

        # Return top k if specified
        if top_k is not None:
            reranked = reranked[:top_k]

        logger.info(f"Reranking complete, returning {len(reranked)} chunks")
        return reranked
