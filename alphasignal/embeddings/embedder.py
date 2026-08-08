"""Embedding generation module."""

import logging
import time

import numpy as np
from openai import OpenAI

from alphasignal.embeddings.cache import EmbeddingCache
from alphasignal.ingestion import Chunk

logger = logging.getLogger(__name__)


class Embedder:
    """Generates embeddings for text chunks using OpenAI API."""

    def __init__(self, config: dict, cache: EmbeddingCache):
        """Initialize embedder.

        Args:
            config: Configuration dictionary containing embeddings settings
            cache: EmbeddingCache instance
        """
        self.config = config
        self.cache = cache

        # Get embedding config
        embedding_config = config.get("embeddings", {})
        self.model = embedding_config.get("model", "text-embedding-3-small")
        self.batch_size = embedding_config.get("batch_size", 100)
        self.max_retries = embedding_config.get("max_retries", 3)
        self.retry_delay = embedding_config.get("retry_delay", 1.0)

        # Initialize OpenAI client (reads OPENAI_API_KEY from env)
        self.client = OpenAI()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts using OpenAI API.

        Args:
            texts: List of text strings to embed

        Returns:
            Array of embeddings with shape (len(texts), 1536)
        """
        if not texts:
            return np.array([])

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            # Retry logic with exponential backoff
            for attempt in range(self.max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=self.model, input=batch
                    )

                    # Extract embeddings
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break

                except Exception as e:
                    if attempt < self.max_retries - 1:
                        # Exponential backoff
                        delay = self.retry_delay * (2**attempt)
                        logger.warning(
                            f"Embedding API error (attempt {attempt + 1}/{self.max_retries}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Failed to embed batch after {self.max_retries} attempts: {e}"
                        )
                        raise

        return np.array(all_embeddings, dtype=np.float32)

    def embed_chunks(self, chunks: list[Chunk]) -> dict[str, np.ndarray]:
        """Embed chunks with caching support.

        Args:
            chunks: List of Chunk objects to embed

        Returns:
            Dictionary mapping chunk_id → embedding for all chunks
        """
        if not chunks:
            return {}

        chunk_ids = [chunk.chunk_id for chunk in chunks]

        # Check cache
        cached_embeddings, uncached_ids = self.cache.get_many(chunk_ids)

        logger.info(
            f"Embedding chunks: {len(cached_embeddings)} from cache, "
            f"{len(uncached_ids)} to embed"
        )

        # Embed uncached chunks
        if uncached_ids:
            # Get texts for uncached chunks
            uncached_chunks = [c for c in chunks if c.chunk_id in uncached_ids]
            uncached_texts = [chunk.text for chunk in uncached_chunks]

            # Embed
            new_embeddings = self.embed_texts(uncached_texts)

            # Add to cache
            for chunk_id, embedding in zip(uncached_ids, new_embeddings):
                self.cache.set(chunk_id, embedding)
                cached_embeddings[chunk_id] = embedding

            # Save cache
            self.cache.save()

        return cached_embeddings
