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

        # chunk_id -> content_hash, so the cache can detect a chunk whose
        # source text changed under an unchanged chunk_id and re-embed it
        # instead of returning a stale vector (audit finding, 2026-09-11).
        content_hashes = {chunk.chunk_id: chunk.content_hash for chunk in chunks}

        # Check cache. A hashless (unknown-provenance) cached entry is
        # ALWAYS a miss here (see EmbeddingCache.get) - there is no
        # evidence it was produced from this chunk's current text, so it
        # is never backfilled with a hash in place; it must be regenerated
        # from the current text like any other cache miss, and the hash is
        # recorded only alongside that freshly generated embedding below
        # (audit correction, 2026-09-11 follow-up).
        cached_embeddings, uncached_ids = self.cache.get_many(content_hashes)

        logger.info(
            f"Embedding chunks: {len(cached_embeddings)} from cache, "
            f"{len(uncached_ids)} to embed"
        )

        # Embed uncached chunks (includes any hashless/untrusted legacy
        # entries, which the cache lookup above already treated as misses)
        if uncached_ids:
            # Get texts for uncached chunks
            uncached_chunks = [c for c in chunks if c.chunk_id in uncached_ids]
            uncached_texts = [chunk.text for chunk in uncached_chunks]

            # Embed - if this raises, no cache.set() below runs for these
            # chunks, so any previous (untrusted) entry is left exactly as
            # it was: not deleted, but also never marked as validated or
            # served as a hit going forward.
            new_embeddings = self.embed_texts(uncached_texts)

            # Add to cache - hash stored only alongside the embedding that
            # was actually just generated from the current text.
            for chunk, embedding in zip(uncached_chunks, new_embeddings):
                self.cache.set(chunk.chunk_id, embedding, chunk.content_hash)
                cached_embeddings[chunk.chunk_id] = embedding

            self.cache.save()

        return cached_embeddings
