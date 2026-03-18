"""Embedding cache module."""

import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Persistent cache mapping chunk_id → embedding vector."""

    def __init__(self, cache_path: str):
        """Initialize embedding cache.

        Args:
            cache_path: Path to cache file
        """
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, np.ndarray] = {}

        # Load existing cache if it exists
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cache)} embeddings from cache")
            except Exception as e:
                logger.warning(f"Failed to load cache from {cache_path}: {e}")
                self.cache = {}

    def get(self, chunk_id: str) -> np.ndarray | None:
        """Get cached embedding for a chunk.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Embedding vector or None if not cached
        """
        return self.cache.get(chunk_id)

    def set(self, chunk_id: str, embedding: np.ndarray):
        """Cache an embedding.

        Args:
            chunk_id: Chunk identifier
            embedding: Embedding vector
        """
        self.cache[chunk_id] = embedding

    def get_many(self, chunk_ids: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
        """Get multiple cached embeddings.

        Args:
            chunk_ids: List of chunk identifiers

        Returns:
            Tuple of (cached_embeddings, uncached_chunk_ids)
        """
        cached = {}
        uncached = []

        for chunk_id in chunk_ids:
            embedding = self.cache.get(chunk_id)
            if embedding is not None:
                cached[chunk_id] = embedding
            else:
                uncached.append(chunk_id)

        return cached, uncached

    def save(self):
        """Persist cache to disk."""
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.info(f"Saved {len(self.cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save cache to {self.cache_path}: {e}")

    def __len__(self) -> int:
        """Return number of cached embeddings."""
        return len(self.cache)
