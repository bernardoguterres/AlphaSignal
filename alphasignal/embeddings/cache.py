"""Embedding cache module."""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Persistent cache mapping chunk_id → embedding vector.

    Stored as a plain .npy vector array plus a .json sidecar list of
    chunk_ids (index i in the sidecar corresponds to row i of the array).
    Audit finding: this used to be a pickle file, which deserializes
    arbitrary objects and is a code-execution risk if the cache file is
    ever replaced by an untrusted one - .npy (with allow_pickle=False) and
    JSON carry no such risk.
    """

    def __init__(self, cache_path: str):
        """Initialize embedding cache.

        Args:
            cache_path: Base path for the cache; the on-disk vector array
                and id sidecar are derived from it (any existing suffix is
                replaced).
        """
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._vectors_path = self.cache_path.with_suffix(".npy")
        self._ids_path = self.cache_path.with_suffix(".json")
        self.cache: dict[str, np.ndarray] = {}

        # Load existing cache if it exists
        if self._vectors_path.exists() and self._ids_path.exists():
            try:
                vectors = np.load(self._vectors_path, allow_pickle=False)
                with open(self._ids_path, "r") as f:
                    chunk_ids = json.load(f)

                if len(chunk_ids) != len(vectors):
                    raise ValueError(
                        f"chunk_ids count ({len(chunk_ids)}) does not match "
                        f"vector count ({len(vectors)})"
                    )

                self.cache = {
                    chunk_id: vectors[i] for i, chunk_id in enumerate(chunk_ids)
                }
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
        """Persist cache to disk.

        Writes to temp files first, then renames into place, so a crash
        mid-write can't leave a truncated/corrupt cache. The temp vector
        file is written via an explicit file handle rather than a bare
        path, since np.save() appends ".npy" to any path that doesn't
        already end with it - passing a ".npy.tmp" path would otherwise
        silently produce "....npy.tmp.npy" instead of the intended file.
        """
        try:
            chunk_ids = list(self.cache.keys())
            vectors = (
                np.stack([self.cache[cid] for cid in chunk_ids])
                if chunk_ids
                else np.empty((0, 0), dtype=np.float32)
            )

            tmp_vectors_path = self._vectors_path.with_name(
                self._vectors_path.name + ".tmp"
            )
            tmp_ids_path = self._ids_path.with_name(self._ids_path.name + ".tmp")

            with open(tmp_vectors_path, "wb") as f:
                np.save(f, vectors, allow_pickle=False)
            with open(tmp_ids_path, "w") as f:
                json.dump(chunk_ids, f)

            tmp_vectors_path.replace(self._vectors_path)
            tmp_ids_path.replace(self._ids_path)

            logger.info(f"Saved {len(self.cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save cache to {self.cache_path}: {e}")

    def __len__(self) -> int:
        """Return number of cached embeddings."""
        return len(self.cache)
