"""FAISS vector store module."""

import json
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages FAISS index for dense retrieval."""

    def __init__(self, index_path: str, dim: int = 1536):
        """Initialize vector store.

        Args:
            index_path: Path to store index files
            dim: Embedding dimension (default 1536 - both ada-002 and text-embedding-3-small)
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.dim = dim
        self.index: faiss.Index | None = None
        self.chunk_ids: list[str] = []

    def _create_index(self) -> faiss.Index:
        """Create a fresh FAISS index.

        Returns:
            FAISS IndexFlatIP for cosine similarity (inner product on normalized vectors)
        """
        return faiss.IndexFlatIP(self.dim)

    def load(self):
        """Load index from disk or create fresh if not exists."""
        index_file = self.index_path / "index.faiss"
        ids_file = self.index_path / "chunk_ids.json"

        if index_file.exists() and ids_file.exists():
            try:
                # Load FAISS index
                self.index = faiss.read_index(str(index_file))

                # Load chunk IDs
                with open(ids_file, 'r') as f:
                    self.chunk_ids = json.load(f)

                logger.info(f"Loaded {len(self.chunk_ids)} vectors from disk")
            except Exception as e:
                logger.error(f"Failed to load index: {e}. Creating fresh index.")
                self.index = self._create_index()
                self.chunk_ids = []
        else:
            logger.info("No existing index found. Creating fresh index.")
            self.index = self._create_index()
            self.chunk_ids = []

    def save(self):
        """Persist index to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return

        try:
            index_file = self.index_path / "index.faiss"
            ids_file = self.index_path / "chunk_ids.json"

            # Save FAISS index
            faiss.write_index(self.index, str(index_file))

            # Save chunk IDs
            with open(ids_file, 'w') as f:
                json.dump(self.chunk_ids, f)

            logger.info(f"Saved {len(self.chunk_ids)} vectors to disk")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def add(self, embeddings: np.ndarray, chunk_ids: list[str]):
        """Add embeddings to the index.

        Args:
            embeddings: Array of embeddings with shape (n, dim)
            chunk_ids: List of chunk IDs corresponding to embeddings
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() first.")

        if len(embeddings) == 0:
            return

        if len(embeddings) != len(chunk_ids):
            raise ValueError("Number of embeddings must match number of chunk_ids")

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)  # Avoid division by zero

        # Add to index
        self.index.add(normalized.astype(np.float32))
        self.chunk_ids.extend(chunk_ids)

        # Save after adding
        self.save()

        logger.info(f"Added {len(chunk_ids)} vectors to index")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 20,
        filter_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        """Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_ids: Optional set of chunk_ids to filter results

        Returns:
            List of (chunk_id, score) tuples, sorted by score descending
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() first.")

        if len(self.chunk_ids) == 0:
            return []

        # Normalize query embedding
        norm = np.linalg.norm(query_embedding)
        normalized_query = query_embedding / (norm + 1e-8)

        # Search for more candidates than needed to allow for filtering
        search_k = min(k * 3, len(self.chunk_ids)) if filter_ids else k
        search_k = max(search_k, 1)

        # Reshape for FAISS
        query_vector = normalized_query.reshape(1, -1).astype(np.float32)

        # Search
        scores, indices = self.index.search(query_vector, search_k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue

            chunk_id = self.chunk_ids[idx]

            # Apply filter if provided
            if filter_ids and chunk_id not in filter_ids:
                continue

            # Ensure score is in [0, 1] range (should be for normalized vectors)
            score = float(np.clip(score, 0.0, 1.0))

            results.append((chunk_id, score))

            # Stop once we have enough results
            if len(results) >= k:
                break

        return results

    def __len__(self) -> int:
        """Return number of vectors in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal
