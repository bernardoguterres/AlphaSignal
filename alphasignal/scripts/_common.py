"""Shared bootstrap helpers for standalone scripts."""

from pathlib import Path

import yaml

from alphasignal.embeddings.cache import EmbeddingCache
from alphasignal.embeddings.embedder import Embedder
from alphasignal.store.metadata_store import MetadataStore
from alphasignal.store.vector_store import VectorStore


def load_config(project_root: Path) -> dict:
    """Load config.yaml from the project root.

    Args:
        project_root: Path to the AlphaSignal project root

    Returns:
        Parsed configuration dictionary
    """
    config_path = project_root / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_storage_components(
    config: dict,
) -> tuple[VectorStore, MetadataStore, Embedder]:
    """Build the vector store, metadata store, and embedder for a standalone script.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (vector_store, metadata_store, embedder)
    """
    storage_config = config.get("storage", {})

    vector_store = VectorStore(
        storage_config.get("faiss_index_path", "data/faiss_index"), dim=1536
    )
    vector_store.load()

    metadata_store = MetadataStore(
        storage_config.get("sqlite_db_path", "data/metadata.db")
    )

    embedding_cache = EmbeddingCache(
        f"{storage_config.get('embeddings_cache_path', 'data/embeddings_cache')}/cache.pkl"
    )
    embedder = Embedder(config, embedding_cache)

    return vector_store, metadata_store, embedder
