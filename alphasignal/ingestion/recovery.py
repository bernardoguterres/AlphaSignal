"""Replay unpublished vector upserts from a local ingestion batch."""

import logging

import numpy as np

from alphasignal.store.metadata_store import MetadataStore
from alphasignal.store.vector_store import VectorStore

logger = logging.getLogger(__name__)


def pending_batch_path(vector_store: VectorStore):
    return vector_store.index_path / "pending-ingestion.npz"


def stage_vector_batch(vector_store: VectorStore, ids, hashes, vectors):
    """Publish a single-file journal before committing SQLite metadata.

    The caller holds the same writer lock used by recovery and publication.
    """
    if vectors.shape != (len(ids), vector_store.dim) or vectors.dtype.kind not in "fiu":
        raise ValueError("Invalid ingestion embedding dimensions or type")
    path = pending_batch_path(vector_store)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as output:
        np.savez(output, ids=ids, hashes=[hashes[cid] for cid in ids], vectors=vectors)
    temporary.replace(path)


def recover_vector_batch(vector_store: VectorStore, metadata: MetadataStore) -> int:
    """Replay only vectors whose metadata commit is visible, without embedding calls."""
    path = pending_batch_path(vector_store)
    with vector_store.writer():
        if not path.exists():
            return 0
        with np.load(path, allow_pickle=False) as pending:
            ids = pending["ids"].tolist()
            hashes = pending["hashes"].tolist()
            vectors = pending["vectors"]
        if vectors.shape != (len(ids), vector_store.dim) or len(hashes) != len(ids):
            raise ValueError("Invalid pending ingestion batch dimensions")

        current = metadata.get_content_hashes()
        matching = [
            i
            for i, (cid, digest) in enumerate(zip(ids, hashes))
            if digest and current.get(cid) == digest
        ]
        eligible_ids = [ids[i] for i in matching]
        if matching and not vector_store.add(
            vectors[matching], eligible_ids, {ids[i]: hashes[i] for i in matching}
        ):
            raise RuntimeError("Could not publish pending ingestion vectors")
        path.unlink()
        logger.info(
            "Replayed %d vector(s) from a pending ingestion batch", len(matching)
        )
        return len(matching)
