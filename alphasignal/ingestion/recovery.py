"""Replay interrupted vector upserts and orphan cleanup."""

import json
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


def _pending_cleanup_path(vector_store: VectorStore):
    return vector_store.index_path / "pending-orphan-cleanup.json"


def stage_orphan_cleanup(
    vector_store: VectorStore, metadata: MetadataStore, ids: set[str]
) -> None:
    """Record exact retirements before removing any data, under the writer lock."""
    hashes = metadata.get_content_hashes(sorted(ids))
    if set(hashes) != ids or not all(hashes.values()):
        raise ValueError("Cannot stage orphan cleanup without metadata content hashes")
    path = _pending_cleanup_path(vector_store)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(hashes), encoding="utf-8")
    temporary.replace(path)


def recover_orphan_cleanup(vector_store: VectorStore, metadata: MetadataStore) -> None:
    """Finish a recorded retirement even if one store has already applied it."""
    path = _pending_cleanup_path(vector_store)
    with vector_store.writer():
        if not path.exists():
            return
        expected = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(expected, dict) or not all(
            chunk_id and isinstance(digest, str) and digest
            for chunk_id, digest in expected.items()
        ):
            raise ValueError("Invalid pending orphan cleanup batch")
        current = metadata.get_content_hashes(list(expected))
        ids = [cid for cid, digest in expected.items() if current.get(cid) == digest]
        # A missing row means deletion already completed for a paired store.
        # A changed row may come from maintenance or a restored metadata store.
        # Neither can authorize another removal.
        if ids:
            vector_store.remove(ids)
            metadata.delete_chunks(ids)
        path.unlink()
        logger.info("Removed %d orphaned chunk(s) from both stores", len(ids))
