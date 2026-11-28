"""Replay interrupted vector upserts and orphan cleanup."""

import json
import logging
from uuid import uuid4

import numpy as np

from alphasignal.store.metadata_store import MetadataStore
from alphasignal.store.vector_store import VectorStore

logger = logging.getLogger(__name__)


def pending_batch_path(vector_store: VectorStore):
    return vector_store.index_path / "pending-ingestion.npz"


def stage_vector_batch(
    vector_store: VectorStore,
    ids,
    hashes,
    vectors,
    *,
    orphan_hashes: dict[str, str] | None = None,
) -> str | None:
    """Publish a single-file journal before committing SQLite metadata.

    The caller holds the same writer lock used by recovery and publication.
    """
    if vectors.shape != (len(ids), vector_store.dim) or vectors.dtype.kind not in "fiu":
        raise ValueError("Invalid ingestion embedding dimensions or type")
    path = pending_batch_path(vector_store)
    temporary = path.with_suffix(".tmp")
    replacement_id = uuid4().hex if orphan_hashes else None
    replacement = (
        {"replacement_id": replacement_id, "orphan_hashes": json.dumps(orphan_hashes)}
        if replacement_id
        else {}
    )
    with temporary.open("wb") as output:
        np.savez(
            output,
            ids=ids,
            hashes=[hashes[cid] for cid in ids],
            vectors=vectors,
            **replacement,
        )
    temporary.replace(path)
    return replacement_id


def recover_vector_batch(vector_store: VectorStore, metadata: MetadataStore) -> int:
    """Replay committed vectors and their planned retirements without embedding calls."""
    path = pending_batch_path(vector_store)
    with vector_store.writer():
        if not path.exists():
            return 0
        replacement_id = None
        orphan_hashes = {}
        with np.load(path, allow_pickle=False) as pending:
            ids = pending["ids"].tolist()
            hashes = pending["hashes"].tolist()
            vectors = pending["vectors"]
            if "replacement_id" in pending or "orphan_hashes" in pending:
                replacement_id = pending["replacement_id"].item()
                orphan_hashes = _validate_orphan_hashes(
                    json.loads(pending["orphan_hashes"].item())
                )
                if (
                    not isinstance(replacement_id, str)
                    or not replacement_id
                    or not ids
                    or not orphan_hashes
                    or set(ids).intersection(orphan_hashes)
                ):
                    raise ValueError("Invalid pending replacement batch")
        if vectors.shape != (len(ids), vector_store.dim) or len(hashes) != len(ids):
            raise ValueError("Invalid pending ingestion batch dimensions")

        # Hashes alone cannot prove a replacement committed: metadata such
        # as attribution can change while all replacement texts stay equal.
        if replacement_id and not metadata.has_replacement_commit(replacement_id):
            path.unlink()
            return 0

        current = metadata.get_content_hashes(ids)
        matching = [
            i
            for i, (cid, digest) in enumerate(zip(ids, hashes))
            if digest and current.get(cid) == digest
        ]
        if replacement_id and len(matching) != len(ids):
            # The database was changed or restored. Do not retire any old
            # data based on an incomplete or superseded replacement.
            logger.warning("Discarding replacement batch with mismatching metadata")
            metadata.delete_chunks([], replacement_id=replacement_id)
            path.unlink()
            return 0

        eligible_ids = [ids[i] for i in matching]
        if matching and not vector_store.add(
            vectors[matching], eligible_ids, {ids[i]: hashes[i] for i in matching}
        ):
            raise RuntimeError("Could not publish pending ingestion vectors")
        if replacement_id:
            _finish_orphan_cleanup(
                vector_store, metadata, orphan_hashes, replacement_id=replacement_id
            )
        path.unlink()
        logger.info(
            "Replayed %d vector(s) from a pending ingestion batch", len(matching)
        )
        return len(matching)


def _pending_cleanup_path(vector_store: VectorStore):
    return vector_store.index_path / "pending-orphan-cleanup.json"


def get_orphan_hashes(metadata: MetadataStore, ids: set[str]) -> dict[str, str]:
    """Capture the exact versions that this ingestion plans to retire."""
    hashes = metadata.get_content_hashes(sorted(ids))
    if set(hashes) != ids or not all(hashes.values()):
        raise ValueError("Cannot stage orphan cleanup without metadata content hashes")
    return hashes


def stage_orphan_cleanup(
    vector_store: VectorStore, metadata: MetadataStore, ids: set[str]
) -> None:
    """Record exact retirements before removing any data, under the writer lock."""
    hashes = get_orphan_hashes(metadata, ids)
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
        expected = _validate_orphan_hashes(json.loads(path.read_text(encoding="utf-8")))
        _finish_orphan_cleanup(vector_store, metadata, expected)
        path.unlink()


def _validate_orphan_hashes(expected) -> dict[str, str]:
    if not isinstance(expected, dict) or not all(
        chunk_id and isinstance(digest, str) and digest
        for chunk_id, digest in expected.items()
    ):
        raise ValueError("Invalid pending orphan cleanup batch")
    return expected


def _finish_orphan_cleanup(
    vector_store: VectorStore,
    metadata: MetadataStore,
    expected: dict[str, str],
    *,
    replacement_id: str | None = None,
) -> None:
    current = metadata.get_content_hashes(list(expected))
    # Neither missing nor changed metadata authorizes more removals.
    ids = [cid for cid, digest in expected.items() if current.get(cid) == digest]
    if ids:
        vector_store.remove(ids)
    if replacement_id is not None:
        metadata.delete_chunks(ids, replacement_id=replacement_id)
    elif ids:
        metadata.delete_chunks(ids)
    logger.info("Removed %d orphaned chunk(s) from both stores", len(ids))
