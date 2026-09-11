"""Embedding cache module."""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Persistent cache mapping chunk_id -> (embedding vector, content_hash).

    Stored as a plain .npy vector array plus a .json sidecar list of
    [chunk_id, content_hash] pairs (index i in the sidecar corresponds to
    row i of the array). Audit finding: this used to be a pickle file, which
    deserializes arbitrary objects and is a code-execution risk if the cache
    file is ever replaced by an untrusted one - .npy (with allow_pickle=False)
    and JSON carry no such risk.

    content_hash lets the cache distinguish "this chunk_id was already
    embedded" from "this chunk_id's *current text* was already embedded":
    chunk_id encodes source position, not content, so re-ingesting a filing
    whose text changed under an unchanged chunk_id must miss the cache
    instead of silently returning a stale vector (audit finding, 2026-09-11).

    A stored content_hash of "" (the legacy on-disk format, or any pre-hash
    cache entry) means UNKNOWN PROVENANCE, not "known unchanged" - there is
    no evidence the cached vector was produced from any particular text.
    Such an entry is always treated as untrusted: get()/get_many() report it
    as a miss unconditionally, forcing a fresh embed on next touch, and the
    hash is recorded only alongside that newly generated embedding (audit
    correction, 2026-09-11 follow-up - an earlier version of this class let
    a hashless entry serve as a hit indefinitely and separately "backfilled"
    a hash onto it without ever regenerating the embedding, which certified
    an embedding of unknown provenance as matching the current text with no
    evidence of that).
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
        # chunk_id -> (embedding, content_hash)
        self.cache: dict[str, tuple[np.ndarray, str]] = {}

        # Load existing cache if it exists
        if self._vectors_path.exists() and self._ids_path.exists():
            try:
                vectors = np.load(self._vectors_path, allow_pickle=False)
                with open(self._ids_path, "r") as f:
                    raw_entries = json.load(f)

                if len(raw_entries) != len(vectors):
                    raise ValueError(
                        f"chunk_ids count ({len(raw_entries)}) does not match "
                        f"vector count ({len(vectors)})"
                    )

                self.cache = {}
                for i, entry in enumerate(raw_entries):
                    # Legacy format: a bare chunk_id string, no hash on record.
                    if isinstance(entry, str):
                        chunk_id, content_hash = entry, ""
                    else:
                        chunk_id, content_hash = entry[0], entry[1]
                    self.cache[chunk_id] = (vectors[i], content_hash)
                logger.info(f"Loaded {len(self.cache)} embeddings from cache")
            except Exception as e:
                logger.warning(f"Failed to load cache from {cache_path}: {e}")
                self.cache = {}

    def get(self, chunk_id: str, content_hash: str | None = None) -> np.ndarray | None:
        """Get cached embedding for a chunk, if its content still matches.

        Args:
            chunk_id: Chunk identifier
            content_hash: Current content fingerprint of the chunk.

        Returns:
            Embedding vector, or None if not cached, of unknown/untrusted
            provenance (stored hash == ""), or stale (stored hash differs
            from `content_hash`).

        A hashless (unknown-provenance) entry is ALWAYS a miss, regardless
        of whether `content_hash` is passed - there is no baseline to prove
        it matches anything, so it must never be served as a hit (audit
        correction, 2026-09-11 follow-up: previously any freshness check
        was skipped entirely when `content_hash` was omitted, which let a
        hashless entry match unconditionally).
        """
        entry = self.cache.get(chunk_id)
        if entry is None:
            return None
        embedding, cached_hash = entry
        if not cached_hash:
            return None
        if content_hash is not None and cached_hash != content_hash:
            return None
        return embedding

    def get_stored_hash(self, chunk_id: str) -> str:
        """Return the content_hash currently stored for a cached chunk_id.

        Returns "" both when the chunk_id isn't cached at all and when it's
        cached with an unknown/untrusted hash - callers that need to tell
        those apart should check membership separately.
        """
        entry = self.cache.get(chunk_id)
        return entry[1] if entry is not None else ""

    def set(self, chunk_id: str, embedding: np.ndarray, content_hash: str = ""):
        """Cache an embedding.

        Args:
            chunk_id: Chunk identifier
            embedding: Embedding vector
            content_hash: Content fingerprint the embedding was computed
                from, used to detect staleness on future lookups.
        """
        self.cache[chunk_id] = (embedding, content_hash)

    def get_many(
        self, chunk_ids: list[str] | dict[str, str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Get multiple cached embeddings.

        Args:
            chunk_ids: Either a plain list of chunk identifiers, or a dict
                mapping chunk_id -> content_hash. Either way, a hashless
                (unknown-provenance) cached entry is always a miss - see
                get().

        Returns:
            Tuple of (cached_embeddings, uncached_chunk_ids)
        """
        if isinstance(chunk_ids, dict):
            items = list(chunk_ids.items())
        else:
            items = [(cid, None) for cid in chunk_ids]

        cached = {}
        uncached = []

        for chunk_id, content_hash in items:
            embedding = self.get(chunk_id, content_hash)
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
                np.stack([self.cache[cid][0] for cid in chunk_ids])
                if chunk_ids
                else np.empty((0, 0), dtype=np.float32)
            )
            entries = [[cid, self.cache[cid][1]] for cid in chunk_ids]

            tmp_vectors_path = self._vectors_path.with_name(
                self._vectors_path.name + ".tmp"
            )
            tmp_ids_path = self._ids_path.with_name(self._ids_path.name + ".tmp")

            with open(tmp_vectors_path, "wb") as f:
                np.save(f, vectors, allow_pickle=False)
            with open(tmp_ids_path, "w") as f:
                json.dump(entries, f)

            tmp_vectors_path.replace(self._vectors_path)
            tmp_ids_path.replace(self._ids_path)

            logger.info(f"Saved {len(self.cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save cache to {self.cache_path}: {e}")

    def __len__(self) -> int:
        """Return number of cached embeddings."""
        return len(self.cache)
