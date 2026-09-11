"""FAISS vector store module."""

import json
import logging
import re
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)

_MANIFEST_NAME = "manifest.json"
_LEGACY_INDEX_NAME = "index.faiss"
_LEGACY_IDS_NAME = "chunk_ids.json"
_GENERATION_RE = re.compile(r"^(?:index|chunk_ids)\.g(\d+)\.(?:faiss|json)$")


class VectorStore:
    """Manages FAISS index for dense retrieval.

    Persistence design (generation + manifest, 2026-09-11): each save()
    writes a complete, self-contained, generation-numbered pair of files
    (index.g{N}.faiss + chunk_ids.g{N}.json), validates that pair, and only
    then atomically activates it by replacing a single small manifest.json
    that names the current generation. Two independent os.replace() calls
    (one per file) are NOT an atomic pair - a crash between them can leave
    a mismatched index/metadata pair on disk if both files are "current" at
    once. This design instead keeps exactly one thing atomically switched
    (the manifest), so at every instant there is exactly one unambiguous,
    internally-consistent generation that's "current," regardless of when a
    crash happens relative to any of the writes.
    """

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
        # chunk_id -> content_hash of the vector currently stored at that
        # chunk_id's position. "" means unknown/untrusted provenance (a
        # migrated-from-legacy or otherwise hash-never-recorded entry).
        self.content_hashes: dict[str, str] = {}
        # -1 = nothing persisted yet (fresh store); save() always writes
        # generation (self.generation + 1).
        self.generation: int = -1

    def _create_index(self) -> faiss.Index:
        """Create a fresh FAISS index.

        Returns:
            FAISS IndexFlatIP for cosine similarity (inner product on normalized vectors)
        """
        return faiss.IndexFlatIP(self.dim)

    def _reset_fresh(self):
        self.index = self._create_index()
        self.chunk_ids = []
        self.content_hashes = {}
        self.generation = -1

    # --- generation-numbered file paths -----------------------------------

    def _manifest_path(self) -> Path:
        return self.index_path / _MANIFEST_NAME

    def _gen_index_path(self, generation: int) -> Path:
        return self.index_path / f"index.g{generation}.faiss"

    def _gen_ids_path(self, generation: int) -> Path:
        return self.index_path / f"chunk_ids.g{generation}.json"

    @staticmethod
    def _extract_generation(path: Path) -> int | None:
        m = _GENERATION_RE.match(path.name)
        return int(m.group(1)) if m else None

    # --- validation ---------------------------------------------------------

    def _validate_pair(
        self,
        index: faiss.Index,
        chunk_ids: list[str],
        manifest: dict | None = None,
        generation: int | str | None = None,
    ) -> bool:
        """Cross-check an (index, chunk_ids) pair for internal consistency
        and, if a manifest is given, against its recorded counts for this
        exact generation. Never "guesses" a pair is compatible."""
        if index.d != self.dim:
            logger.error(
                f"FAISS generation {generation}: index dim {index.d} != "
                f"expected {self.dim} - rejecting."
            )
            return False
        if index.ntotal != len(chunk_ids):
            logger.error(
                f"FAISS generation {generation}: vector count {index.ntotal} "
                f"!= chunk_id count {len(chunk_ids)} - rejecting."
            )
            return False
        if len(chunk_ids) != len(set(chunk_ids)):
            logger.error(
                f"FAISS generation {generation}: duplicate chunk_ids present "
                "- rejecting."
            )
            return False
        if manifest is not None and manifest.get("generation") == generation:
            expected_count = manifest.get("vector_count")
            if expected_count is not None and expected_count != index.ntotal:
                logger.error(
                    f"FAISS generation {generation}: manifest vector_count "
                    f"{expected_count} != actual {index.ntotal} - rejecting."
                )
                return False
            expected_dim = manifest.get("dim")
            if expected_dim is not None and expected_dim != index.d:
                logger.error(
                    f"FAISS generation {generation}: manifest dim "
                    f"{expected_dim} != actual {index.d} - rejecting."
                )
                return False
        return True

    def _load_generation_files(
        self, generation: int
    ) -> tuple[faiss.Index, list[str], dict[str, str]] | None:
        index_file = self._gen_index_path(generation)
        ids_file = self._gen_ids_path(generation)
        if not (index_file.exists() and ids_file.exists()):
            return None
        try:
            index = faiss.read_index(str(index_file))
            with open(ids_file, "r") as f:
                raw = json.load(f)
            chunk_ids = [entry[0] for entry in raw]
            content_hashes = {entry[0]: entry[1] for entry in raw}
            return index, chunk_ids, content_hashes
        except Exception as e:
            logger.error(f"Failed to read FAISS generation {generation}: {e}")
            return None

    def _try_activate_generation(
        self, generation: int, manifest: dict | None = None
    ) -> bool:
        loaded = self._load_generation_files(generation)
        if loaded is None:
            return False
        index, chunk_ids, content_hashes = loaded
        if not self._validate_pair(index, chunk_ids, manifest, generation):
            return False
        self.index = index
        self.chunk_ids = chunk_ids
        self.content_hashes = content_hashes
        self.generation = generation
        return True

    def _read_manifest(self) -> dict | None:
        path = self._manifest_path()
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                manifest = json.load(f)
            if not isinstance(manifest, dict) or "generation" not in manifest:
                logger.error(f"Malformed FAISS manifest at {path} - ignoring.")
                return None
            return manifest
        except Exception as e:
            logger.error(f"Failed to read FAISS manifest {path}: {e}")
            return None

    def _cleanup_old_generations(self, keep: set):
        """Delete generation files not in `keep` (typically {current,
        previous}). Safe: only ever removes files that are unreferenced by
        any generation the manifest could still point to."""
        keep = {g for g in keep if g is not None}
        for path in list(self.index_path.glob("index.g*.faiss")) + list(
            self.index_path.glob("chunk_ids.g*.json")
        ):
            gen = self._extract_generation(path)
            if gen is not None and gen not in keep:
                try:
                    path.unlink()
                except OSError:
                    pass

    def _migrate_legacy_if_present(self) -> bool:
        """Read a pre-generation-scheme flat index.faiss/chunk_ids.json
        pair, validate it, and migrate it into the generation+manifest
        layout as generation 0. Never guesses a legacy pair is compatible -
        an unreadable or internally-inconsistent legacy pair is rejected
        explicitly (logged), not silently accepted."""
        legacy_index = self.index_path / _LEGACY_INDEX_NAME
        legacy_ids = self.index_path / _LEGACY_IDS_NAME
        if not (legacy_index.exists() and legacy_ids.exists()):
            return False

        try:
            index = faiss.read_index(str(legacy_index))
            with open(legacy_ids, "r") as f:
                raw = json.load(f)
            if raw and isinstance(raw[0], str):
                chunk_ids = raw
                content_hashes = {}
            else:
                chunk_ids = [entry[0] for entry in raw]
                content_hashes = {entry[0]: entry[1] for entry in raw}
        except Exception as e:
            logger.error(
                f"Legacy FAISS index at {self.index_path} could not be read "
                f"({e}) - rejecting rather than guessing compatible; "
                "starting from a fresh, empty index. Re-run ingestion to "
                "rebuild this ticker's corpus."
            )
            return False

        if not self._validate_pair(index, chunk_ids, generation="legacy"):
            logger.error(
                f"Legacy FAISS index at {self.index_path} is internally "
                "inconsistent (dim/count mismatch) - rejecting rather than "
                "guessing compatible; starting from a fresh, empty index. "
                "Re-run ingestion to rebuild this ticker's corpus."
            )
            return False

        self.index = index
        self.chunk_ids = chunk_ids
        self.content_hashes = content_hashes
        self.generation = -1  # forces save() below to write generation 0

        self.save()
        if self.generation != -1:
            legacy_index.unlink(missing_ok=True)
            legacy_ids.unlink(missing_ok=True)
            logger.info(
                f"Migrated legacy FAISS index at {self.index_path} into the "
                f"generation-based layout (generation {self.generation})."
            )
            return True

        logger.error(
            f"Migration of legacy FAISS index at {self.index_path} failed to "
            "persist - leaving legacy files in place and starting from a "
            "fresh, empty index for this run."
        )
        self._reset_fresh()
        return False

    def load(self):
        """Load index from disk, or create fresh if none exists.

        Never loads an index file paired with metadata from a different
        generation, and never guesses a legacy or corrupt pair is
        compatible - see class docstring.
        """
        manifest = self._read_manifest()
        if manifest is not None:
            generation = manifest.get("generation")
            previous = manifest.get("previous_generation")

            if self._try_activate_generation(generation, manifest):
                logger.info(
                    f"Loaded {len(self.chunk_ids)} vectors from disk "
                    f"(generation {generation})"
                )
                self._cleanup_old_generations({generation, previous})
                return

            if previous is not None and self._try_activate_generation(previous):
                logger.error(
                    f"FAISS index generation {generation} failed validation; "
                    f"recovered previous generation {previous} instead. The "
                    "invalid generation's files are left on disk for "
                    "inspection - the next successful save() will overwrite "
                    "them."
                )
                return

            logger.error(
                f"FAISS index manifest at {self._manifest_path()} names "
                f"generation {generation} (previous {previous}), but no "
                "valid generation could be loaded - starting from a fresh, "
                "empty index rather than guessing. This project's local "
                "corpus is not automatically reproducible; re-run ingestion "
                "to rebuild."
            )
            self._reset_fresh()
            return

        # No manifest yet: either a brand new store, or on-disk files from
        # before this generation+manifest scheme existed.
        if self._migrate_legacy_if_present():
            return

        logger.info("No existing index found. Creating fresh index.")
        self._reset_fresh()

    def save(self):
        """Persist the index as a new generation, then atomically activate it.

        Writes a complete new generation's files, validates them by reading
        them back, and only then atomically replaces manifest.json to point
        at the new generation - the ONE atomic operation that changes what
        "current" means. If anything before that point fails, the
        previously active generation is untouched and remains current; if
        it fails after, the new generation is simply not yet referenced by
        anything and is inert. Re-running ingestion is the recovery path
        for whatever didn't make it to the newly active generation.
        """
        if self.index is None:
            logger.warning("No index to save")
            return

        try:
            new_generation = self.generation + 1
            index_file = self._gen_index_path(new_generation)
            ids_file = self._gen_ids_path(new_generation)
            tmp_index_file = index_file.with_name(index_file.name + ".tmp")
            tmp_ids_file = ids_file.with_name(ids_file.name + ".tmp")

            faiss.write_index(self.index, str(tmp_index_file))
            entries = [
                [cid, self.content_hashes.get(cid, "")] for cid in self.chunk_ids
            ]
            with open(tmp_ids_file, "w") as f:
                json.dump(entries, f)

            tmp_index_file.replace(index_file)
            tmp_ids_file.replace(ids_file)

            # Validate the newly written generation by reading it back
            # before activating it - defends against a write that
            # "succeeded" without raising but produced a corrupt/mismatched
            # pair (e.g. a partial disk failure faiss/json didn't surface).
            reloaded = self._load_generation_files(new_generation)
            if reloaded is None or not self._validate_pair(
                reloaded[0], reloaded[1], generation=new_generation
            ):
                logger.error(
                    f"Newly written FAISS generation {new_generation} failed "
                    "validation after write - NOT activating it. Previous "
                    f"generation ({self.generation}) remains active and "
                    "queryable."
                )
                index_file.unlink(missing_ok=True)
                ids_file.unlink(missing_ok=True)
                return

            new_manifest = {
                "generation": new_generation,
                "previous_generation": (
                    self.generation if self.generation >= 0 else None
                ),
                "vector_count": self.index.ntotal,
                "dim": self.dim,
            }
            tmp_manifest_path = self._manifest_path().with_name(_MANIFEST_NAME + ".tmp")
            with open(tmp_manifest_path, "w") as f:
                json.dump(new_manifest, f)
            # Single atomic activation: this is the one operation that
            # changes which generation is "current." Everything above can
            # fail or be interrupted without this ever being reached.
            tmp_manifest_path.replace(self._manifest_path())

            activated_generation = self.generation
            self.generation = new_generation
            self._cleanup_old_generations({new_generation, activated_generation})

            logger.info(
                f"Saved {len(self.chunk_ids)} vectors to disk "
                f"(generation {new_generation})"
            )
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return (embeddings / (norms + 1e-8)).astype(np.float32)  # avoid div-by-zero

    def _reconstruct_all(self) -> np.ndarray:
        """Return every vector currently in the index, in self.chunk_ids order.

        IndexFlatIP stores raw vectors (no quantization/inversion), so
        reconstruct_n is fully supported - including after a save/reload
        round-trip, verified empirically (not assumed) for this project's
        actual faiss version/index type before relying on it here.
        """
        n = self.index.ntotal
        if n == 0:
            return np.empty((0, self.dim), dtype=np.float32)
        return self.index.reconstruct_n(0, n)

    def _rebuild(self, ids: list[str], vectors: np.ndarray):
        """Replace the index in-place with exactly these ids/vectors, in order.

        IndexFlatIP (and FAISS flat indices generally) has no
        remove/update-by-position primitive, so an in-place replace isn't
        available - the architecture-compatible way to guarantee a stale
        vector stops being searchable is to rebuild the whole index from
        its current vectors (via reconstruct_n) minus/plus whatever
        changed. O(index size) per rebuild rather than O(1) - acceptable
        for this project's local, batch-ingestion FAISS index size, not for
        a large-scale/high-churn deployment (documented trade-off,
        2026-09-11).
        """
        # Build the replacement index in a local variable and only publish
        # it (both self.index and self.chunk_ids, together) once it's fully
        # built - if faiss.add() raises partway (bad shape, OOM), self.index
        # and self.chunk_ids must stay exactly as they were, not end up
        # split between an emptied-out new index and the old id list
        # (in-memory inconsistency that search()/reconstruct() would then
        # index out of bounds against).
        new_index = self._create_index()
        if len(vectors):
            new_index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self.index = new_index
        self.chunk_ids = ids

    def add(
        self,
        embeddings: np.ndarray,
        chunk_ids: list[str],
        content_hashes: dict[str, str] | None = None,
    ):
        """Add (or, if content changed, replace) embeddings in the index.

        Args:
            embeddings: Array of embeddings with shape (n, dim)
            chunk_ids: List of chunk IDs corresponding to embeddings
            content_hashes: Optional chunk_id -> content_hash map. For a
                chunk_id already in the index: a *different* stored hash
                means changed content (replace); a matching stored hash
                means unchanged (skip, idempotent re-ingestion); an
                unknown/blank stored hash means unknown provenance -
                treated as untrusted and always replaced, never certified
                by recording a hash without a matching regenerated
                embedding. A chunk_id with no entry in this map at all
                falls back to the original identity-only dedup (assumed
                unchanged, skipped, no hash recorded) - the pre-hash-
                tracking behavior for callers not using this feature.

                IMPORTANT: the identity-only fallback (omitting this
                argument, or omitting a specific chunk_id from it) does
                NOT certify that the stored vector matches any particular
                text - it only prevents duplicate rows. The ONE production
                ingestion path (IngestionPipeline.store_chunks) always
                passes a real content_hash for every chunk. Any caller
                that adds vectors without one is producing entries
                HybridRetriever will treat as unverified/unknown-provenance
                and exclude from dense search results at query time (see
                HybridRetriever._exclude_stale_dense_hits) - this mode
                exists only for isolated/low-level VectorStore usage (e.g.
                unit tests exercising indexing mechanics on their own), not
                for anything meant to be actually queried through the
                retrieval stack.
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() first.")

        if len(embeddings) == 0:
            return

        if len(embeddings) != len(chunk_ids):
            raise ValueError("Number of embeddings must match number of chunk_ids")

        content_hashes = content_hashes or {}
        existing_ids = set(self.chunk_ids)

        new_mask, changed_mask = [], []
        for cid in chunk_ids:
            if cid not in existing_ids:
                new_mask.append(True)
                changed_mask.append(False)
                continue
            new_mask.append(False)
            new_hash = content_hashes.get(cid, "")
            old_hash = self.content_hashes.get(cid, "")
            # A caller not tracking hashes at all (new_hash=="") keeps the
            # original, pre-hash identity-only dedup: an existing chunk_id
            # is assumed unchanged and skipped, exactly as before this
            # feature existed - no hash means nothing is being certified,
            # so there's nothing unsafe about it.
            #
            # A caller that DOES provide a real new_hash for an existing
            # chunk_id whose stored hash is unknown/blank (a hashless,
            # unknown-provenance legacy entry) must treat it as changed -
            # forcing a real replace with the embedding the caller is
            # providing now (which, via Embedder.embed_chunks' matching
            # fix, is guaranteed to be a freshly regenerated one for such a
            # chunk_id, never a stale cache hit). There is no baseline to
            # prove the existing vector matches anything, so it must never
            # be kept and silently "certified" by just recording a hash
            # next to it (audit correction, 2026-09-11 follow-up: this
            # previously backfilled the hash in place without ever
            # regenerating the embedding - exactly the unsafe behavior
            # being fixed here).
            #
            # A genuine, KNOWN hash mismatch (both sides non-empty and
            # different) also counts as changed, as before.
            changed_mask.append(
                bool(new_hash) and (not old_hash or new_hash != old_hash)
            )

        n_skipped = sum(
            1
            for is_new, is_changed in zip(new_mask, changed_mask)
            if not is_new and not is_changed
        )
        if n_skipped:
            logger.info(
                f"Skipping {n_skipped} chunk(s) already present with unchanged content"
            )

        to_add_mask = [
            is_new or is_changed for is_new, is_changed in zip(new_mask, changed_mask)
        ]
        if not any(to_add_mask):
            return

        add_embeddings = embeddings[np.array(to_add_mask, dtype=bool)]
        add_chunk_ids = [cid for cid, keep in zip(chunk_ids, to_add_mask) if keep]
        add_normalized = self._normalize(add_embeddings)

        changed_ids = {
            cid for cid, is_changed in zip(chunk_ids, changed_mask) if is_changed
        }

        if changed_ids:
            all_vectors = self._reconstruct_all()
            keep_mask = [cid not in changed_ids for cid in self.chunk_ids]
            surviving_ids = [
                cid for cid, keep in zip(self.chunk_ids, keep_mask) if keep
            ]
            surviving_vectors = (
                all_vectors[np.array(keep_mask, dtype=bool)]
                if self.chunk_ids
                else all_vectors
            )

            rebuilt_ids = surviving_ids + add_chunk_ids
            rebuilt_vectors = (
                np.vstack([surviving_vectors, add_normalized])
                if len(surviving_vectors)
                else add_normalized
            )

            self._rebuild(rebuilt_ids, rebuilt_vectors)
            logger.info(
                f"Rebuilt index to replace {len(changed_ids)} changed vector(s)"
            )
        else:
            self.index.add(add_normalized)
            self.chunk_ids.extend(add_chunk_ids)

        for cid in add_chunk_ids:
            new_hash = content_hashes.get(cid, "")
            if new_hash:
                self.content_hashes[cid] = new_hash

        # Save after adding
        self.save()

        logger.info(f"Added/replaced {len(add_chunk_ids)} vectors in index")

    def remove(self, chunk_ids: list[str]) -> int:
        """Remove vectors by chunk_id, rebuilding the index without them.

        Used for orphan cleanup: when a source document's re-chunking
        produces fewer/differently-numbered chunks than a previous
        ingestion, the excess old chunk_ids from that source must stop
        being searchable, not linger forever (audit finding, 2026-09-11 -
        no deletion path existed at all; only add()/upsert did).

        Args:
            chunk_ids: Chunk IDs to remove. IDs not present are ignored.

        Returns:
            Number of vectors actually removed.
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() first.")

        remove_set = set(chunk_ids)
        if not remove_set.intersection(self.chunk_ids):
            return 0

        all_vectors = self._reconstruct_all()
        keep_mask = [cid not in remove_set for cid in self.chunk_ids]
        surviving_ids = [cid for cid, keep in zip(self.chunk_ids, keep_mask) if keep]
        surviving_vectors = (
            all_vectors[np.array(keep_mask, dtype=bool)]
            if self.chunk_ids
            else all_vectors
        )
        n_removed = len(self.chunk_ids) - len(surviving_ids)

        self._rebuild(surviving_ids, surviving_vectors)
        surviving_set = set(surviving_ids)
        self.content_hashes = {
            cid: h for cid, h in self.content_hashes.items() if cid in surviving_set
        }
        self.save()

        logger.info(f"Removed {n_removed} vector(s) from index")
        return n_removed

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 20,
        filter_ids: set[str] | None = None,
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
