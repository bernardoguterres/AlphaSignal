"""Generation/manifest-based FAISS persistence tests (2026-09-11 correction
pass, objective 2): failure-injection coverage for VectorStore's atomic
activation design (write new generation -> validate -> single manifest
replace), proving a crash at any point leaves exactly one valid,
self-consistent generation loadable - never a mismatched index/metadata
pair, and never a silently-guessed-compatible corrupt one.
"""

import json
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np

from alphasignal.store.vector_store import VectorStore

DIM = 8


def _vec(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i % DIM] = 1.0
    return v


def _manifest(index_path: Path) -> dict:
    with open(index_path / "manifest.json") as f:
        return json.load(f)


class TestGenerationActivationFailures:
    def test_failure_before_either_file_written(self, tmp_path):
        """A failure before the new generation's index file is even
        written must leave nothing new on disk and the store unchanged."""
        index_path = tmp_path / "index"
        store = VectorStore(str(index_path), dim=DIM)
        store.load()
        store.add(np.array([_vec(0)]), ["a"], {"a": "h1"})
        gen0 = store.generation

        with patch(
            "alphasignal.store.vector_store.faiss.write_index",
            side_effect=OSError("disk full"),
        ):
            store.add(np.array([_vec(1)]), ["a"], {"a": "h2"})  # triggers save()

        assert store.generation == gen0
        assert _manifest(index_path)["generation"] == gen0
        # No orphaned next-generation files.
        assert not (index_path / f"index.g{gen0 + 1}.faiss").exists()

        # Store must still be fully usable/reloadable afterward.
        store2 = VectorStore(str(index_path), dim=DIM)
        store2.load()
        assert len(store2) == 1
        assert store2.chunk_ids == ["a"]

    def test_failure_after_index_written_before_metadata(self, tmp_path):
        """New generation's index.faiss lands, but chunk_ids.json fails -
        the new generation is incomplete and must never be activated."""
        index_path = tmp_path / "index"
        store = VectorStore(str(index_path), dim=DIM)
        store.load()
        store.add(np.array([_vec(0)]), ["a"], {"a": "h1"})
        gen0 = store.generation

        real_open = open

        def flaky_open(path, mode="r", *args, **kwargs):
            if "chunk_ids.g" in str(path) and "w" in mode:
                raise OSError("disk full")
            return real_open(path, mode, *args, **kwargs)

        with patch("builtins.open", side_effect=flaky_open):
            store.add(np.array([_vec(1)]), ["a"], {"a": "h2"})

        assert store.generation == gen0
        assert _manifest(index_path)["generation"] == gen0
        # The orphaned index file for the failed attempt may exist, but
        # must not be referenced by the manifest and must not corrupt a
        # fresh load.
        store2 = VectorStore(str(index_path), dim=DIM)
        store2.load()
        assert len(store2) == 1
        assert store2.generation == gen0

    def test_failure_after_metadata_before_manifest_activation(self, tmp_path):
        """Both new-generation files land completely and validate, but the
        manifest write itself fails - the new generation exists on disk
        but is inert (never activated) until a future successful save."""
        index_path = tmp_path / "index"
        store = VectorStore(str(index_path), dim=DIM)
        store.load()
        store.add(np.array([_vec(0)]), ["a"], {"a": "h1"})
        gen0 = store.generation

        real_open = open

        def flaky_open(path, mode="r", *args, **kwargs):
            if "manifest.json.tmp" in str(path):
                raise OSError("disk full")
            return real_open(path, mode, *args, **kwargs)

        with patch("builtins.open", side_effect=flaky_open):
            store.add(np.array([_vec(1)]), ["a"], {"a": "h2"})

        assert store.generation == gen0
        assert _manifest(index_path)["generation"] == gen0
        # The new generation's files DID land (unlike the metadata-failure
        # case above) but must be ignored since the manifest never
        # switched.
        assert (index_path / f"index.g{gen0 + 1}.faiss").exists()

        store2 = VectorStore(str(index_path), dim=DIM)
        store2.load()
        assert len(store2) == 1
        assert store2.generation == gen0
        assert store2.content_hashes["a"] == "h1", "old generation, not the new one"

    def test_failure_during_manifest_activation_itself(self, tmp_path):
        """A failure during the manifest's final os.replace (the one
        genuinely atomic step) must leave the manifest pointing at
        whatever it pointed at before - either the old generation
        (replace never happened) - proving the design never claims a
        half-finished activation as complete."""
        index_path = tmp_path / "index"
        store = VectorStore(str(index_path), dim=DIM)
        store.load()
        store.add(np.array([_vec(0)]), ["a"], {"a": "h1"})
        gen0 = store.generation
        manifest_before = _manifest(index_path)

        with patch.object(
            Path, "replace", side_effect=OSError("simulated interruption")
        ):
            store.add(np.array([_vec(1)]), ["a"], {"a": "h2"})

        # Path.replace is used for index/ids/manifest all three; patching
        # it globally means NONE of the three renames happen, which is a
        # valid (if pessimistic) simulation of "interrupted before/during
        # activation" - the manifest.json content on disk must be exactly
        # what it was before this save() attempt.
        assert _manifest(index_path) == manifest_before
        assert store.generation == gen0

    def test_restart_with_incomplete_new_generation_and_valid_previous(self, tmp_path):
        """Simulates exactly the state test_failure_after_metadata_before_
        manifest_activation leaves behind, but from a FRESH process (new
        VectorStore instance / restart) - must recover the previous, valid
        generation cleanly, not the dangling new one."""
        index_path = tmp_path / "index"
        store = VectorStore(str(index_path), dim=DIM)
        store.load()
        store.add(np.array([_vec(0)]), ["a"], {"a": "h1"})
        gen0 = store.generation

        # Manually write a complete, valid-looking but unactivated next
        # generation (as if save() got this far and then the process died
        # right before the manifest replace).
        ghost_index = faiss.IndexFlatIP(DIM)
        ghost_index.add(np.array([_vec(9)], dtype=np.float32))
        faiss.write_index(ghost_index, str(index_path / f"index.g{gen0 + 1}.faiss"))
        with open(index_path / f"chunk_ids.g{gen0 + 1}.json", "w") as f:
            json.dump([["a", "h2"]], f)
        # Manifest is deliberately NOT updated - this is the crash point.

        restarted = VectorStore(str(index_path), dim=DIM)
        restarted.load()

        assert restarted.generation == gen0
        assert restarted.content_hashes["a"] == "h1"
        assert len(restarted) == 1

    def test_restart_with_mismatched_index_and_metadata_no_valid_generation(
        self, tmp_path
    ):
        """The manifest points at a generation whose index/metadata pair
        is mismatched (count mismatch) and there is no previous_generation
        to fall back to - must start fresh rather than load a mismatched
        pair."""
        index_path = tmp_path / "index"
        index_path.mkdir()

        bad_index = faiss.IndexFlatIP(DIM)
        bad_index.add(np.array([_vec(0), _vec(1)], dtype=np.float32))  # 2 vectors
        faiss.write_index(bad_index, str(index_path / "index.g0.faiss"))
        with open(index_path / "chunk_ids.g0.json", "w") as f:
            json.dump([["a", "h1"]], f)  # only 1 id - count mismatch
        with open(index_path / "manifest.json", "w") as f:
            json.dump(
                {
                    "generation": 0,
                    "previous_generation": None,
                    "vector_count": 2,
                    "dim": DIM,
                },
                f,
            )

        store = VectorStore(str(index_path), dim=DIM)
        store.load()

        assert len(store) == 0
        assert store.chunk_ids == []
        # Must still be fully usable afterward.
        store.add(np.array([_vec(0)]), ["x"])
        assert len(store) == 1

    def test_corrupted_metadata_file_rejected_not_guessed(self, tmp_path):
        index_path = tmp_path / "index"
        index_path.mkdir()

        good_index = faiss.IndexFlatIP(DIM)
        good_index.add(np.array([_vec(0)], dtype=np.float32))
        faiss.write_index(good_index, str(index_path / "index.g0.faiss"))
        (index_path / "chunk_ids.g0.json").write_text("not valid json{{{")
        with open(index_path / "manifest.json", "w") as f:
            json.dump(
                {
                    "generation": 0,
                    "previous_generation": None,
                    "vector_count": 1,
                    "dim": DIM,
                },
                f,
            )

        store = VectorStore(str(index_path), dim=DIM)
        store.load()

        assert len(store) == 0
        assert store.chunk_ids == []

    def test_count_mismatched_metadata_against_manifest_rejected(self, tmp_path):
        """The index/ids pair is internally self-consistent, but disagrees
        with the manifest's recorded vector_count - a subtler corruption
        the pairwise (index vs ids) check alone wouldn't catch."""
        index_path = tmp_path / "index"
        index_path.mkdir()

        idx = faiss.IndexFlatIP(DIM)
        idx.add(np.array([_vec(0), _vec(1)], dtype=np.float32))
        faiss.write_index(idx, str(index_path / "index.g0.faiss"))
        with open(index_path / "chunk_ids.g0.json", "w") as f:
            json.dump([["a", "h1"], ["b", "h2"]], f)  # self-consistent: 2 and 2
        with open(index_path / "manifest.json", "w") as f:
            json.dump(
                {
                    "generation": 0,
                    "previous_generation": None,
                    "vector_count": 99,  # manifest disagrees
                    "dim": DIM,
                },
                f,
            )

        store = VectorStore(str(index_path), dim=DIM)
        store.load()

        assert len(store) == 0, "manifest/file disagreement must be rejected"

    def test_never_activates_index_paired_with_wrong_generation_metadata(
        self, tmp_path
    ):
        """Direct proof of the core requirement: an index file from one
        generation must never be loaded together with metadata from
        another. Simulated by manually cross-wiring generation 0's index
        with generation 1's metadata under generation 0's expected paths."""
        index_path = tmp_path / "index"
        store = VectorStore(str(index_path), dim=DIM)
        store.load()
        store.add(np.array([_vec(0)]), ["a"], {"a": "h1"})
        store.add(np.array([_vec(5)]), ["a"], {"a": "h2"})  # generation 1
        gen1 = store.generation
        assert gen1 >= 1

        # Cross-wire: point generation 1's ids file content at generation
        # 0's old content by overwriting it directly (simulating disk
        # corruption/a wrong file landing at that path).
        with open(index_path / f"chunk_ids.g{gen1}.json", "w") as f:
            json.dump([["a", "h1"], ["b", "hX"]], f)  # 2 ids, index has 1 vector

        restarted = VectorStore(str(index_path), dim=DIM)
        restarted.load()

        # Count mismatch (1 vector vs 2 ids) must be caught - generation 1
        # rejected, falls back to generation 0.
        assert restarted.generation == gen1 - 1 or len(restarted) == 0


class TestGenerationCleanup:
    def test_only_current_and_previous_generation_files_retained(self, tmp_path):
        index_path = tmp_path / "index"
        store = VectorStore(str(index_path), dim=DIM)
        store.load()
        for i in range(5):
            store.add(np.array([_vec(i)]), ["a"], {"a": f"h{i}"})

        remaining = {
            store._extract_generation(p)
            for p in list(index_path.glob("index.g*.faiss"))
        }
        assert remaining == {store.generation, store.generation - 1}

    def test_legacy_flat_files_migrated_to_generation_zero(self, tmp_path):
        index_path = tmp_path / "index"
        index_path.mkdir()
        legacy_index = faiss.IndexFlatIP(DIM)
        legacy_index.add(np.array([_vec(0), _vec(1)], dtype=np.float32))
        faiss.write_index(legacy_index, str(index_path / "index.faiss"))
        (index_path / "chunk_ids.json").write_text(json.dumps(["a", "b"]))

        store = VectorStore(str(index_path), dim=DIM)
        store.load()

        assert len(store) == 2
        assert store.chunk_ids == ["a", "b"]
        assert (index_path / "manifest.json").exists()
        assert not (
            index_path / "index.faiss"
        ).exists(), "legacy file removed after migration"
        assert not (index_path / "chunk_ids.json").exists()

    def test_legacy_flat_files_with_mismatched_counts_rejected_not_guessed(
        self, tmp_path
    ):
        index_path = tmp_path / "index"
        index_path.mkdir()
        legacy_index = faiss.IndexFlatIP(DIM)
        legacy_index.add(np.array([_vec(0), _vec(1)], dtype=np.float32))
        faiss.write_index(legacy_index, str(index_path / "index.faiss"))
        (index_path / "chunk_ids.json").write_text(json.dumps(["a"]))  # only 1 id

        store = VectorStore(str(index_path), dim=DIM)
        store.load()

        assert len(store) == 0, "mismatched legacy pair must be rejected, not guessed"
        # Legacy files left in place for inspection since migration failed.
        assert (index_path / "index.faiss").exists()
