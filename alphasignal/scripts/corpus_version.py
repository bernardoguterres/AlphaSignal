"""Corpus versions: show, snapshot, verify and activate (restore) a corpus.

A corpus is three artifacts that must belong together: ``metadata.db``
(text + provenance), ``faiss_index/`` (vectors) and, optionally,
``embeddings_cache/``. Versions are derived from the stored data (see
``alphasignal.store.versioning``): a snapshot is only taken - and a corpus
only activated - when the database's content digest equals the index's.

    python -m alphasignal.scripts.corpus_version show     --data data
    python -m alphasignal.scripts.corpus_version snapshot --data data [--with-cache]
    python -m alphasignal.scripts.corpus_version verify   data/versions/<version>
    python -m alphasignal.scripts.corpus_version activate data/versions/<version> --data data

``activate`` (which is also "restore") verifies the target, snapshots the
currently active corpus first (the recoverable backup), stages the target
inside ``--data``, swaps the artifacts in, re-verifies, and rolls back to the
backup if the result is not the expected version. Replaced artifacts are
moved to ``<data>/.replaced-<version>-<time>/``, never deleted. Stop the
service before activating and restart it afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ARTIFACTS = ("metadata.db", "faiss_index", "embeddings_cache")


def describe(data: Path) -> dict:
    from alphasignal.store.metadata_store import MetadataStore
    from alphasignal.store.vector_store import VectorStore

    ms = MetadataStore(str(data / "metadata.db"))
    info = ms.version_info()
    ms.engine.dispose()
    vs = VectorStore(str(data / "faiss_index"), dim=1536)
    vs.load()
    info["index_digest"] = vs.content_digest()
    info["vectors"] = len(vs.chunk_ids)
    info["faiss_generation"] = vs.generation
    info["consistent"] = info["index_digest"] == info["content_digest"]
    return info


def _copy_db(src: Path, dst: Path) -> None:
    """Consistent copy through SQLite's backup API (safe if the source is open)."""
    with sqlite3.connect(f"file:{src}?mode=ro", uri=True) as s, sqlite3.connect(
        dst
    ) as d:
        s.backup(d)


def _copy_index(src: Path, dst: Path) -> None:
    """Active generation files + manifest only (not older generations)."""
    dst.mkdir(parents=True)
    manifest = json.loads((src / "manifest.json").read_text())
    g = manifest["generation"]
    for name in (f"index.g{g}.faiss", f"chunk_ids.g{g}.json", "manifest.json"):
        shutil.copy2(src / name, dst / name)


def snapshot(data: Path, dest_root: Path, with_cache: bool) -> Path:
    info = describe(data)
    if not info["consistent"]:
        raise SystemExit(
            f"refusing to snapshot: database {info['content_digest']} != index "
            f"{info['index_digest']} (not one version)"
        )
    dest = dest_root / info["corpus_version"]
    if dest.exists():
        if verify(dest, quiet=True):
            print(f"snapshot {dest} already exists and verifies")
            return dest
        raise SystemExit(f"{dest} exists but does not verify - inspect it")
    tmp = dest_root / f".tmp-{info['corpus_version']}"
    tmp.mkdir(parents=True)
    _copy_db(data / "metadata.db", tmp / "metadata.db")
    _copy_index(data / "faiss_index", tmp / "faiss_index")
    if with_cache and (data / "embeddings_cache").exists():
        shutil.copytree(data / "embeddings_cache", tmp / "embeddings_cache")
    info.update(
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source=str(data.resolve()),
        with_cache=with_cache,
    )
    (tmp / "version.json").write_text(json.dumps(info, indent=1))
    os.replace(tmp, dest)
    print(f"snapshot {dest}")
    return dest


def verify(snap: Path, quiet: bool = False) -> bool:
    recorded = json.loads((snap / "version.json").read_text())
    now = describe(snap)
    ok = now["consistent"] and all(
        now[k] == recorded[k]
        for k in ("corpus_version", "content_digest", "index_digest")
    )
    if not quiet:
        print(json.dumps({"snapshot": str(snap), "verified": ok, **now}, indent=1))
    return ok


def activate(snap: Path, data: Path, versions: Path) -> None:
    if not verify(snap, quiet=True):
        raise SystemExit(f"{snap} does not verify; not activating")
    target = json.loads((snap / "version.json").read_text())["corpus_version"]
    backup = snapshot(data, versions, with_cache=True)  # recoverable backup
    current = json.loads((backup / "version.json").read_text())["corpus_version"]
    if current == target:
        print(f"{target} is already active")
        return

    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    staging = data / f".staging-{target}"
    staging.mkdir()
    shutil.copy2(snap / "metadata.db", staging / "metadata.db")
    shutil.copytree(snap / "faiss_index", staging / "faiss_index")
    if (snap / "embeddings_cache").exists():
        shutil.copytree(snap / "embeddings_cache", staging / "embeddings_cache")
    staged = describe(staging)
    if staged["corpus_version"] != target or not staged["consistent"]:
        raise SystemExit(f"staged copy in {staging} did not verify; nothing changed")

    replaced = data / f".replaced-{current}-{stamp}"
    replaced.mkdir()
    for name in ARTIFACTS:
        if (staging / name).exists():
            if (data / name).exists():
                os.replace(data / name, replaced / name)
            os.replace(staging / name, data / name)
    staging.rmdir()
    active = describe(data)
    if active["corpus_version"] != target or not active["consistent"]:
        print("activation did not verify - restoring the previous version")
        for name in ARTIFACTS:
            if (replaced / name).exists():
                if (data / name).exists():
                    os.replace(data / name, staging / name)
                os.replace(replaced / name, data / name)
        raise SystemExit(f"rolled back to {current}")
    print(
        f"ACTIVE corpus_version {target} (index {active['index_digest']}); previous "
        f"{current} is backed up at {backup}; replaced files in {replaced}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("show", "snapshot", "activate"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, default=Path("data"))
        p.add_argument("--versions", type=Path, default=None)
        if name == "snapshot":
            p.add_argument("--with-cache", action="store_true")
        if name == "activate":
            p.add_argument("snapshot", type=Path)
    v = sub.add_parser("verify")
    v.add_argument("snapshot", type=Path)
    args = ap.parse_args()

    if args.cmd == "verify":
        sys.exit(0 if verify(args.snapshot) else 1)
    versions = args.versions or args.data / "versions"
    if args.cmd == "show":
        print(json.dumps(describe(args.data), indent=1))
    elif args.cmd == "snapshot":
        snapshot(args.data, versions, args.with_cache)
    else:
        activate(args.snapshot, args.data, versions)


if __name__ == "__main__":
    main()
