"""Rebuild the FAISS index from the CURRENT chunk text, with content hashes.

Why: vectors in a legacy (pre-content-hash) index have unknown provenance,
so HybridRetriever excludes every one of them and "hybrid" retrieval runs
as BM25-only. There is no way to certify an existing vector without
re-embedding its text, so the repair is a rebuild.

Safety: writes into --out-index (a NEW directory, refused if it already
holds an index) and never touches the live index. Activate it only after
the verification report passes (swap directories, keeping the old one as a
backup).

Verification (printed as JSON, non-zero exit on failure):
  - vector count == chunk count, every chunk_id present once
  - every vector's recorded content_hash == the chunk's SQLite content_hash
  - spot check: re-embed a random sample of chunk texts and compare with the
    stored vectors (cosine >= 0.999 - the same text and model)
  - self-retrieval: each sampled chunk's own vector retrieves itself first

Usage:
    python -m alphasignal.scripts.rebuild_index --db data/metadata.db \
        --out-index data/faiss_index.rebuilt --cache data/embeddings_cache.rebuilt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from sqlmodel import Session, select

from alphasignal.embeddings.cache import EmbeddingCache
from alphasignal.embeddings.embedder import Embedder
from alphasignal.scripts._common import load_config
from alphasignal.store.metadata_store import ChunkRecord, MetadataStore
from alphasignal.store.vector_store import VectorStore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out-index", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="skip embedding; verify an existing --out-index",
    )
    args = ap.parse_args()

    out = Path(args.out_index)
    if not args.verify_only and out.exists() and any(out.iterdir()):
        print(f"refusing: {out} is not empty", file=sys.stderr)
        return 2

    config = load_config(Path(__file__).resolve().parents[2])
    store = MetadataStore(args.db)
    with Session(store.engine) as s:
        chunks = [MetadataStore._to_chunk(r) for r in s.exec(select(ChunkRecord)).all()]
    embedder = Embedder(config, EmbeddingCache(str(Path(args.cache) / "cache.npy")))
    vs = VectorStore(str(out), dim=1536)
    vs.load()

    t0 = time.time()
    for i in range(0, 0 if args.verify_only else len(chunks), args.batch):
        batch = chunks[i : i + args.batch]
        emb = embedder.embed_chunks(batch)
        ids = [c.chunk_id for c in batch]
        vs.add(
            np.array([emb[c] for c in ids], dtype=np.float32),
            ids,
            {c.chunk_id: c.content_hash for c in batch},
        )
        print(
            f"embedded {min(i + args.batch, len(chunks))}/{len(chunks)} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
    if not args.verify_only:
        vs.save()

    # ---- verification on a fresh load ----
    v = VectorStore(str(out), dim=1536)
    v.load()
    by_id = {c.chunk_id: c for c in chunks}
    report = {
        "chunks": len(chunks),
        "vectors": int(v.index.ntotal),
        "unique_ids": len(set(v.chunk_ids)),
        # Reported, not raised: a vector whose chunk no longer exists, or a
        # chunk with no vector, is exactly what verification must surface.
        "orphan_vectors": sum(1 for cid in v.chunk_ids if cid not in by_id),
        "chunks_without_vector": len(by_id.keys() - set(v.chunk_ids)),
        "hash_mismatches": sum(
            1
            for cid in v.chunk_ids
            if cid in by_id and v.content_hashes.get(cid) != by_id[cid].content_hash
        ),
        "missing_hashes": sum(
            1 for cid in v.chunk_ids if not v.content_hashes.get(cid)
        ),
    }
    rng = random.Random(args.seed)
    pos = {cid: i for i, cid in enumerate(v.chunk_ids)}
    indexed = [c for c in chunks if c.chunk_id in pos]
    sample = rng.sample(indexed, min(args.sample, len(indexed)))
    fresh = embedder.embed_texts([c.text for c in sample])
    stored = np.vstack([v.index.reconstruct(pos[c.chunk_id]) for c in sample])
    fresh_n = fresh / np.linalg.norm(fresh, axis=1, keepdims=True)
    stored_n = stored / np.linalg.norm(stored, axis=1, keepdims=True)
    cos = (fresh_n * stored_n).sum(axis=1)
    report["reembed_cosine_min"] = round(float(cos.min()), 6)
    # A chunk's own vector must retrieve itself - or a chunk with
    # IDENTICAL text (the corpus holds exact duplicates whose vectors are
    # identical, so either is a correct top hit).
    self_hits = 0
    for c, vec in zip(sample, stored):
        res = v.search(vec, k=1)
        top = res[0][0] if res else None
        self_hits += top is not None and (
            top == c.chunk_id
            or (top in by_id and by_id[top].content_hash == c.content_hash)
        )
    report["self_retrieval"] = f"{self_hits}/{len(sample)}"
    report["seconds"] = round(time.time() - t0, 1)
    ok = (
        report["vectors"] == report["chunks"] == report["unique_ids"]
        and report["hash_mismatches"] == 0
        and report["orphan_vectors"] == 0
        and report["chunks_without_vector"] == 0
        and report["missing_hashes"] == 0
        and report["reembed_cosine_min"] >= 0.999
        and self_hits == len(sample)
    )
    report["verified"] = ok
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
