"""Corpus and index versions.

A corpus version is derived from what is stored, not assigned, so it cannot
drift from the data:

* ``content_digest`` - sha256 over the sorted (chunk_id, content_hash) pairs.
  SQLite and FAISS each compute it from their own records; the two are the
  same version only when the digests agree.
* ``corpus_version`` - sha256 over every SQLite row's identity, content hash
  and provenance (source_id, accession, reporting period, filing date, url,
  ticker, form, section), so a provenance change is a new version too.
"""

import hashlib
from collections.abc import Iterable


def content_digest(pairs: Iterable[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for chunk_id, content_hash in sorted(pairs):
        h.update(f"{chunk_id}\t{content_hash}\n".encode())
    return "cd-" + h.hexdigest()[:16]


def corpus_digest(rows: Iterable[tuple]) -> str:
    h = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r[0]):
        h.update(("\t".join("" if v is None else str(v) for v in row) + "\n").encode())
    return "cv-" + h.hexdigest()[:16]


def chunker_versions(chunk_ids: Iterable[str]) -> dict[str, int]:
    """Count of chunks per chunker version (legacy unversioned ids = ch1)."""
    from alphasignal.ingestion.chunker import source_of

    counts: dict[str, int] = {}
    for cid in chunk_ids:
        rest = cid[len(source_of(cid)) + 1 :]
        version = rest.split("_")[0] if "_" in rest else "ch1"
        counts[version] = counts.get(version, 0) + 1
    return counts
