"""Versioned chunk identities and corpus/index versions (2026-10)."""

import json
import subprocess
import sys
from datetime import date

import numpy as np

from alphasignal.ingestion import Chunk
from alphasignal.ingestion import chunker as chunker_mod
from alphasignal.ingestion.chunker import chunk_id_for, source_of
from alphasignal.ingestion.pipeline import IngestionPipeline
from alphasignal.store.metadata_store import MetadataStore
from alphasignal.store.vector_store import VectorStore


def _chunks(texts, version):
    out = []
    for i, t in enumerate(texts):
        out.append(
            Chunk(
                chunk_id=f"nvda_10k_86184b4a_{version}_{i:04d}",
                ticker="NVDA",
                text=t,
                token_count=5,
                doc_type="10-K",
                source="SEC EDGAR",
                section="item_7",
                date=date(2025, 2, 26),
                url=None,
                chunk_index=i,
                total_chunks=len(texts),
                source_id="nvda_10k_86184b4a",
                accession_number="0001045810-25-000023",
                attribution="filer",
            )
        )
    return out


def _stores(tmp_path, dim=4):
    ms = MetadataStore(str(tmp_path / "data" / "metadata.db"))
    vs = VectorStore(str(tmp_path / "data" / "faiss_index"), dim=dim)
    vs.load()
    p = IngestionPipeline.__new__(IngestionPipeline)
    p.metadata_store, p.vector_store = ms, vs
    return p, ms, vs


def _store(p, chunks, dim=4):
    emb = {
        c.chunk_id: np.ones(dim, dtype=np.float32) * (i + 1)
        for i, c in enumerate(chunks)
    }
    p.store_chunks(chunks, emb)


def test_ids_carry_the_chunker_version_and_parse_back():
    cid = chunk_id_for("nvda_10k_86184b4a", 3)
    assert cid == f"nvda_10k_86184b4a_{chunker_mod.CHUNKER_VERSION}_0003"
    assert source_of(cid) == "nvda_10k_86184b4a"
    assert source_of("nvda_10k_86184b4a_0003") == "nvda_10k_86184b4a"  # legacy
    assert source_of("msft_news_c1661790_0003") == "msft_news_c1661790"  # hex hash


def test_rechunking_under_a_new_version_never_repoints_old_ids(tmp_path):
    p, ms, vs = _stores(tmp_path)
    _store(p, _chunks(["alpha beta", "gamma delta"], "ch1"))
    old = {c: ms.get_chunk(c).text for c in ms.get_all_chunk_ids()}
    _store(p, _chunks(["alpha", "beta gamma", "delta"], "ch2"))  # new boundaries
    ids = set(ms.get_all_chunk_ids())
    assert not ids & set(old)  # old ids removed, not pointed at new text
    assert ids == {f"nvda_10k_86184b4a_ch2_{i:04d}" for i in range(3)}
    assert set(vs.chunk_ids) == ids


def test_database_and_index_digests_agree_only_for_the_same_chunks(tmp_path):
    p, ms, vs = _stores(tmp_path)
    _store(p, _chunks(["a", "b"], "ch2"))
    assert ms.version_info()["content_digest"] == vs.content_digest()
    vs.remove(["nvda_10k_86184b4a_ch2_0001"])
    assert ms.version_info()["content_digest"] != vs.content_digest()


def _cv(*args, cwd):
    return subprocess.run(
        [sys.executable, "-m", "alphasignal.scripts.corpus_version", *map(str, args)],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def test_snapshot_activate_and_restore_round_trip(tmp_path, monkeypatch):
    p, ms, vs = _stores(tmp_path, dim=1536)
    _store(p, _chunks(["first version text"], "ch1"), dim=1536)
    vs.save()
    data = tmp_path / "data"
    root = str(__import__("pathlib").Path(__file__).resolve().parents[2])
    r = _cv("snapshot", "--data", data, cwd=root)
    assert r.returncode == 0, r.stderr
    (v1,) = (data / "versions").iterdir()
    assert _cv("verify", v1, cwd=root).returncode == 0

    # a new version replaces the corpus
    _store(p, _chunks(["second", "version"], "ch2"), dim=1536)
    vs.save()
    v2_info = json.loads(_cv("show", "--data", data, cwd=root).stdout)
    assert v2_info["consistent"] and v2_info["corpus_version"] != v1.name

    # restore v1: the active v2 is backed up first, then v1 is swapped in
    r = _cv("activate", v1, "--data", data, cwd=root)
    assert r.returncode == 0, r.stdout + r.stderr
    active = json.loads(_cv("show", "--data", data, cwd=root).stdout)
    assert active["corpus_version"] == v1.name and active["consistent"]
    assert (data / "versions" / v2_info["corpus_version"]).exists()  # backup
    assert (
        MetadataStore(str(data / "metadata.db"))
        .get_chunk("nvda_10k_86184b4a_ch1_0000")
        .text
        == "first version text"
    )
