"""Public HTTP reproduction of stale sparse boosts in early candidate caps."""

from dataclasses import replace
from datetime import date
from unittest.mock import patch

import numpy as np
import pytest

from alphasignal.generation import GenerationResult
from alphasignal.ingestion import Chunk
from alphasignal.tests.test_api import client  # noqa: F401


@pytest.mark.parametrize("table_lane", [False, True], ids=["period", "table"])
def test_early_lane_cutoff_uses_current_scores(client, monkeypatch, table_lane):
    monkeypatch.delenv("ALPHASIGNAL_API_KEY", raising=False)
    state = client.app.state.app_state
    pipeline = state.pipeline
    retriever = state.retriever
    retriever.table_lane = table_lane
    retriever.period_candidates = 1 if table_lane else 2
    retriever.dense_candidates = 1
    retriever.sparse_candidates = 1

    def chunk(name, text):
        # No explicit reported-basis tag: exercise the ranking cap, without
        # falling back to the separately reserved reported-statement lane.
        if table_lane:
            text = f"Columns: Year Ended December 31, 2025\nLine item: {text}"
        return Chunk(
            chunk_id=f"aapl_synthetic_{name}_0000",
            ticker="AAPL",
            text=text,
            token_count=10,
            doc_type="10-K",
            source="SEC EDGAR",
            section="item_8_table" if table_lane else "item_7",
            date=date(2027 if table_lane else 2026, 2, 1),
            period_end=date(2026 if table_lane else 2025, 12, 31),
            url=None,
            chunk_index=0,
            total_chunks=1,
            source_id=f"aapl_synthetic_{name}",
        )

    def vec(similarity):
        arr = np.zeros(1536, dtype=np.float32)
        arr[:2] = [similarity, np.sqrt(1 - similarity**2)]
        return arr

    old = chunk("stale", "orchid orchard")
    current = chunk("stale", "turbine manufacturing")
    best = chunk("best", "current evidence")
    runner_up = chunk("runner", "second current evidence")
    vectors = {
        best.chunk_id: vec(0.9),
        runner_up.chunk_id: vec(0.1 if table_lane else 0.8),
        old.chunk_id: vec(0.8 if table_lane else 0.1),
    }
    chunks = [old, best, runner_up]
    if table_lane:
        own_filing = replace(
            chunk("own_filing", "own filing"),
            section="item_7",
            text="own filing",
            content_hash="",
            date=date(2026, 2, 1),
            period_end=date(2025, 12, 31),
        )
        chunks.append(own_filing)
        vectors[own_filing.chunk_id] = vec(0.5)
    pipeline.store_chunks(chunks, vectors)
    retriever.build_bm25_index()
    pipeline.store_chunks([current], {current.chunk_id: vectors[old.chunk_id]})
    assert retriever.bm25_chunks[old.chunk_id].text == old.text
    assert pipeline.vector_store.content_hashes[old.chunk_id] == current.content_hash

    def query():
        response = client.post(
            "/query/",
            json={"query": "orchid FY2025", "ticker_filter": "AAPL", "top_k": 2},
        )
        assert response.status_code == 200, response.text
        return response.json()["retrieved_chunk_ids"]

    with (
        patch.object(pipeline.embedder, "embed_texts", return_value=[vec(1)]),
        patch.object(
            state.reranker, "rerank", side_effect=lambda query, chunks, **kw: chunks
        ),
        patch.object(
            state.generator,
            "generate",
            side_effect=lambda query, chunks: GenerationResult(
                "synthetic", chunks, 0, 0, "stub"
            ),
        ),
    ):
        stale_snapshot_ids = query()
        retriever.build_bm25_index()
        control_ids = query()

    expected = (
        [own_filing.chunk_id, best.chunk_id]
        if table_lane
        else [best.chunk_id, runner_up.chunk_id]
    )
    assert control_ids == expected
    assert stale_snapshot_ids == expected
