"""Query provenance when an ingestion updates SQLite before FAISS."""

from dataclasses import replace
from datetime import date
from unittest.mock import patch

import numpy as np
import pytest

from alphasignal.generation import GenerationResult
from alphasignal.ingestion import Chunk
from alphasignal.tests.test_api import client  # noqa: F401


@pytest.mark.parametrize("during_query", [False, True])
def test_sparse_citations_match_the_ranked_text(client, monkeypatch, during_query):
    monkeypatch.delenv("ALPHASIGNAL_API_KEY", raising=False)
    state = client.app.state.app_state
    pipeline = state.pipeline
    old = Chunk(
        chunk_id="aapl_filing_item_1_0000",
        ticker="AAPL",
        text="orchid orchard",
        token_count=2,
        doc_type="10-K",
        source="SEC EDGAR",
        section="item_1",
        date=date(2025, 1, 1),
        url=None,
        chunk_index=0,
        total_chunks=1,
        source_id="aapl_filing",
    )
    new = replace(old, text="turbine manufacturing", content_hash="")
    unrelated = [
        replace(
            old,
            chunk_id=f"msft_{i}_0000",
            ticker="MSFT",
            text=text,
            source_id=f"msft_{i}",
            content_hash="",
        )
        for i, text in enumerate(["electronics chips", "software applications"])
    ]
    vector = np.eye(1, 1536, dtype=np.float32)[0]
    chunks = [old, *unrelated]
    pipeline.store_chunks(chunks, {chunk.chunk_id: vector for chunk in chunks})
    state.retriever.build_bm25_index()

    def query(text, ticker="AAPL"):
        response = client.post(
            "/query/", json={"query": text, "ticker_filter": ticker, "top_k": 2}
        )
        assert response.status_code == 200, response.text
        return response.json()

    with (
        patch.object(pipeline, "ingest_ticker", return_value=([], [])),
        patch.object(pipeline, "chunk_documents", return_value=[new]),
        patch.object(pipeline, "chunk_articles", return_value=[]),
        patch.object(
            pipeline.embedder, "embed_chunks", return_value={new.chunk_id: vector}
        ),
        patch.object(pipeline.embedder, "embed_texts", return_value=[vector]),
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
        assert [c["excerpt"] for c in query("orchid")["citations"]] == [
            "orchid orchard"
        ]

        if during_query:
            sparse_search = state.retriever._sparse_search

            def rank_then_replace(*args, **kwargs):
                ranked = sparse_search(*args, **kwargs)
                pipeline.metadata_store.add_chunks([new])
                return ranked

            with patch.object(
                state.retriever, "_sparse_search", side_effect=rank_then_replace
            ):
                assert query("orchid")["citations"] == []
        else:
            with patch.object(pipeline.vector_store, "save", return_value=None):
                failed = client.post(
                    "/ingest/AAPL", json={"ticker": "AAPL", "include_news": False}
                )
            assert failed.status_code == 200, failed.text
            assert failed.json()["status"] == "failed"
            assert pipeline.metadata_store.get_chunk(old.chunk_id).text == new.text
            after = query("orchid")
            assert after["retrieval"]["dense_excluded_unverified"] == 1
            assert after["citations"] == []
            assert [c["excerpt"] for c in query("turbine")["citations"]] == [new.text]

        assert any(
            c["excerpt"] == unrelated[0].text
            for c in query("electronics", "MSFT")["citations"]
        )
        retried = client.post(
            "/ingest/AAPL", json={"ticker": "AAPL", "include_news": False}
        )
        assert retried.status_code == 200, retried.text
        assert retried.json()["status"] == "completed"
        assert pipeline.vector_store.content_hashes[new.chunk_id] == new.content_hash
        assert [c["excerpt"] for c in query("turbine")["citations"]] == [new.text]


def test_stale_sparse_score_does_not_displace_rank_20(client, monkeypatch):
    monkeypatch.delenv("ALPHASIGNAL_API_KEY", raising=False)
    state = client.app.state.app_state
    pipeline = state.pipeline
    old = Chunk(
        chunk_id="aapl_stale_item_1_0000",
        ticker="AAPL",
        text="orchid orchard",
        token_count=2,
        doc_type="10-K",
        source="SEC EDGAR",
        section="item_1",
        date=date(2025, 1, 1),
        url=None,
        chunk_index=0,
        total_chunks=1,
        source_id="aapl_stale",
    )
    replacement = replace(old, text="turbine manufacturing", content_hash="")
    other = [
        replace(
            old,
            chunk_id=f"aapl_current_{i}_item_1_0000",
            source_id=f"aapl_current_{i}",
            text=f"current evidence number {i}",
            content_hash="",
        )
        for i in range(21)
    ]

    def vec(similarity):
        arr = np.zeros(1536, dtype=np.float32)
        arr[:2] = [similarity, np.sqrt(1 - similarity**2)]
        return arr

    vectors = {
        chunk.chunk_id: vec(0.7 + 0.01 * i if i < 20 else 0)
        for i, chunk in enumerate(other)
    }
    vectors[old.chunk_id] = vec(0.6)
    pipeline.store_chunks([old, *other], vectors)
    state.retriever.build_bm25_index()
    # A separate writer can finish ingestion while this query worker keeps
    # its BM25 snapshot. Model that boundary with the normal ingestion storage
    # call, which publishes coherent new SQLite and FAISS records.
    pipeline.store_chunks([replacement], {replacement.chunk_id: vec(0.6)})
    assert (
        pipeline.vector_store.content_hashes[old.chunk_id] == replacement.content_hash
    )
    assert state.retriever.bm25_chunks[old.chunk_id].text == old.text

    with (
        patch.object(pipeline.embedder, "embed_texts", return_value=[vec(1)]),
        patch.object(
            state.reranker, "rerank", side_effect=lambda query, chunks, **kw: chunks
        ) as reranker,
        patch.object(
            state.generator,
            "generate",
            side_effect=lambda query, chunks: GenerationResult(
                "synthetic", chunks, 0, 0, "stub"
            ),
        ),
    ):
        response = client.post(
            "/query/", json={"query": "orchid", "ticker_filter": "AAPL", "top_k": 20}
        )

    assert response.status_code == 200, response.text
    actual = reranker.call_args.kwargs["chunks"]
    expected_ids = {chunk.chunk_id for chunk in other[:20]}
    by_id = {chunk.chunk_id: chunk for chunk in actual}
    assert set(by_id) == expected_ids, [
        (c.chunk_id, c.dense_score, c.sparse_score, c.hybrid_score) for c in actual
    ]
    assert set(response.json()["retrieved_chunk_ids"]) == expected_ids
