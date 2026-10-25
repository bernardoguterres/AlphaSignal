"""Table lane: period-matched table candidates, row-wise scoring, bounded
share of the final evidence (2026-10-21)."""

from datetime import date
from unittest.mock import patch

import numpy as np

from alphasignal.embeddings.cache import EmbeddingCache
from alphasignal.embeddings.embedder import Embedder
from alphasignal.ingestion import Chunk
from alphasignal.retrieval import RetrievedChunk
from alphasignal.retrieval.periods import column_matches, expected_period_end
from alphasignal.retrieval.reranker import CrossEncoderReranker, compose_context
from alphasignal.retrieval.retriever import HybridRetriever
from alphasignal.store.metadata_store import MetadataStore
from alphasignal.store.vector_store import VectorStore


def test_column_matching_by_date_quarter_and_fiscal_year():
    fy25 = expected_period_end(2025, "FY", (6, 30))
    assert column_matches("Year Ended June 30, 2025", 2025, "FY", fy25)
    assert not column_matches("Year Ended June 30, 2024", 2025, "FY", fy25)
    assert not column_matches("Three Months Ended June 30, 2025", 2025, "FY", fy25)
    assert column_matches("2025", 2025, "FY", fy25)
    q2 = expected_period_end(2026, "Q2", (12, 31))
    assert column_matches("2Q26", 2026, "Q2", q2)
    assert column_matches("Three months ended June 30, 2026", 2026, "Q2", q2)
    assert not column_matches("Six months ended June 30, 2026", 2026, "Q2", q2)
    assert not column_matches("1Q26", 2026, "Q2", q2)


def _rc(cid, lane="prose", score=0.0, text="t", cols=()):
    c = RetrievedChunk(
        chunk_id=cid,
        ticker="MSFT",
        text=text,
        doc_type="10-K",
        source="SEC EDGAR",
        section="item_8_table" if lane == "table" else "item_7",
        date=date(2025, 7, 30),
        url=None,
        dense_score=0,
        sparse_score=0,
        hybrid_score=0,
        final_score=score,
        lane=lane,
        match_columns=list(cols),
    )
    return c


def test_compose_reserves_table_slots_and_keeps_prose_order():
    ranked = [_rc("t1", "table"), _rc("t2", "table")] + [_rc(f"p{i}") for i in range(6)]
    out = [c.chunk_id for c in compose_context(ranked, 5, table_slots=1)]
    assert out == ["t1", "p0", "p1", "p2", "p3"]
    late_table = [_rc(f"p{i}") for i in range(6)] + [_rc("t1", "table")]
    out = [c.chunk_id for c in compose_context(late_table, 5, table_slots=1)]
    assert out == ["p0", "p1", "p2", "p3", "t1"]  # only the 5th prose gives way
    no_tables = [_rc(f"p{i}") for i in range(6)]
    assert len(compose_context(no_tables, 5)) == 5


TABLE = (
    "Financial table from MSFT Form 10-K\nTable: Income Statements\nUnits: In millions\n"
    "Columns: Year Ended June 30, 2025 | Year Ended June 30, 2024\n"
    "Revenue: Year Ended June 30, 2025 = 281,724; Year Ended June 30, 2024 = 245,122\n"
    "Net income: Year Ended June 30, 2025 = 101,832; Year Ended June 30, 2024 = 88,136"
)


def test_rows_are_scored_on_the_requested_column_only():
    r = CrossEncoderReranker.__new__(CrossEncoderReranker)
    r.window_words = 150
    chunk = _rc("t", "table", text=TABLE, cols=["Year Ended June 30, 2025"])
    chunk.header = "MSFT Form 10-K"
    windows = r._windows(chunk)
    assert len(windows) == 2
    assert windows[1].endswith("Net income: Year Ended June 30, 2025 = 101,832")
    assert all("2024" not in w.split("):", 1)[1] for w in windows)


def _chunk(cid, text, period_end, section):
    return Chunk(
        chunk_id=cid,
        ticker="MSFT",
        text=text,
        token_count=20,
        doc_type="10-K",
        source="SEC EDGAR",
        section=section,
        date=period_end,
        url=None,
        chunk_index=0,
        total_chunks=1,
        period_end=period_end,
    )


def test_table_lane_keeps_only_tables_with_the_requested_period(tmp_path):
    fy25, fy24 = date(2025, 6, 30), date(2024, 6, 30)
    other = TABLE.replace("2025", "2023").replace("281,724", "211,915")
    unaligned = (
        "Financial table\nTable: x\nUnits: In millions\n"
        "Columns: NOT ALIGNED to values (periods not asserted); headings as printed: 2025\n"
        "Net income: 1 | 2"
    )
    chunks = [
        _chunk("msft_10k_a_t0000", TABLE, fy25, "item_8_table"),
        _chunk("msft_10k_b_t0000", other, fy24, "item_8_table"),
        _chunk("msft_10k_a_t0001", unaligned, fy25, "item_8_table"),
        _chunk("msft_10k_a_0000", "net income discussion", fy25, "item_7"),
    ]
    ms = MetadataStore(str(tmp_path / "m.db"))
    ms.add_chunks(chunks)
    vs = VectorStore(str(tmp_path / "idx"), dim=8)
    vs.load()
    vs.add(
        np.ones((len(chunks), 8), dtype=np.float32),
        [c.chunk_id for c in chunks],
        {c.chunk_id: c.content_hash for c in chunks},
    )
    with patch("alphasignal.embeddings.embedder.OpenAI"):
        emb = Embedder({"embeddings": {}}, EmbeddingCache(str(tmp_path / "c.npy")))
    emb.embed_texts = lambda texts: [np.ones(8, dtype=np.float32) for _ in texts]
    r = HybridRetriever(
        {
            "retrieval": {
                "table_lane": True,
                "dense_candidates": 5,
                "sparse_candidates": 5,
            }
        },
        emb,
        vs,
        ms,
    )
    pool = r.retrieve(
        "What was Microsoft's net income in fiscal year 2025?", "MSFT", top_k=5
    )
    tables = [c for c in pool if c.lane == "table"]
    assert [c.chunk_id for c in tables] == ["msft_10k_a_t0000"]
    assert tables[0].match_columns == ["Year Ended June 30, 2025"]
    assert (
        "column(s) for the requested period: Year Ended June 30, 2025"
        in tables[0].header
    )
    assert "msft_10k_a_0000" in [c.chunk_id for c in pool]  # prose kept


def test_reported_statements_always_reach_the_lane(tmp_path):
    fy25 = date(2025, 6, 30)
    statement = (
        "Financial table from MSFT Form 10-K\nTable: INCOME STATEMENTS\nUnits: In millions\n"
        'Basis: reported (heading: "INCOMESTATEMENTS")\n'
        "Columns: Year Ended June 30, 2025 | Year Ended June 30, 2024\n"
        "Net income: Year Ended June 30, 2025 = 101,832; Year Ended June 30, 2024 = 88,136"
    )
    tax = statement.replace("INCOME STATEMENTS", "Current taxes").replace(
        'Basis: reported (heading: "INCOMESTATEMENTS")', "Basis: not stated"
    )
    chunks = [_chunk("msft_10k_a_ch7_0001", statement, fy25, "item_8_table")] + [
        _chunk(f"msft_10k_a_ch7_01{i:02d}", tax + f" {i}", fy25, "item_8_table")
        for i in range(40)
    ]
    ms = MetadataStore(str(tmp_path / "m.db"))
    ms.add_chunks(chunks)
    vs = VectorStore(str(tmp_path / "idx"), dim=8)
    vs.load()
    vecs = np.zeros((len(chunks), 8), dtype=np.float32)
    vecs[:, 0] = 1.0
    vecs[0] = np.eye(8, dtype=np.float32)[1]  # the statement is far from the query
    vs.add(
        vecs, [c.chunk_id for c in chunks], {c.chunk_id: c.content_hash for c in chunks}
    )
    with patch("alphasignal.embeddings.embedder.OpenAI"):
        emb = Embedder({"embeddings": {}}, EmbeddingCache(str(tmp_path / "c.npy")))
    emb.embed_texts = lambda texts: [np.eye(8, dtype=np.float32)[0] for _ in texts]
    r = HybridRetriever(
        {
            "retrieval": {
                "table_lane": True,
                "dense_candidates": 5,
                "sparse_candidates": 5,
                "period_scoped_candidates": 3,
            }
        },
        emb,
        vs,
        ms,
    )
    pool = r.retrieve("Microsoft current taxes fiscal year 2025?", "MSFT", top_k=5)
    lane = [c for c in pool if c.lane == "table"]
    assert "msft_10k_a_ch7_0001" in [c.chunk_id for c in lane]
    assert (
        next(c for c in lane if c.chunk_id == "msft_10k_a_ch7_0001").basis == "reported"
    )
