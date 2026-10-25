"""Reporting periods, period-scoped retrieval and financial-table extraction (2026-10)."""

import subprocess
import sys
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from alphasignal.embeddings.cache import EmbeddingCache
from alphasignal.embeddings.embedder import Embedder
from alphasignal.generation.generator import RAGGenerator
from alphasignal.ingestion import Chunk, RawDocument
from alphasignal.ingestion.chunker import SemanticChunker
from alphasignal.ingestion.edgar import EDGARIngester
from alphasignal.ingestion.tables import (
    TABLE_SECTION_SUFFIX,
    extract_tables,
    render_table,
)
from alphasignal.retrieval import RetrievedChunk
from alphasignal.retrieval.periods import (
    fiscal_label,
    requested_periods,
    source_header,
)
from alphasignal.retrieval.reranker import CrossEncoderReranker
from alphasignal.retrieval.retriever import HybridRetriever
from alphasignal.store.metadata_store import MetadataStore
from alphasignal.store.vector_store import VectorStore

# --------------------------------------------------------------------------
# Fiscal labels and query periods
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "doc_type, period_end, fye, expected",
    [
        ("10-K", date(2025, 1, 26), (1, 25), (2025, "FY")),  # NVDA FY2025
        ("10-Q", date(2024, 4, 28), (1, 25), (2025, "Q1")),  # NVDA Q1 FY2025
        ("10-Q", date(2024, 10, 27), (1, 25), (2025, "Q3")),
        ("10-K", date(2025, 6, 30), (6, 30), (2025, "FY")),  # MSFT FY2025
        ("10-Q", date(2024, 9, 30), (6, 30), (2025, "Q1")),
        ("10-Q", date(2024, 12, 28), (9, 27), (2025, "Q1")),  # AAPL Q1 FY2025
        ("10-K", date(2024, 12, 31), (12, 31), (2024, "FY")),  # JPM
        ("10-Q", date(2024, 3, 31), (12, 31), (2024, "Q1")),
        ("news", date(2024, 3, 31), (12, 31), None),
        ("10-K", None, (12, 31), None),
    ],
)
def test_fiscal_label(doc_type, period_end, fye, expected):
    assert fiscal_label(doc_type, period_end, fye) == expected


def test_requested_periods_reads_fiscal_years_and_quarters():
    assert requested_periods("employees at the end of fiscal year 2025?", (1, 25)) == [
        (2025, "FY")
    ]
    assert requested_periods("FY26 revenue", (1, 25)) == [(2026, "FY")]
    assert requested_periods("first quarter of fiscal 2025", (1, 25)) == [(2025, "Q1")]
    assert requested_periods("Q4 fiscal 2025", (1, 25)) == [(2025, "FY")]
    assert requested_periods("What was the company's revenue?", (1, 25)) == []


def test_bare_calendar_year_only_counts_for_calendar_fiscal_years():
    assert requested_periods("net income in 2024?", (12, 31)) == [(2024, "FY")]
    # NVIDIA's "2024" is ambiguous (fiscal 2024 ended January 2024)
    assert requested_periods("net income in 2024?", (1, 25)) == []


def test_source_header_separates_reporting_period_from_filing_date():
    chunk = SimpleNamespace(
        ticker="NVDA",
        doc_type="10-K",
        date=date(2025, 2, 26),
        period_end=date(2025, 1, 26),
    )
    header = source_header(chunk, (1, 25))
    assert "reporting period: fiscal year 2025, ended 2025-01-26" in header
    assert "filed 2025-02-26" in header
    unknown = source_header(
        SimpleNamespace(ticker="X", doc_type="10-Q", date=date(2020, 1, 1)), None
    )
    assert "reporting period: unknown" in unknown


# --------------------------------------------------------------------------
# Table extraction
# --------------------------------------------------------------------------

# JPM-style: period row on its own grid, "$" cells, a parent spanning the
# six-month columns and a footnote cell inside the values.
_JPM = """<html><body>
<p>Consolidated financial highlights</p>
<table>
<tr><td colspan="6">As of or for the period ended, (in millions)</td>
    <td colspan="9">Six months ended June 30,</td></tr>
<tr><td colspan="3">2Q26</td><td colspan="3">1Q26</td><td colspan="3">2026</td>
    <td colspan="3">2025</td></tr>
<tr><td colspan="6">Total net revenue</td><td>$</td><td>57,347</td>
    <td colspan="3">(e)</td><td>$</td><td>49,836</td><td>$</td><td>107,183</td>
    <td>$</td><td>90,222</td></tr>
<tr><td colspan="6">Net income</td><td colspan="2">(1,234</td><td>)</td>
    <td colspan="2">10</td><td colspan="2">20</td><td colspan="2">30</td></tr>
</table>
<table><tr><td>Signature</td><td>Name</td></tr><tr><td>/s/ A</td><td>B</td></tr></table>
</body></html>"""

# NVDA-style: "Year Ended" spanning two date columns.
_NVDA = """<html><body>
<p>Consolidated Statements of Income (In millions)</p>
<table>
<tr><td colspan="3"></td><td colspan="9">Year Ended</td></tr>
<tr><td colspan="3"></td><td colspan="3">Jan 26, 2025</td><td colspan="3"></td>
    <td colspan="3">Jan 28, 2024</td></tr>
<tr><td colspan="3">Revenue</td><td>$</td><td colspan="2">130,497</td>
    <td colspan="3"></td><td>$</td><td colspan="2">60,922</td></tr>
<tr><td colspan="3">Operating expenses</td></tr>
<tr><td colspan="3">Net income</td><td>$</td><td colspan="2">72,880</td>
    <td colspan="3"></td><td>$</td><td colspan="2">29,760</td></tr>
</table></body></html>"""


def _rows(table):
    return {label: dict(vals) for label, vals in table.rows}


def test_jpm_style_values_get_their_own_period_never_a_shifted_one():
    tables, marked = extract_tables(_JPM)
    assert len(tables) == 1  # the signature layout table is skipped
    rows = _rows(tables[0])
    assert rows["Total net revenue"] == {
        "2Q26": "57,347",
        "1Q26": "49,836",
        "Six months ended June 30, 2026": "107,183",
        "Six months ended June 30, 2025": "90,222",
    }
    assert rows["Net income"]["2Q26"] == "(1,234)"
    assert tables[0].units == "in millions"
    assert "[[TABLE 0]]" in marked


def test_nvda_style_parent_and_dates_and_group_heading():
    (table,) = extract_tables(_NVDA)[0]
    rows = _rows(table)
    assert rows["Revenue"] == {
        "Year Ended Jan 26, 2025": "130,497",
        "Year Ended Jan 28, 2024": "60,922",
    }
    assert rows["Operating expenses"] == {}  # group heading kept, no values
    assert "Consolidated Statements of Income" in table.title
    assert table.units.lower() == "in millions"


def test_rendered_pieces_repeat_context_heading_units_and_columns():
    (table,) = extract_tables(_NVDA)[0]
    table.rows = table.rows * 30  # force a split
    pieces = render_table(
        table, "Financial table from NVDA", 120, lambda s: len(s.split())
    )
    assert len(pieces) > 1
    for piece in pieces:
        head = piece.split("\n")[:5]
        assert head[0] == "Financial table from NVDA"
        assert head[1].startswith("Table: ") and head[2].startswith("Units: ")
        assert head[3].startswith("Basis: ") and head[4].startswith("Columns: ")
        assert len(piece.split()) <= 120


def _write_filing(tmp_path, html):
    d = tmp_path / "0000000000-25-000001"
    d.mkdir()
    (d / "primary-document.html").write_text(html)
    (d / "full-submission.txt").write_text(
        "CONFORMED PERIOD OF REPORT:\t20250630\nFILED AS OF DATE:\t\t20250801\n"
    )
    return d


def test_tables_are_chunked_after_unchanged_prose(tmp_path):
    prose = " ".join(f"Sentence number {i} about the business." for i in range(400))
    html = (
        "<html><body><p>Item 1. Business</p><p>"
        + prose
        + "</p><p>Item 7. Management's discussion</p><p>"
        + prose
        + "</p>"
        + _NVDA.split("<body>")[1]
    )
    d = _write_filing(tmp_path, html)
    ing = EDGARIngester.__new__(EDGARIngester)
    fp = d / "primary-document.html"
    text = ing.parse_filing(fp)
    assert "130,497" not in text  # prose extraction still drops tables
    sections = ing.extract_sections(text, "10-K")
    fd, period = ing.extract_filing_dates(d)
    base = dict(
        ticker="MSFT",
        doc_type="10-K",
        filing_date=fd,
        period_of_report=period,
        source="SEC EDGAR",
        sections=sections,
        file_path=str(fp),
        accession_number=d.name,
    )
    chunker = SemanticChunker({"chunking": {"max_tokens": 400}})
    without = chunker.chunk_document(RawDocument(**base))
    tables = ing.parse_tables(fp, "10-K")
    assert [s for s, _ in tables] == ["item_7"]
    with_tables = chunker.chunk_document(RawDocument(**base, tables=tables))
    prose_part = with_tables[: len(without)]
    assert [(c.chunk_id, c.text) for c in prose_part] == [
        (c.chunk_id, c.text) for c in without
    ]
    table_chunks = with_tables[len(without) :]
    assert table_chunks and all(
        c.section == "item_7" + TABLE_SECTION_SUFFIX
        and c.period_end == date(2025, 6, 30)
        and "reporting period ended 2025-06-30" in c.text
        for c in table_chunks
    )
    assert all(c.period_end == date(2025, 6, 30) for c in with_tables)


def test_missing_period_of_report_is_none_not_the_filing_date(tmp_path):
    d = tmp_path / "f"
    d.mkdir()
    (d / "full-submission.txt").write_text("FILED AS OF DATE:\t\t20250801\n")
    ing = EDGARIngester.__new__(EDGARIngester)
    assert ing.extract_filing_dates(d) == (date(2025, 8, 1), None)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


def test_filtered_dense_search_is_exact_within_the_filter(tmp_path):
    vs = VectorStore(str(tmp_path / "idx"), dim=8)
    vs.load()
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(500, 8)).astype(np.float32)
    ids = [f"c{i}" for i in range(500)]
    vs.add(vecs, ids, {i: "h" for i in ids})
    q = vecs[0]
    # the filtered chunk is far from the query: it ranks well outside the
    # global top k*3, and used to be dropped
    far = int(np.argmin(vecs @ q / np.linalg.norm(vecs, axis=1)))
    res = vs.search(q, k=1, filter_ids={f"c{far}"})
    assert res and res[0][0] == f"c{far}"


def test_bm25_tokens_ignore_punctuation():
    assert HybridRetriever._tokenize(
        "fiscal year 2025? Revenue was $130.5 billion."
    ) == [
        "fiscal",
        "year",
        "2025",
        "revenue",
        "was",
        "130.5",
        "billion",
    ]
    assert HybridRetriever._tokenize("approximately 36,000 employees") == [
        "approximately",
        "36,000",
        "employees",
    ]


def _chunk(cid, text, period_end, doc_type="10-K", filed=None):
    return Chunk(
        chunk_id=cid,
        ticker="NVDA",
        text=text,
        token_count=20,
        doc_type=doc_type,
        source="SEC EDGAR",
        section="item_7",
        date=filed or period_end,
        url=None,
        chunk_index=0,
        total_chunks=1,
        period_end=period_end,
    )


def test_period_scope_adds_the_requested_years_filing_to_the_pool(tmp_path):
    # 30 near-identical FY2024 chunks crowd the unscoped top_k; the single
    # FY2025 chunk only gets into the pool through the scoped pass.
    chunks = [
        _chunk(
            f"nvda_10k_a_{i:04d}",
            f"employees fiscal year headcount {i}",
            date(2024, 1, 28),
        )
        for i in range(30)
    ] + [_chunk("nvda_10k_b_0000", "the workforce grew", date(2025, 1, 26))]
    ms = MetadataStore(str(tmp_path / "m.db"))
    ms.add_chunks(chunks)
    vs = VectorStore(str(tmp_path / "idx"), dim=8)
    vs.load()
    vs.add(
        np.ones((len(chunks), 8), dtype=np.float32),
        [c.chunk_id for c in chunks],
        {c.chunk_id: c.content_hash for c in chunks},
    )
    config = {"retrieval": {"dense_candidates": 5, "sparse_candidates": 5}}
    with patch("alphasignal.embeddings.embedder.OpenAI"):
        emb = Embedder({"embeddings": {}}, EmbeddingCache(str(tmp_path / "c.npy")))
    emb.embed_texts = lambda texts: [np.ones(8, dtype=np.float32) for _ in texts]
    r = HybridRetriever(config, emb, vs, ms)
    q = "How many employees at the end of fiscal year 2025?"
    pool = r.retrieve(q, ticker="NVDA", top_k=5)
    assert "nvda_10k_b_0000" in [c.chunk_id for c in pool]
    assert r.last_diagnostics["period_scope"] == {
        "periods": ["FY2025"],
        "filing_chunks": 1,
    }
    scoped = next(c for c in pool if c.chunk_id == "nvda_10k_b_0000")
    assert "fiscal year 2025" in scoped.header
    unscoped = r.retrieve("How many employees?", ticker="NVDA", top_k=5)
    assert "nvda_10k_b_0000" not in [c.chunk_id for c in unscoped]


def _rc(cid, text, section="item_7"):
    return RetrievedChunk(
        chunk_id=cid,
        ticker="NVDA",
        text=text,
        doc_type="10-K",
        source="SEC EDGAR",
        section=section,
        date=date(2025, 2, 26),
        url=None,
        dense_score=0.0,
        sparse_score=0.0,
        hybrid_score=0.0,
        header="NVDA Form 10-K (annual report); reporting period: fiscal year 2025",
    )


def test_window_reranking_scores_a_chunk_by_its_best_window():
    filler = " ".join(f"Unrelated sentence {i} about supply chains." for i in range(60))
    long_chunk = _rc("long", filler + " Revenue was $130.5 billion.")
    other = _rc("other", "Revenue is discussed elsewhere.")
    table = _rc("tab", "Table: x\n" + filler, section="item_7" + TABLE_SECTION_SUFFIX)
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.window_words = 60
    seen = []

    def predict(pairs):
        seen.extend(pairs)
        return [
            5.0 if "130.5" in p else (1.0 if "elsewhere" in p else 0.0)
            for _, p in pairs
        ]

    reranker.model = SimpleNamespace(predict=predict)
    out = reranker.rerank("revenue fiscal 2025", [other, long_chunk, table], top_k=None)
    assert [c.chunk_id for c in out][0] == "long"
    assert all(p.startswith("NVDA Form 10-K") for _, p in seen)  # header leads
    assert sum(1 for _, p in seen if "Table: x" in p) == 1  # tables not split


def test_prompt_shows_reporting_period_and_number_rules():
    with patch("alphasignal.generation.generator.OpenAI"):
        g = RAGGenerator({"generation": {}})
    system, user = g.build_prompt("q?", [_rc("a", "text")])
    assert (
        "Document: NVDA Form 10-K (annual report); reporting period: fiscal year 2025"
        in user
    )
    assert "never from the filing date" in system
    assert "Do not derive new figures" in system


# --------------------------------------------------------------------------
# Backfill script and sentiment isolation
# --------------------------------------------------------------------------


def test_reporting_period_backfill_is_atomic_and_keeps_a_backup(tmp_path):
    db = tmp_path / "metadata.db"
    ms = MetadataStore(str(db))
    c = _chunk("nvda_10k_x_0000", "text", None, filed=date(2025, 2, 26))
    c.accession_number = "0001045810-25-000023"
    ms.add_chunks([c])
    ms.engine.dispose()
    raw = tmp_path / "raw" / "sec-edgar-filings" / "NVDA" / "10-K" / c.accession_number
    raw.mkdir(parents=True)
    (raw / "full-submission.txt").write_text(
        "CONFORMED PERIOD OF REPORT:\t20250126\nFILED AS OF DATE:\t\t20250226\n"
    )
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "alphasignal.scripts.migrate_reporting_period",
            "--db",
            str(db),
            "--edgar-raw",
            str(tmp_path / "raw"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"chunks_updated": 1' in out.stdout
    assert MetadataStore(str(db)).get_chunk(c.chunk_id).period_end == date(2025, 1, 26)
    backups = list(tmp_path.glob("metadata.db.pre-period-*"))
    assert len(backups) == 1
    assert MetadataStore(str(backups[0])).get_chunk(c.chunk_id).period_end is None
    assert not list(tmp_path.glob("*.period-tmp"))


def test_table_chunks_are_never_sentiment_evidence():
    from alphasignal.generation.sentiment import SentimentExtractor

    ext = SentimentExtractor.__new__(SentimentExtractor)
    t = _chunk("nvda_10k_t_0000", "Revenue: 2025 = 1", date.today())
    t.section = "item_7" + TABLE_SECTION_SUFFIX
    t.attribution = "filer"
    assert ext.aggregate("NVDA", [t]).status == "no_data"


def test_ingest_can_skip_news():
    from alphasignal.ingestion.pipeline import IngestionPipeline

    p = IngestionPipeline.__new__(IngestionPipeline)
    p.ingest_ticker_edgar = lambda *a, **k: ["doc"]
    calls = []
    p.ingest_ticker_news = lambda t: calls.append(t) or ["article"]
    assert p.ingest_ticker("NVDA", include_news=False) == (["doc"], [])
    assert calls == []
    assert p.ingest_ticker("NVDA") == (["doc"], ["article"])


def test_missing_period_falls_back_to_next_years_comparatives(tmp_path):
    chunks = [
        _chunk("nvda_10k_c_0000", "fiscal 2024 report", date(2024, 1, 28)),
        _chunk("nvda_10k_d_0000", "fiscal 2025 report", date(2025, 1, 26)),
    ]
    ms = MetadataStore(str(tmp_path / "m.db"))
    ms.add_chunks(chunks)
    r = HybridRetriever({}, None, None, ms)
    assert r._period_scope("revenue in fiscal year 2025?", "NVDA") == (
        ["FY2025"],
        {"nvda_10k_d_0000"},
    )
    assert r._period_scope("revenue in fiscal year 2023?", "NVDA") == (
        ["FY2023 (via FY2024 comparatives)"],
        {"nvda_10k_c_0000"},
    )
    assert r._period_scope("revenue in fiscal year 2021?", "NVDA") == (
        ["FY2021"],
        set(),
    )


# JPM preferred-stock style: header rows on different grids, text values
# (issue date, floating rate) between numbers, one heading over 3 values.
_MIXED = """<html><body><p>Preferred stock summary</p><table>
<tr><td colspan="3"></td><td colspan="6">Shares</td><td colspan="6">Carrying value (in millions)</td>
    <td colspan="3">Issue date</td><td colspan="9">Dividend declared per share</td></tr>
<tr><td colspan="3"></td><td colspan="3">2024</td><td colspan="3">2023</td>
    <td colspan="3">2024</td><td colspan="3">2023</td></tr>
<tr><td colspan="3">Series R</td><td colspan="2">—</td><td></td><td colspan="2">150,000</td><td></td>
    <td colspan="2">—</td><td></td><td colspan="2">1,500</td><td></td><td colspan="3">7/29/2013</td>
    <td colspan="3">228.30</td><td colspan="3">221.70</td><td colspan="3">528.30</td></tr>
<tr><td colspan="3">Series S</td><td colspan="2">200,000</td><td></td><td colspan="2">200,000</td><td></td>
    <td colspan="2">2,000</td><td></td><td colspan="2">2,000</td><td></td><td colspan="3">1/22/2014</td>
    <td colspan="3">NA</td><td colspan="3">1.00</td><td colspan="3">(c)(f)</td></tr>
</table></body></html>"""


def test_unalignable_table_keeps_every_cell_and_asserts_no_period():
    (table,) = extract_tables(_MIXED)[0]
    assert table.aligned is False
    rows = {label: [v for _, v in vals] for label, vals in table.rows}
    assert rows["Series R"] == [
        "—",
        "150,000",
        "—",
        "1,500",
        "7/29/2013",
        "228.30",
        "221.70",
        "528.30",
    ]
    text = render_table(table, "ctx", 400, lambda s: len(s.split()))[0]
    assert "NOT ALIGNED" in text and "2024 = " not in text
    assert "Series R: — | 150,000 | — | 1,500 | 7/29/2013 | 228.30" in text


def test_placeholders_and_trailing_comments_do_not_break_alignment():
    html = """<table><tr><td></td><td>2025</td><td>2024</td><td>Change</td></tr>
    <tr><td>Net income</td><td>120,067</td><td>72,880</td><td>Up 65%</td></tr>
    <tr><td>Ratio</td><td>NM</td><td>1.2</td><td>NA</td></tr></table>"""
    (table,) = extract_tables(html)[0]
    assert table.aligned is True
    assert dict(table.rows)["Net income"][:2] == [
        ("2025", "120,067"),
        ("2024", "72,880"),
    ]
