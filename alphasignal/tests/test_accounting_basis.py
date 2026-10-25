"""Accounting basis of table figures and premise-neutral retrieval (2026-10-21)."""

from datetime import date

from alphasignal.ingestion.tables import extract_tables, render_table, table_basis
from alphasignal.retrieval import RetrievedChunk
from alphasignal.retrieval.premise import retrieval_query
from alphasignal.retrieval.reranker import compose_context
from alphasignal.retrieval.retriever import _asks_for_basis, _basis_of

ROWS = """<table><tr><td>(In millions)</td><td>2024</td><td>2023</td></tr>
<tr><td>Revenue</td><td>247,442</td><td>219,790</td></tr>
<tr><td>Net income</td><td>88,308</td><td>71,383</td></tr></table>"""


def _one(html):
    (table,) = extract_tables(html)[0]
    return table


def test_pro_forma_from_the_lead_in_sentence():
    t = _one(
        "<p>Following are the supplemental consolidated financial results of the "
        "Company on an unaudited pro forma basis, as if the acquisition had been "
        "consummated on July 1, 2022:</p>" + ROWS
    )
    assert t.basis == "PRO FORMA" and "lead-in" in t.basis_evidence
    text = render_table(t, "ctx", 400, lambda s: len(s.split()))[0]
    assert "Basis: PRO FORMA (lead-in:" in text


def test_pro_forma_from_the_footnote_alone_is_not_reported():
    t = _one(
        "<p>Supplemental results:</p>" + ROWS + "<p>(a) These pro forma results "
        "were based on estimates and assumptions.</p>"
    )
    assert t.basis.startswith("MIXED (footnote marks some figures PRO FORMA")


def test_next_tables_lead_in_is_not_this_tables_footnote():
    html = (
        "<p>INC OME STATEMENTS</p>" + ROWS + "<p>Following are the unaudited pro "
        "forma results:</p>" + ROWS
    )
    first, second = extract_tables(html)[0]
    assert first.basis == "reported"  # split heading still recognised
    assert second.basis == "PRO FORMA"


def test_headings_with_reported_and_managed_columns_are_mixed():
    assert table_basis("", "Reported FTE adjustments Managed basis", "")[0].startswith(
        "MIXED (reported and MANAGED BASIS"
    )
    assert table_basis("Consolidated Statements of Income", "", "")[0] == "reported"
    assert table_basis("Fiscal Year 2025 Summary", "2025 2024", "")[0] == "not stated"


def test_lane_basis_helpers():
    assert _basis_of("x\nBasis: PRO FORMA (lead-in: ...)\nColumns: a") == "PRO FORMA"
    assert not _asks_for_basis("What was total revenue in fiscal 2024?", "PRO FORMA")
    assert _asks_for_basis("What was pro forma revenue in fiscal 2024?", "PRO FORMA")


def _rc(cid, basis):
    return RetrievedChunk(
        chunk_id=cid,
        ticker="MSFT",
        text="t",
        doc_type="10-K",
        source="SEC EDGAR",
        section="item_8_table",
        date=date(2024, 7, 30),
        url=None,
        dense_score=0,
        sparse_score=0,
        hybrid_score=0,
        lane="table",
        basis=basis,
    )


def test_reserved_table_slot_prefers_an_explicitly_reported_table():
    ranked = [_rc("summary", "not stated"), _rc("statement", "reported")]
    assert [c.chunk_id for c in compose_context(ranked, 5)] == ["statement"]


def test_premise_neutral_retrieval_query():
    q = "What was JPMorganChase's net loss in 2024?"
    assert retrieval_query(q) == q + " (net income)"
    assert "increased decreased" in retrieval_query("Why did revenue decline in 2024?")
    plain = "How many employees did Apple have?"
    assert retrieval_query(plain) == plain


def test_requested_basis_steers_the_reserved_table_slot():
    from alphasignal.retrieval.retriever import requested_bases

    ranked = [_rc("statement", "reported"), _rc("proforma", "PRO FORMA")]
    q = "What would revenue have been on the unaudited pro forma basis?"
    assert requested_bases(q) == {"PRO FORMA"}
    out = compose_context(ranked, 5, prefer_bases=requested_bases(q))
    assert [c.chunk_id for c in out] == ["proforma"]
    assert requested_bases("What was revenue in 2024?") == {"reported"}


def test_shifted_period_headings_are_not_aligned():
    # JPM 10-K highlights layout: the first year heading sits in the label
    # columns, so it was lost and 2024's figures came out under "... 2023"
    html = """<table>
    <tr><td colspan="9">Financial performance of X</td></tr>
    <tr><td colspan="3">Year ended December 31, (in millions)</td></tr>
    <tr><td colspan="3">2024</td><td colspan="3">2023</td><td colspan="3"></td>
        <td colspan="3">Change</td></tr>
    <tr><td colspan="3">Total net revenue</td><td>$</td><td colspan="2">180,593</td>
        <td>$</td><td colspan="2">162,366</td><td></td><td colspan="2"></td>
        <td colspan="3">11</td></tr>
    <tr><td colspan="3">Net income</td><td>$</td><td colspan="2">58,471</td>
        <td>$</td><td colspan="2">49,552</td><td></td><td colspan="2"></td>
        <td colspan="3">18</td></tr></table>"""
    (t,) = extract_tables(html)[0]
    rendered = render_table(t, "ctx", 400, lambda s: len(s.split()))[0]
    assert "2023 = 180,593" not in rendered  # never the shifted period
    assert t.aligned is False or ("2024 = 180,593" in rendered)


def test_headings_that_do_not_cover_all_values_assert_no_period():
    html = """<table><tr><td></td><td colspan="2">Performance</td><td></td></tr>
    <tr><td>(in millions)</td><td>2023</td><td>Change</td></tr>
    <tr><td>Revenue</td><td>180,593</td><td>162,366</td><td>11</td></tr>
    <tr><td>Net income</td><td>58,471</td><td>49,552</td><td>18</td></tr></table>"""
    (t,) = extract_tables(html)[0]
    assert t.aligned is False
    assert "2023 = " not in render_table(t, "ctx", 400, lambda s: len(s.split()))[0]


# --- fresh-review findings (2026-10-21) ---------------------------------


def test_combined_basis_is_kept_out_of_the_lane_and_matches_a_request():
    from alphasignal.retrieval.retriever import _NON_REPORTED_BASES, _basis_parts

    assert _basis_parts("PRO FORMA / NON-GAAP") & _NON_REPORTED_BASES
    assert not _asks_for_basis("What was net income in 2024?", "PRO FORMA / NON-GAAP")
    assert _asks_for_basis("What was pro forma net income?", "PRO FORMA / NON-GAAP")
    out = compose_context(
        [_rc("s", "reported"), _rc("pf", "PRO FORMA / NON-GAAP")],
        5,
        prefer_bases={"PRO FORMA"},
    )
    assert [c.chunk_id for c in out] == ["pf"]


def test_unicode_hyphens_do_not_hide_non_gaap_or_pro_forma():
    for dash in ("‑", "–", "‐"):
        basis, _ = table_basis(
            f"The following table presents non{dash}GAAP results.", "", ""
        )
        assert basis == "NON-GAAP", dash
        assert (
            table_basis(f"Unaudited pro{dash}forma results:", "", "")[0] == "PRO FORMA"
        )
    assert _asks_for_basis("What was non‑GAAP EPS?", "NON-GAAP")


def test_as_though_wording_and_pro_forma_row_labels():
    assert (
        table_basis(
            "The combined results as though the acquisition had been completed on "
            "January 1, 2023 were:",
            "2024 2023",
            "",
        )[0]
        == "PRO FORMA"
    )
    assert (
        table_basis(
            "Supplemental information:",
            "2024 2023",
            "",
            "Pro forma revenue | Pro forma net income",
        )[0]
        == "PRO FORMA"
    )


def test_unrelated_note_after_a_statement_does_not_demote_it():
    basis, evidence = table_basis(
        "CONSOLIDATED STATEMENTS OF INCOME",
        "2024 2023",
        "Pro forma results have not been presented because the effect was immaterial.",
    )
    assert basis == "reported" and "footnote marks some figures PRO FORMA" in evidence


def test_lead_in_is_the_tables_own_introduction():
    basis, _ = table_basis(
        "The following table shows revenue by segment:", "2024 2023", ""
    )
    assert basis == "not stated"
    (t,) = extract_tables(
        "<p>Revenue in constant currency grew 5%. The following table shows revenue "
        "by segment:</p>" + ROWS
    )[0]
    assert t.basis == "not stated"


def test_loose_basis_words_are_not_basis_requests():
    from alphasignal.retrieval.retriever import requested_bases

    assert requested_bases("What was UnitedHealth's managed care revenue?") == {
        "reported"
    }
    assert requested_bases("What was revenue adjusted for the divestiture?") == {
        "reported"
    }
    assert requested_bases("What was adjusted EBITDA in 2024?") == {"NON-GAAP"}
    assert requested_bases("Total net revenue on a managed basis?") == {"MANAGED BASIS"}


def test_a_bare_balance_sheet_mention_is_not_a_reported_marker():
    assert (
        table_basis("Amounts are reported in the balance sheets as follows:", "", "")[0]
        == "not stated"
    )
    assert table_basis("Off-balance sheet arrangements:", "", "")[0] == "not stated"
    assert table_basis("CONSOLIDATED BALANCE SHEETS", "", "")[0] == "reported"
