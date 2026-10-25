"""Financial-table extraction from SEC filing HTML.

Prose extraction (EDGARIngester.parse_filing) drops every <table>, so
statement figures that appear only in tables were absent from the corpus.
This module turns numeric tables into text that keeps what a reader needs
to use a figure correctly: the table's heading, its units, and the column
heading (usually the period) of every value, e.g.

    Revenue: Year Ended Jan 26, 2025 = $130,497; Year Ended Jan 28, 2024 = $60,922

Layout tables (no numeric rows) are skipped. Each table is rendered on its
own and split by rows when long, with the heading/units/columns repeated in
every piece, so a chunk never carries a number without its period.
"""

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, NavigableString

TABLE_SECTION_SUFFIX = "_table"
_VALUE = re.compile(
    r"^\(?\$?\s?-?[\d,]*\.?\d+\)?%?$|^[—–-]$"
    r"|^(?:NM|NA|N/A|n/a|nm|#)$"  # placeholders occupy a value column
)
_UNITS = re.compile(
    r"\(?\b(?:(?:dollars|amounts|shares|in)\s+(?:in\s+)?)"
    r"(?:millions|thousands|billions)\b[^)\n]{0,60}\)?",
    re.IGNORECASE,
)
_SPACES = re.compile(r"\s+")
_FOOTNOTE = re.compile(r"^(?:\([a-z]{1,2}\))+$")  # "(a)", "(c)(f)"
_YEAR_CELL = re.compile(r"^(?:19|20)\d\d$")
# A column heading naming a period: a year, a quarter ("2Q26", "Q3 2025"),
# or a date ("Jan 26, 2025", "December 31, 2024").
_PERIOD = re.compile(
    r"\b(?:19|20)\d\d\b|\b[1-4]Q\d\d\b|\bQ[1-4]\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d",
    re.IGNORECASE,
)
_PARENT = re.compile(r"(?:months|weeks|year|quarter)s?\s+ended", re.IGNORECASE)


def _parent_text(text: str) -> str:
    """'(In millions) Year Ended June 30,' -> 'Year Ended June 30,'."""
    t = re.sub(r"\([^)]*\)", "", text).strip(" ,;")
    return t + "," if t[-1:].isdigit() else t


def _is_period(text: str) -> bool:
    return len(text) <= 40 and bool(_PERIOD.search(text)) and not _UNITS.search(text)


@dataclass
class ExtractedTable:
    """One numeric table: heading, units, column labels and labelled rows."""

    index: int  # position among all <table> elements in the document
    title: str
    units: str
    columns: list[str]
    # (row label, [(column label, value), ...]); a row with no values is a
    # group heading (e.g. "Operating expenses").
    rows: list[tuple[str, list[tuple[str, str]]]] = field(default_factory=list)
    # False when values could not be matched to headings reliably (header
    # rows on different column grids, text cells mixed into the values):
    # rows then carry every data cell in order with no column label, and
    # the headings are kept verbatim instead of being asserted per value.
    aligned: bool = True
    header_lines: list[str] = field(default_factory=list)
    # Accounting basis of the figures (see table_basis): "reported",
    # "PRO FORMA", "NON-GAAP", "MANAGED BASIS", "CONSTANT CURRENCY",
    # "MIXED (...)" or "not stated", with the source words it rests on.
    basis: str = "not stated"
    basis_evidence: str = ""


def _text(node) -> str:
    return _SPACES.sub(" ", node.get_text(" ", strip=True)).strip()


def _cells(tr) -> list[tuple[int, int, str]]:
    """(start column, span, text) for each cell of a row."""
    out, col = [], 0
    for td in tr.find_all(["td", "th"], recursive=False) or tr.find_all(["td", "th"]):
        try:
            span = max(int(td.get("colspan", 1) or 1), 1)
        except ValueError:
            span = 1
        out.append((col, span, _text(td)))
        col += span
    return out


def _merge_values(cells: list[tuple[int, int, str]]) -> list[tuple[int, str]]:
    """Value cells as (column, text): "$" cells dropped, ")" and "%" cells
    joined to the value before them."""
    values: list[tuple[int, str]] = []
    for col, _, txt in cells:
        if not txt or txt == "$":
            continue
        if txt in (")", "%", ")%") and values:
            c, v = values[-1]
            values[-1] = (c, v + txt)
            continue
        if _VALUE.match(txt.replace(" ", "")):
            values.append((col, txt.replace(" ", "")))
    return values


def _heading_before(table, limit: int = 300) -> str:
    """Nearest preceding text (statement title, or the sentence introducing
    the table), closest first, up to `limit` characters."""
    parts: list[str] = []
    for s in table.find_all_previous(string=True, limit=40):
        t = _SPACES.sub(" ", str(s)).strip()
        if not t or (s.parent and s.parent.name in ("script", "style")):
            continue
        if s.find_parent("table") is not None or t.startswith("[[TABLE"):
            break  # reached the previous table
        if parts and len(t) > 160:
            break  # a body paragraph above a short heading is not the title
        parts.append(t)
        if sum(len(p) for p in parts) >= 120 or len(parts) >= 3:
            break
    title = " ".join(reversed(parts))
    return title if len(title) <= limit else "..." + title[-limit:]


def _text_around(table, before: int = 600, after: int = 600) -> tuple[str, str]:
    """Text immediately before (lead-in) and after (footnotes) a table, each
    stopping at the neighbouring table."""
    lead: list[str] = []
    for s in table.find_all_previous(string=True, limit=200):
        t = _SPACES.sub(" ", str(s)).strip()
        if s.find_parent("table") is not None or t.startswith("[[TABLE"):
            break
        if t and not (s.parent and s.parent.name in ("script", "style")):
            lead.append(t)
            if sum(map(len, lead)) >= before:
                break
    tail: list[str] = []
    for s in table.find_all_next(string=True, limit=400):
        if s.find_parent("table") is table:
            continue
        t = _SPACES.sub(" ", str(s)).strip()
        if s.find_parent("table") is not None or t.startswith("[[TABLE"):
            break
        if t:
            tail.append(t)
            if sum(map(len, tail)) >= after:
                break
    footnotes = " ".join(tail)
    # stop at the sentence that introduces the NEXT table ("Following are
    # the ... pro forma results:"), which belongs to that table, not this one
    nxt = _NEXT_TABLE_INTRO.search(footnotes)
    if nxt:
        footnotes = footnotes[: nxt.start()]
    # the lead-in is the table's own introduction: the last two sentences
    # before it (a wider window picked up the previous table's footnote)
    lead_text = " ".join(reversed(lead))[-before:]
    sentences = re.split(r"(?<=[.!?])\s+", lead_text)
    # the sentence that introduces the table ("The following table shows
    # ...:") is the lead-in; without one, the last two sentences
    last = sentences[-1] if sentences else ""
    intro = _INTRO.search(last) or last.rstrip().endswith(":")
    lead_in = last if intro else " ".join(sentences[-2:])
    return lead_in[-400:], footnotes[:after]


_INTRO = re.compile(r"\bfollowing\b|\bas follows\b|\bbelow\b", re.IGNORECASE)


_NEXT_TABLE_INTRO = re.compile(
    r"(?:^|(?<=[.:]\s))(?:The\s+)?following\s+(?:is|are|table)\b|\bas follows\b"
    r"|\bSee accompanying notes\b|\bNOTE\s+\d+\b",
    re.IGNORECASE,
)
# Statement headings are sometimes split mid-word ("INC OME STATEMENTS"), so
# they are also matched with all whitespace removed.
_REPORTED_COMPACT = re.compile(
    r"CONSOLIDATED(?:STATEMENTS?|BALANCESHEETS?)|STATEMENTSOF(?:INCOME|OPERATIONS|EARNINGS|COMPREHENSIVEINCOME|CASHFLOWS)"
    r"|(?:INCOME|CASHFLOWS?|STOCKHOLDERS.?EQUITY)STATEMENTS?|BALANCESHEETS?",
    re.IGNORECASE,
)


# EDGAR HTML writes hyphens as U+2010..U+2015, U+2212 or a soft hyphen;
# basis detection works on text with all of them folded to "-".
_DASHES = str.maketrans(
    {c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u00ad"}
)


def fold_dashes(text: str) -> str:
    return text.translate(_DASHES)


_NON_REPORTED = (
    (
        "PRO FORMA",
        re.compile(
            r"pro[\s-]*forma|as (?:if|though)\b[^.]{0,120}\bhad (?:occurred|taken place|"
            r"been (?:consummated|completed|effective))",
            re.IGNORECASE,
        ),
    ),
    ("NON-GAAP", re.compile(r"non[\s-]*GAAP", re.IGNORECASE)),
    (
        "MANAGED BASIS",
        re.compile(r"managed[\s-]+basis|fully taxable[\s-]+equivalent", re.IGNORECASE),
    ),
    ("CONSTANT CURRENCY", re.compile(r"constant[\s-]+currency", re.IGNORECASE)),
)
# Explicit markers that the figures are the reported (GAAP) results: a
# financial-statement heading or a GAAP / "as reported" label. A bare
# mention of "balance sheet(s)" is not one ("reported in the balance sheets
# as follows", "off-balance sheet arrangements").
_REPORTED = re.compile(
    r"consolidated\s+(?:statements?|balance\s+sheets?)|statements?\s+of\s+(?:income|"
    r"operations|earnings|comprehensive\s+income|cash\s+flows|financial\s+position)|"
    r"income\s+statements?|(?<!non-)(?<!non )\bGAAP\b|\bas reported\b|\breported basis\b",
    re.IGNORECASE,
)


def table_basis(
    lead: str, headers: str, footnotes: str, row_labels: str = ""
) -> tuple[str, str]:
    """Accounting basis of a table's figures, from the table's own lead-in,
    its column headings, its row labels and the footnotes after it - not the
    heading alone.

    * A non-reported basis (pro forma, non-GAAP, managed basis, constant
      currency) in the lead-in, headings or row labels wins; headings that
      also say "reported" make it MIXED (columns differ). Several bases are
      joined with " / ".
    * Otherwise an explicit statement heading or GAAP marker means
      "reported". A footnote that flags some figures as another basis is
      recorded in the evidence ("footnote marks some figures NON-GAAP: ...")
      without demoting the statement, since footnote text can belong to a
      neighbouring note.
    * A footnote flag with no reported heading gives MIXED; nothing at all
      gives "not stated", which answers must qualify.
    """
    lead, headers, footnotes, row_labels = map(
        fold_dashes, (lead, headers, footnotes, row_labels)
    )
    strong, flagged = [], []
    for name, rx in _NON_REPORTED:
        for where, text in (
            ("lead-in", lead),
            ("headings", headers),
            ("row labels", row_labels),
            ("footnote", footnotes),
        ):
            m = rx.search(text)
            if m:
                a, b = max(0, m.start() - 60), min(len(text), m.end() + 60)
                hit = (name, f'{where}: "...{text[a:b].strip()}..."')
                (flagged if where == "footnote" else strong).append(hit)
                break
    names = list(dict.fromkeys(n for n, _ in strong))
    if names:
        if re.search(r"\breported\b", headers, re.IGNORECASE):
            return f"MIXED (reported and {' / '.join(names)} columns)", strong[0][1]
        return " / ".join(names), strong[0][1]
    note = ""
    if flagged:
        fnames = " / ".join(dict.fromkeys(n for n, _ in flagged))
        note = f"; footnote marks some figures {fnames}: {flagged[0][1]}"
    reported = _reported_marker(lead, headers)
    if reported:
        return "reported", reported + note
    if flagged:
        return f"MIXED (footnote marks some figures {fnames})", flagged[0][1]
    return "not stated", ""


def _reported_marker(lead: str, headers: str) -> str:
    for src in (lead[-300:], headers):
        m = _REPORTED.search(src)
        if m:
            a, b = max(0, m.start() - 40), min(len(src), m.end() + 40)
            return f'"...{src[a:b].strip()}..."'
    # statement headings split mid-word ("INC OME STATEMENTS"): only an
    # upper-case heading directly above the table is read with spaces removed
    tail = lead[-80:]
    letters = [ch for ch in tail if ch.isalpha()]
    if letters and sum(ch.isupper() for ch in letters) / len(letters) > 0.6:
        c = _REPORTED_COMPACT.search(re.sub(r"\s+", "", tail))
        if c:
            return f'heading: "{c.group(0)}"'
    return ""


def parse_table(table, index: int) -> ExtractedTable | None:
    """Parse one <table>; None when it is not a numeric table."""
    rows = [_cells(tr) for tr in table.find_all("tr")]
    header_rows: list[list[tuple[int, int, str]]] = []
    data: list[tuple[str, list[tuple[int, str]]]] = []
    label_ends: list[int] = []  # where each data row's label cell ends
    raw_cells: list[list[str]] = []  # every data cell after the label, in order
    mixed = False  # a data row has text values between its numbers
    for cells in rows:
        if not any(txt for _, _, txt in cells):
            continue
        values = _merge_values(cells)
        first_value_col = values[0][0] if values else None
        label_cell = next(
            (
                (col, span, txt)
                for col, span, txt in cells
                if txt
                and re.search(r"[A-Za-z]", txt)
                and (first_value_col is None or col < first_value_col)
            ),
            None,
        )
        label = label_cell[2] if label_cell else ""
        label = label.rstrip(": ").strip()
        if (
            values
            and not any(v for _, v in data)
            and all(_YEAR_CELL.match(v) for _, v in values)
        ):
            header_rows.append(cells)  # a row of years is column headings
        elif values and label:
            data.append((label, values))
            label_ends.append(min(label_cell[0] + label_cell[1], values[0][0]))
            after = []
            for col, _, txt in cells:
                if (
                    col <= label_cell[0]
                    or not txt
                    or txt == "$"
                    or _FOOTNOTE.match(txt)
                ):
                    continue
                if txt in (")", "%", ")%") and after:
                    after[-1] += txt
                else:
                    after.append(txt)
            raw_cells.append(after)
            # text values BETWEEN numeric ones (dates, "SOFR + 3.30") can shift
            # positional matching; a trailing comment ("Up 65%") cannot
            first, last = values[0][0], values[-1][0]
            mixed = mixed or any(
                first < col < last
                and txt
                and txt not in ("$", ")", "%", ")%")
                and not _FOOTNOTE.match(txt)
                and not _VALUE.match(txt.replace(" ", ""))
                for col, _, txt in cells
            )
        elif not data and not values:
            header_rows.append(cells)
        elif label and not values:
            data.append((label, []))  # group heading inside the table body
        elif not data and values and not label:
            header_rows.append(cells)  # e.g. a row of years as column headings
    if sum(1 for _, v in data if v) < 2:
        return None

    value_cols = sorted({c for _, vals in data for c, _ in vals})
    label_end = min(label_ends)  # columns before this hold row labels

    def column_label(col: int, prev_col: int) -> str:
        """Header text covering `col`; failing that, the nearest header cell
        to its left after the previous value column (filings often put a
        year heading one cell left of its "$" and value cells)."""
        parts: list[str] = []
        for cells in header_rows:
            covering = [
                txt for start, span, txt in cells if txt and start <= col < start + span
            ]
            if not covering:
                covering = [
                    txt
                    for start, _, txt in cells
                    if txt and label_end <= start <= col and start > prev_col
                ][-1:]
            for txt in covering:
                if start_is_units(txt) or txt in parts:
                    continue  # units are reported once, in the Units line
                parts.append(txt)
        return " ".join(parts)

    def start_is_units(txt: str) -> bool:
        return bool(_UNITS.fullmatch(txt.strip(" ()$")) or _UNITS.fullmatch(txt))

    labels = {}
    for i, c in enumerate(value_cols):
        labels[c] = column_label(c, value_cols[i - 1] if i else -1) or f"column {i + 1}"

    # Ordinal alignment. Filings are inconsistent about "$"/spacer cells and
    # header rows often sit on a different column grid from the data rows
    # (JPM's period row starts at column 0, its data rows after a 6-column
    # label), so values are matched to the leaf period headings IN ORDER
    # when a row has exactly as many values as there are headings. Parent
    # headings ("Six months ended June 30,") are attached by column
    # coverage when that is consistent, else in equal ordinal blocks.
    leaf_row = max(
        (
            [
                (st, sp, t)
                for st, sp, t in cells
                if t and _is_period(t) and not _PARENT.search(t)
            ]
            for cells in header_rows
        ),
        key=len,
        default=[],
    )
    leaf = [t for _, _, t in leaf_row]
    parents = [
        (st, sp, _parent_text(t))
        for cells in header_rows
        for st, sp, t in cells
        if t and _PARENT.search(t)
    ]
    leaf_full = list(leaf)
    if parents and leaf:
        by_cover = [
            next((t for st, sp, t in parents if st <= l_st < st + sp), None)
            for l_st, _, _ in leaf_row
        ]
        if all(by_cover):
            leaf_full = [f"{p_txt} {t}" for p_txt, t in zip(by_cover, leaf)]
        elif len(leaf) % len(parents) == 0 and len(parents) > 1:
            block = len(leaf) // len(parents)
            leaf_full = [f"{parents[k // block][2]} {t}" for k, t in enumerate(leaf)]
        elif len(parents) == 1:
            # One "... ended" parent: it dates every bare-year column (all of
            # them when the row is only years; when quarters are mixed in,
            # the bare years are the parent's periods).
            p_txt = parents[0][2]
            leaf_full = [f"{p_txt} {t}" if _YEAR_CELL.match(t) else t for t in leaf]

    def row_pairs(vals: list[tuple[int, str]]) -> list[tuple[str, str]]:
        if leaf_full and len(vals) == len(leaf_full):
            return [(leaf_full[k], v) for k, (_, v) in enumerate(vals)]
        return [(labels[c], v) for c, v in vals]

    rendered_rows = [(lab, row_pairs(vals)) for lab, vals in data]
    columns = list(dict.fromkeys(c for _, pairs in rendered_rows for c, _ in pairs))
    if columns == ["Page"]:
        return None  # table of contents
    duplicate = any(
        len({c for c, _ in pairs}) < len(pairs) for _, pairs in rendered_rows
    )
    # a column labelled only with a parent heading ("Financial performance
    # of JPMorganChase") next to the same heading WITH a period ("... 2023")
    # means a period heading was lost and the remaining ones are shifted
    # (JPM's 10-K highlights put 2024's figures under "... 2023")
    parent_only = any(
        c
        and not _PERIOD.search(c)
        and any(
            o != c and o.startswith(c) and _PERIOD.search(o[len(c) :]) for o in columns
        )
        for c in columns
    )
    # some value columns got no heading while others got a period: the
    # headings do not cover the values, so the periods are not trusted
    partial = any(c.startswith("column ") for c in columns) and any(
        _PERIOD.search(c) for c in columns
    )
    aligned = not (mixed or duplicate or parent_only or partial)
    header_lines = [" | ".join(t for _, _, t in cells if t) for cells in header_rows]
    if not aligned:
        cells_iter = iter(raw_cells)
        rendered_rows = [
            (lab, [("", v) for v in next(cells_iter)] if vals else [])
            for lab, vals in data
        ]
        columns = []
    title = _heading_before(table)
    lead, footnotes = _text_around(table)
    basis, basis_evidence = table_basis(
        lead,
        " ".join(t for cells in header_rows for _, _, t in cells),
        footnotes,
        " | ".join(lab for lab, _ in data),
    )
    # Row-label text of the header rows (e.g. "Year ended December 31, (in
    # millions)") says what the columns are; keep it with the title.
    header_label = " ".join(
        t for cells in header_rows for start, _, t in cells if t and start < label_end
    )
    if header_label and header_label not in title:
        title = f"{title} | {header_label}" if title else header_label
    header_text = " ".join(t for cells in header_rows for _, _, t in cells)
    m = _UNITS.search(header_text) or _UNITS.search(title)
    units = m.group(0).strip(" ()") if m else ""
    if not units and (
        all(v.endswith("%") for _, vals in data for _, v in vals)
        or "percentage" in title.lower()
    ):
        units = "percent"
    return ExtractedTable(
        index=index,
        title=title,
        units=units or "not stated",
        columns=columns,
        rows=rendered_rows,
        aligned=aligned,
        header_lines=header_lines,
        basis=basis,
        basis_evidence=basis_evidence,
    )


def extract_tables(html: str) -> tuple[list[ExtractedTable], str]:
    """All numeric tables of a filing, plus the document text with a marker
    ("[[TABLE n]]") where each kept table stood, for locating its section.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()
    tables: list[ExtractedTable] = []
    for i, table in enumerate(soup.find_all("table")):
        if table.find_parent("table") is not None:
            continue  # nested; handled with its outer table
        parsed = parse_table(table, i)
        if parsed is None:
            table.decompose()
            continue
        tables.append(parsed)
        table.replace_with(NavigableString(f" [[TABLE {i}]] "))
    text = _SPACES.sub(" ", soup.get_text()).strip()
    return tables, text


def render_table(
    table: ExtractedTable, context: str, max_tokens: int, count_tokens
) -> list[str]:
    """Text pieces for one table, each within `max_tokens` and each carrying
    the context line, heading, units and column labels."""
    head = [
        context,
        f"Table: {table.title}" if table.title else "Table: (no heading)",
        f"Units: {table.units}",
        f"Basis: {table.basis}"
        + (f" ({table.basis_evidence})" if table.basis_evidence else ""),
        (
            "Columns: " + " | ".join(table.columns)
            if table.aligned
            else "Columns: NOT ALIGNED to values (periods not asserted); headings "
            "as printed: " + " / ".join(table.header_lines)
        ),
    ]
    lines = []
    for label, vals in table.rows:
        if vals and not table.aligned:
            lines.append(f"{label}: " + " | ".join(v for _, v in vals))
        elif vals:
            lines.append(f"{label}: " + "; ".join(f"{c} = {v}" for c, v in vals))
        else:
            lines.append(f"{label}:")
    pieces: list[list[str]] = [[]]
    budget = max_tokens - count_tokens("\n".join(head)) - 8
    used = 0
    for line in lines:
        n = count_tokens(line) + 1
        if pieces[-1] and used + n > budget:
            pieces.append([])
            used = 0
        pieces[-1].append(line)
        used += n
    total = len(pieces)
    out = []
    for k, body in enumerate(pieces, start=1):
        part = [] if total == 1 else [f"(part {k} of {total})"]
        out.append("\n".join(head + part + body))
    return out
