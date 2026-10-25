"""Reporting-period handling: fiscal labels for filings and periods named in queries.

A filing's *reporting period* (SEC "CONFORMED PERIOD OF REPORT", stored as
``period_end``) is distinct from its *filing date* (``date``): NVIDIA's
fiscal-2025 10-K covers the year ended 2025-01-26 but was filed 2025-02-26,
and its fiscal-2026 10-K reports fiscal 2025 only as a comparative.

Fiscal years are named by the calendar year in which they end (NVIDIA's
year ending January 2025 is fiscal 2025; Microsoft's ending June 2025 is
fiscal 2025). That is the SEC/issuer convention for every ticker in this
corpus; a company whose fiscal year ends in the first days of January and
is named for the prior year would be mislabelled by one year.
"""

import re
from datetime import date, timedelta

# A 52/53-week fiscal year ends on a weekday near a fixed date, so its end
# moves by up to a week from year to year.
_FYE_TOLERANCE = timedelta(days=7)

_QUARTER_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4}
_FY = re.compile(r"\b(?:fiscal(?:\s+year)?|fy)\s*'?(\d{4}|\d{2})\b", re.IGNORECASE)
_QUARTER = re.compile(
    r"\b(?:q([1-4])|(first|second|third|fourth)[\s-]+(?:fiscal[\s-]+)?quarter)\b",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")


def fiscal_label(
    doc_type: str, period_end: date | None, fye: tuple[int, int] | None
) -> tuple[int, str] | None:
    """(fiscal_year, "FY" | "Q1".."Q3") for a filing, or None if unknown.

    Args:
        doc_type: "10-K" or "10-Q" (anything else has no fiscal label).
        period_end: The filing's period of report.
        fye: The company's fiscal year end as (month, day), taken from its
            own 10-K period ends.
    """
    if period_end is None or fye is None or doc_type not in ("10-K", "10-Q"):
        return None
    month, day = fye
    this_end = _safe_date(period_end.year, month, day)
    if period_end <= this_end + _FYE_TOLERANCE:
        fiscal_year, prev_end = period_end.year, _safe_date(
            period_end.year - 1, month, day
        )
    else:
        fiscal_year, prev_end = period_end.year + 1, this_end
    if doc_type == "10-K":
        return fiscal_year, "FY"
    quarter = round((period_end - prev_end).days / 91.3)
    return fiscal_year, f"Q{min(max(quarter, 1), 3)}"


def _safe_date(year: int, month: int, day: int) -> date:
    try:
        return date(year, month, day)
    except ValueError:  # 29 Feb in a non-leap year
        return date(year, month, day - 1)


def format_period(label: tuple[int, str] | None) -> str | None:
    if label is None:
        return None
    year, part = label
    return f"fiscal year {year}" if part == "FY" else f"{part} of fiscal year {year}"


def requested_periods(query: str, fye: tuple[int, int] | None) -> list[tuple[int, str]]:
    """Fiscal periods a question explicitly names, as (fiscal_year, part).

    "fiscal year 2025" / "fiscal 2025" / "FY2025" -> (2025, "FY"); "the
    first quarter of fiscal 2025" / "Q1 FY2025" -> (2025, "Q1"); a fourth
    quarter maps to the annual report (there is no Q4 10-Q). A bare
    calendar year ("in 2024") is read as that fiscal year only when the
    company's fiscal year is the calendar year; otherwise it is ambiguous
    and nothing is returned.
    """
    years = [int(y) if len(y) == 4 else 2000 + int(y) for y in _FY.findall(query)]
    if not years and fye is not None and fye[0] == 12:
        years = [int(y) for y in _YEAR.findall(query)]
    if not years:
        return []
    quarters = [
        int(num) if num else _QUARTER_WORDS[word.lower()]
        for num, word in _QUARTER.findall(query)
    ]
    out: list[tuple[int, str]] = []
    for year in dict.fromkeys(years):
        if quarters:
            for q in dict.fromkeys(quarters):
                out.append((year, "FY" if q == 4 else f"Q{q}"))
        else:
            out.append((year, "FY"))
    return out


def source_header(chunk, fye: tuple[int, int] | None) -> str:
    """One-line identity of a chunk's source document, reporting period
    first, filing/publication date separately."""
    period_end = getattr(chunk, "period_end", None)
    if chunk.doc_type in ("10-K", "10-Q"):
        kind = "annual report" if chunk.doc_type == "10-K" else "quarterly report"
        parts = [f"{chunk.ticker} Form {chunk.doc_type} ({kind})"]
        period = format_period(fiscal_label(chunk.doc_type, period_end, fye))
        if period and period_end:
            parts.append(f"reporting period: {period}, ended {period_end.isoformat()}")
        elif period_end:
            parts.append(f"reporting period ended {period_end.isoformat()}")
        else:
            parts.append("reporting period: unknown")
        parts.append(f"filed {chunk.date.isoformat()}")
    else:
        parts = [
            f"{chunk.ticker} {chunk.doc_type}",
            f"published {chunk.date.isoformat()}",
        ]
    return "; ".join(parts)


_MONTHS = {
    m: i
    for i, m in enumerate(
        (
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ),
        start=1,
    )
}
_DATE_IN_LABEL = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})",
    re.IGNORECASE,
)
_QUARTER_LABEL = re.compile(
    r"\b([1-4])Q(\d{2})\b|\bQ([1-4])\s*'?(\d{2,4})\b", re.IGNORECASE
)
_SUBYEAR = re.compile(r"\b(three|six|nine)\s+months\b|\bquarter\b", re.IGNORECASE)


def expected_period_end(
    year: int, part: str, fye: tuple[int, int] | None
) -> date | None:
    """Approximate end date of a fiscal period (exact to within the 52/53-week
    tolerance) from the company's fiscal year end."""
    if fye is None:
        return None
    end = _safe_date(year, *fye)
    if part == "FY":
        return end
    q = int(part[1])
    return end - timedelta(days=round((4 - q) * 91.3))


def column_matches(label: str, year: int, part: str, end: date | None) -> bool:
    """Whether a table column heading denotes the requested fiscal period:
    a date within a week of the period end (with a three/six/nine-month
    qualifier only for quarters), a quarter label ("2Q26", "Q2 2026"), or a
    bare fiscal-year heading ("2025") for an annual period."""
    text = label.strip()
    sub_year = bool(_SUBYEAR.search(text))
    for mon, day, yr in _DATE_IN_LABEL.findall(text):
        try:
            d = date(int(yr), _MONTHS[mon[:3].lower()], int(day))
        except ValueError:
            continue
        if end and abs((d - end).days) <= 7:
            if part == "FY" and not sub_year:
                return True
            if part != "FY" and (
                re.search(r"\bthree\s+months\b|\bquarter\b", text, re.IGNORECASE)
                or "months" not in text.lower()
            ):
                return True  # a six/nine-month column is not the quarter
    m = _QUARTER_LABEL.search(text)
    if m and part != "FY":
        q = int(m.group(1) or m.group(3))
        yr = int(m.group(2) or m.group(4))
        yr = yr + 2000 if yr < 100 else yr
        return (yr, f"Q{q}") == (year, part)
    return part == "FY" and text == str(year)
