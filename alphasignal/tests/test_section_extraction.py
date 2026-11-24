"""S8: section extraction must not drop text or mistake cross-references and
table-of-contents lines for headings (JPMorganChase 10-Qs kept 1-3% of their
text; the 10-K's MD&A was stored as item_8)."""

import tempfile

import pytest

from alphasignal.ingestion.edgar import EDGARIngester


@pytest.fixture
def ing():
    return EDGARIngester({}, download_dir=tempfile.mkdtemp())


def test_text_before_first_heading_is_kept(ing):
    body = "Cover page and highlights. " * 50
    text = (
        body
        + "Item 3. Quantitative and Qualitative Disclosures. market risk text. "
        + "Item 4. Controls and Procedures. controls text."
    )
    secs = ing.extract_sections(text, "10-Q")
    assert secs["front_matter"].startswith("Cover page")
    assert sum(len(v) for v in secs.values()) >= len(text) - 5


def test_mdna_opening_sentence_counts_as_item_2(ing):
    text = (
        "Front. The following is Management’s discussion and analysis of the "
        "financial condition. "
        + "Revenue rose strongly. " * 200
        + "Item 3. Quantitative disclosures. x. Item 4. Controls and Procedures. y."
    )
    secs = ing.extract_sections(text, "10-Q")
    assert "Revenue rose strongly" in secs["item_2"]


def test_cross_references_are_not_headings(ing):
    text = (
        "Item 1. Business. "
        + "We sell things. " * 100
        + "Refer to Part I, Item 1A: Risk Factors on pages 9-31. "
        + "More business text. " * 50
        + "Item 1A. Risk Factors. "
        + "Risk discussion. " * 100
        + "Item 7. Management's discussion. "
        + "MD&A body. " * 100
    )
    secs = ing.extract_sections(text, "10-K")
    assert "More business text" in secs["item_1"]
    assert secs["item_1a"].startswith("Item 1A. Risk Factors.")


def test_table_of_contents_loses_to_the_real_section(ing):
    toc = (
        "Item 1. Business 3 Item 1A. Risk Factors 10 "
        "Item 7. Management's Discussion 40 Item 8. Financial Statements 80 "
    )
    text = (
        toc
        + "Item 1. Business. "
        + "Business body. " * 100
        + "Item 1A. Risk Factors. "
        + "Risk body. " * 100
        + "Item 7. Management's Discussion. "
        + "MD&A body. " * 100
        + "Item 8. Financial Statements. "
        + "Statements. " * 100
    )
    secs = ing.extract_sections(text, "10-K")
    assert "MD&A body" in secs["item_7"]
    assert "Statements." in secs["item_8"]
    assert "MD&A body" not in secs["item_1a"]
    assert secs["front_matter"].startswith("Item 1. Business 3")
