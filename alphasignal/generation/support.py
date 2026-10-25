"""Lexical citation-support check for generated answers.

Answers whether each cited sentence is *lexically* backed by the passages
it cites - separate from whether its [Source N] markers resolve (that is
_parse_citations' job). It is a screening heuristic, not entailment: it
flags claims to inspect, it cannot prove a claim true.

Per sentence:
  - "number_not_in_source": a figure in the sentence (e.g. 391,035 / 12.5% /
    $1.2 billion / 2024) does not appear in any cited passage - the most
    common way a fluent, cited answer is unsupported.
  - "weak_overlap": under 40% of the sentence's content words occur in the
    cited passages.
  - "supported_lexically": otherwise.
  - "uncited_claim": a sentence containing a figure but no citation.
"""

from __future__ import annotations

import re

_CITE = re.compile(r"\[Source (\d+)\]")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"[a-z][a-z\-]{4,}")
_STOP = {
    "which",
    "their",
    "there",
    "these",
    "those",
    "would",
    "could",
    "should",
    "about",
    "other",
    "while",
    "where",
    "being",
    "based",
    "after",
    "before",
    "source",
    "sources",
    "company",
    "according",
    "including",
    "however",
}


def _numbers(text: str) -> set[str]:
    out = set()
    for m in _NUM.findall(text):
        n = m.replace(",", "").rstrip(".")
        if n:
            out.add(n.lstrip("0") or "0")
    return out


def assess_support(answer: str, cited_texts: list[str]) -> list[dict]:
    """cited_texts[i] is the text of [Source i+1] in the (renumbered) answer."""
    results = []
    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z\-*•])", answer.strip()):
        if not sentence.strip():
            continue
        refs = sorted({int(n) for n in _CITE.findall(sentence)})
        claim = _CITE.sub("", sentence)
        nums = _numbers(claim)
        if not refs:
            if nums:
                results.append(
                    {
                        "sentence": sentence.strip(),
                        "sources": [],
                        "verdict": "uncited_claim",
                        "missing_numbers": sorted(nums),
                    }
                )
            continue
        src = " ".join(
            cited_texts[r - 1] for r in refs if 0 < r <= len(cited_texts)
        ).lower()
        src_nums = _numbers(src)
        missing = sorted(n for n in nums if n not in src_nums)
        words = {w for w in _WORD.findall(claim.lower()) if w not in _STOP}
        overlap = (sum(1 for w in words if w in src) / len(words)) if words else 1.0
        verdict = (
            "number_not_in_source"
            if missing
            else "weak_overlap" if overlap < 0.4 else "supported_lexically"
        )
        results.append(
            {
                "sentence": sentence.strip(),
                "sources": refs,
                "verdict": verdict,
                "missing_numbers": missing,
                "word_overlap": round(overlap, 2),
            }
        )
    return results
