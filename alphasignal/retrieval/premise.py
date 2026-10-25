"""Retrieval query for questions that carry a premise about a measure.

"What was X's net loss in 2024?" ranks passages about losses; when the
company reported net income, the passage that contradicts the premise
("reported net income of $58.5 billion") was never retrieved, and the
answer could only say the figure was not stated (development item M3,
2026-10-21). For retrieval and reranking only, a signed measure is paired
with its neutral form, and a claimed direction ("decline", "fall") with
both directions, so evidence for either outcome can surface. The model
still answers the question exactly as asked.
"""

import re

_SIGNED = (
    (re.compile(r"\bnet\s+loss(es)?\b", re.I), "net income"),
    (re.compile(r"\boperating\s+loss(es)?\b", re.I), "operating income"),
    (re.compile(r"\bloss(es)?\s+per\s+share\b", re.I), "earnings per share"),
    (re.compile(r"\bgross\s+loss\b", re.I), "gross profit"),
)
_DIRECTION = re.compile(
    r"\b(declin\w*|decreas\w*|fall|fell|fallen|drop\w*|shr[ia]nk\w*|contract\w*"
    r"|increas\w*|ris(e|en|ing)|rose|gr[eo]w\w*)\b",
    re.I,
)


def retrieval_query(question: str) -> str:
    """The question plus neutral measure / both-direction terms, for search."""
    extra = [neutral for rx, neutral in _SIGNED if rx.search(question)]
    if _DIRECTION.search(question):
        extra.append("increased decreased compared with the prior year")
    return f"{question} ({'; '.join(extra)})" if extra else question
