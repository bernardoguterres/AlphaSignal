"""Reranking module using cross-encoder models."""

import logging
import re

from sentence_transformers import CrossEncoder

from alphasignal.ingestion.tables import TABLE_SECTION_SUFFIX
from alphasignal.retrieval import RetrievedChunk

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks retrieved chunks using cross-encoder for better relevance."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        window_words: int = 0,
    ):
        """Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace cross-encoder model name
            window_words: When > 0, score each chunk as the best of its
                overlapping sentence windows of about this many words
                (MaxP passage scoring) instead of as one block. 0 keeps
                whole-chunk scoring.
        """
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.window_words = window_words
        logger.info("Cross-encoder model loaded successfully")

    def _windows(self, chunk: RetrievedChunk) -> list[str]:
        """Texts to score for one chunk, each led by its source header.

        A filing chunk often covers several topics (~2,000 characters), and
        one relevant sentence among unrelated ones scored low as a block.
        Windows of consecutive sentences, half-overlapping, let that
        sentence be judged on its own; the chunk keeps its best window.
        Table chunks are scored whole (their heading/columns must stay with
        the rows), as is anything that already fits one window.
        """
        head = f"{chunk.header}\n" if chunk.header else ""
        text = chunk.text
        if chunk.lane == "table":
            return self._row_windows(chunk, head)
        if (
            self.window_words <= 0
            or (chunk.section or "").endswith(TABLE_SECTION_SUFFIX)
            or len(text.split()) <= self.window_words
        ):
            return [head + text]
        sentences = [s for s in _SENTENCE.split(text) if s.strip()]
        windows, start = [], 0
        while start < len(sentences):
            words, end = 0, start
            while end < len(sentences) and (
                words == 0 or words + len(sentences[end].split()) <= self.window_words
            ):
                words += len(sentences[end].split())
                end += 1
            windows.append(head + " ".join(sentences[start:end]))
            if end >= len(sentences):
                break
            start = max(start + 1, (start + end) // 2)
        return windows

    @staticmethod
    def _row_windows(chunk: RetrievedChunk, head: str) -> list[str]:
        """A table-lane chunk is scored by its best row, each row read as one
        statement with the table heading and units ("Table: ... (Units: ...):
        Net income: Year Ended June 30, 2025 = 101,832") and, when a period
        was requested, only that period's column(s). A whole table block of
        "label: column = value" lines scored poorly with an MS MARCO model
        trained on prose."""
        lines = chunk.text.split("\n")
        title = next((ln for ln in lines if ln.startswith("Table: ")), "")
        units = next((ln for ln in lines if ln.startswith("Units: ")), "")
        rows = [
            ln
            for ln in lines
            if " = " in ln and not ln.startswith(("Table:", "Columns:"))
        ]
        out = []
        for row in rows:
            label, _, rest = row.partition(": ")
            cells = rest.split("; ")
            if chunk.match_columns:
                cells = [
                    c for c in cells if c.split(" = ", 1)[0] in chunk.match_columns
                ]
                if not cells:
                    continue
            out.append(f"{head}{title} ({units}): {label}: " + "; ".join(cells))
        return out or [head + chunk.text]

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Rerank chunks by computing cross-encoder relevance scores.

        Args:
            query: Query text
            chunks: List of retrieved chunks to rerank
            top_k: Number of top results to return (None = return all)

        Returns:
            Reranked list of chunks sorted by final_score
        """
        if not chunks:
            return []

        logger.info(f"Reranking {len(chunks)} chunks")

        # Query-passage pairs. The source header (form, reporting period,
        # filing date) leads every passage so the model can tell a
        # fiscal-2025 passage from its fiscal-2024 twin.
        owners, pairs = [], []
        for i, chunk in enumerate(chunks):
            for window in self._windows(chunk):
                owners.append(i)
                pairs.append((query, window))

        # Compute relevance scores; a chunk scores as its best window
        window_scores = self.model.predict(pairs)
        scores = [float("-inf")] * len(chunks)
        for i, score in zip(owners, window_scores):
            scores[i] = max(scores[i], float(score))

        # Set final_score on each chunk
        for chunk, score in zip(chunks, scores):
            chunk.final_score = float(score)

        # Sort by final_score
        reranked = sorted(chunks, key=lambda x: x.final_score or 0.0, reverse=True)

        # Return top k if specified
        if top_k is not None:
            reranked = reranked[:top_k]

        logger.info(f"Reranking complete, returning {len(reranked)} chunks")
        return reranked


def compose_context(
    reranked: list[RetrievedChunk],
    top_k: int,
    table_slots: int = 1,
    prefer_bases: set[str] | None = None,
) -> list[RetrievedChunk]:
    """Final evidence with a fixed budget per lane: the best ``table_slots``
    period-matched table chunk(s), when the table lane has any, and the best
    prose for the remaining slots; kept in reranker order.

    Scores are not compared across lanes: row-wise table scores beat useful
    prose (a whole subscriber-count passage was pushed out by two tables)
    while unrelated prose beat the matched table (2026-10-21 development
    runs). With no table candidates this is the plain top_k."""
    lane = [c for c in reranked if c.lane == "table"]
    # a table on the basis the question asks for (default: explicitly
    # reported) is preferred over one whose basis is mixed or not stated
    prefer = prefer_bases or {"reported"}
    tables = (
        [c for c in lane if {b.strip() for b in c.basis.split(" / ")} & prefer] or lane
    )[:table_slots]
    prose = [c for c in reranked if c.lane != "table"][: top_k - len(tables)]
    keep = {id(c) for c in tables + prose}
    return [c for c in reranked if id(c) in keep]
