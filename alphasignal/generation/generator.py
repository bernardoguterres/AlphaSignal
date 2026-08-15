"""RAG answer generation module."""

import logging
import re

from openai import OpenAI

from alphasignal.generation import GenerationResult
from alphasignal.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


class RAGGenerator:
    """Generates answers using retrieved context and LLM."""

    SYSTEM_MESSAGE = """You are a financial research assistant with access to SEC filings and financial news. Answer questions accurately using only the provided context. Always cite your sources using [Source N] notation. If the context does not contain enough information to answer, say so explicitly - do not speculate."""

    def __init__(self, config: dict):
        """Initialize RAG generator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        generation_config = config.get("generation", {})

        # Initialize OpenAI client
        self.client = OpenAI()

        # Store generation parameters
        self.model = generation_config.get("model", "gpt-5.6-luna")
        self.max_tokens = generation_config.get("max_tokens", 1000)
        self.temperature = generation_config.get("temperature", 0.1)

        logger.info(f"Initialized RAGGenerator with model={self.model}")

    def build_prompt(self, query: str, chunks: list[RetrievedChunk]) -> tuple[str, str]:
        """Build system and user messages for LLM.

        Args:
            query: User's question
            chunks: Retrieved context chunks

        Returns:
            Tuple of (system_message, user_message)
        """
        system_message = self.SYSTEM_MESSAGE

        # Build context section
        context_parts = []
        for idx, chunk in enumerate(chunks, start=1):
            context_parts.append(
                f"[Source {idx}]\n"
                f"Ticker: {chunk.ticker}\n"
                f"Date: {chunk.date}\n"
                f"Type: {chunk.doc_type}\n"
                f"Source: {chunk.source}\n"
                f"Text: {chunk.text}\n"
            )

        context_section = "\n".join(context_parts)

        # Build user message
        user_message = (
            f"Context:\n\n{context_section}\n" f"Question: {query}\n\n" f"Answer:"
        )

        return system_message, user_message

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        """Generate answer using retrieved chunks.

        Args:
            query: User's question
            chunks: Retrieved context chunks

        Returns:
            GenerationResult with answer and citations
        """
        if not chunks:
            logger.warning("No chunks provided for generation")
            return GenerationResult(
                answer="I don't have enough information to answer this question.",
                cited_chunks=[],
                prompt_tokens=0,
                completion_tokens=0,
                model=self.model,
            )

        logger.info(
            f"Generating answer for query: '{query[:50]}...' with {len(chunks)} chunks"
        )

        # Build prompt
        system_message, user_message = self.build_prompt(query, chunks)

        # Call OpenAI API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                max_completion_tokens=self.max_tokens,
                # self.temperature (config.yaml's generation.temperature,
                # default 0.1) omitted: gpt-5.6-luna rejects any non-default
                # temperature (reasoning-tier restriction, same as
                # sentiment.py) - only the model's default (1) is accepted.
            )

            # Extract answer
            answer = response.choices[0].message.content or ""

            # Parse citations
            answer, cited_chunks = self._parse_citations(answer, chunks)

            # Get token usage
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

            logger.info(f"Generated answer with {len(cited_chunks)} citations")

            return GenerationResult(
                answer=answer,
                cited_chunks=cited_chunks,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=self.model,
            )

        except Exception as e:
            logger.error(f"Error generating answer: {e}", exc_info=True)
            return GenerationResult(
                answer="An error occurred while generating the answer.",
                cited_chunks=[],
                prompt_tokens=0,
                completion_tokens=0,
                model=self.model,
            )

    def _parse_citations(
        self, answer: str, chunks: list[RetrievedChunk]
    ) -> tuple[str, list[RetrievedChunk]]:
        """Parse [Source N] citations from answer and make the answer text
        internally consistent with the returned citations list.

        Citation integrity fix (Prompt 2.5 remediation, 2026-08-15 - see
        FINAL_ENGINEERING_AUDIT.md / END_TO_END_VALIDATION.md issue 4).
        Two defects existed here previously:

        1. A [Source N] marker referencing an out-of-range N (beyond
           len(chunks) - the model citing more sources than were actually
           retrieved) was silently dropped from cited_chunks, but the
           literal "[Source N]" text was never removed from the answer -
           so the response could contain a citation marker the citations
           array could not resolve.
        2. Even for in-range citations, cited_chunks was built in
           order-of-first-appearance in the text, not renumbered - so if
           the model cited "[Source 3] ... [Source 1]", citations[0] (the
           first chunk in the list) corresponded to the text's "[Source 3]",
           not "[Source 1]": positional correlation between text markers
           and the returned array wasn't actually guaranteed for any
           response, not just malformed ones.

        Fix: every valid, resolvable citation is renumbered sequentially in
        first-appearance order (so cited_chunks[i] always corresponds to
        the answer text's "[Source i+1]" after this rewrite - a real
        invariant, not just true "usually"). Every out-of-range or
        otherwise unresolvable [Source N] marker is removed from the answer
        text entirely - not fabricated a chunk to satisfy, not left dangling.
        Repeated references to the same source correctly reuse that
        source's single renumbered slot rather than duplicating it.

        Args:
            answer: Generated answer text
            chunks: List of context chunks

        Returns:
            Tuple of (answer, cited_chunks) - the returned answer's every
            "[Source N]" marker resolves to cited_chunks[N-1] and nothing
            else; the returned answer never references an N beyond
            len(cited_chunks).
        """
        citation_pattern = r"\[Source (\d+)\]"

        cited_chunks: list[RetrievedChunk] = []
        seen_indices: dict[int, int] = {}  # 0-based chunk_index -> new 1-based number
        renumber_map: dict[int, int] = {}  # original source_num -> new 1-based number

        for match in re.finditer(citation_pattern, answer):
            source_num = int(match.group(1))
            chunk_index = source_num - 1  # Convert to 0-based index

            if not (0 <= chunk_index < len(chunks)):
                continue  # unresolvable - no mapping entry, removed below

            if chunk_index not in seen_indices:
                cited_chunks.append(chunks[chunk_index])
                seen_indices[chunk_index] = len(cited_chunks)
            renumber_map[source_num] = seen_indices[chunk_index]

        def _rewrite(match: re.Match) -> str:
            source_num = int(match.group(1))
            new_num = renumber_map.get(source_num)
            if new_num is None:
                return ""  # unresolvable reference: removed, never fabricated
            return f"[Source {new_num}]"

        cleaned_answer = re.sub(citation_pattern, _rewrite, answer)
        # Collapse whitespace left behind by any removed marker.
        cleaned_answer = re.sub(r"[ \t]{2,}", " ", cleaned_answer).strip()

        return cleaned_answer, cited_chunks
