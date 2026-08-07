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

    def build_prompt(
        self,
        query: str,
        chunks: list[RetrievedChunk]
    ) -> tuple[str, str]:
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
            f"Context:\n\n{context_section}\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        return system_message, user_message

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk]
    ) -> GenerationResult:
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
                model=self.model
            )

        logger.info(f"Generating answer for query: '{query[:50]}...' with {len(chunks)} chunks")

        # Build prompt
        system_message, user_message = self.build_prompt(query, chunks)

        # Call OpenAI API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
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
                model=self.model
            )

        except Exception as e:
            logger.error(f"Error generating answer: {e}", exc_info=True)
            return GenerationResult(
                answer="An error occurred while generating the answer.",
                cited_chunks=[],
                prompt_tokens=0,
                completion_tokens=0,
                model=self.model
            )

    def _parse_citations(
        self,
        answer: str,
        chunks: list[RetrievedChunk]
    ) -> tuple[str, list[RetrievedChunk]]:
        """Parse [Source N] citations from answer.

        Args:
            answer: Generated answer text
            chunks: List of context chunks

        Returns:
            Tuple of (answer, cited_chunks)
        """
        # Find all [Source N] patterns
        citation_pattern = r'\[Source (\d+)\]'
        matches = re.findall(citation_pattern, answer)

        # Map to chunks
        cited_chunks = []
        seen_indices = set()

        for match in matches:
            source_num = int(match)
            chunk_index = source_num - 1  # Convert to 0-based index

            # Check if index is valid and not already added
            if 0 <= chunk_index < len(chunks) and chunk_index not in seen_indices:
                cited_chunks.append(chunks[chunk_index])
                seen_indices.add(chunk_index)

        return answer, cited_chunks
