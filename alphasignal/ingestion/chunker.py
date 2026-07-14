"""Document chunking module."""

import hashlib
import logging
import re

import tiktoken

from alphasignal.ingestion import Chunk, RawArticle, RawDocument

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Chunks documents into token-sized pieces with overlap."""

    def __init__(self, config: dict):
        """Initialize semantic chunker.

        Args:
            config: Configuration dictionary containing chunking settings
        """
        self.config = config

        # Get chunking parameters from config
        chunking_config = config.get("chunking", {})
        self.target_tokens = chunking_config.get("target_tokens", 300)
        self.min_tokens = chunking_config.get("min_tokens", 100)
        self.max_tokens = chunking_config.get("max_tokens", 400)
        self.overlap_tokens = chunking_config.get("overlap_tokens", 50)

        # Initialize tiktoken encoder (cl100k_base - used by OpenAI embedding models)
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if not text:
            return 0
        return len(self.encoder.encode(text))

    def split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences respecting abbreviations.

        Args:
            text: Text to split

        Returns:
            List of sentence strings
        """
        if not text:
            return []

        # Common abbreviations to protect
        abbreviations = [
            "U.S.", "U.K.", "E.U.", "e.g.", "i.e.", "etc.",
            "Dr.", "Mr.", "Mrs.", "Ms.", "Jr.", "Sr.",
            "Inc.", "Corp.", "Ltd.", "Co.", "LLC"
        ]

        # Build placeholder map and a single regex for each direction
        abbr_to_placeholder = {abbr: f"__ABBR{i}__" for i, abbr in enumerate(abbreviations)}
        placeholder_to_abbr = {v: k for k, v in abbr_to_placeholder.items()}
        protect_re = re.compile("|".join(re.escape(a) for a in abbreviations))
        restore_re = re.compile("|".join(re.escape(p) for p in placeholder_to_abbr))

        protected_text = protect_re.sub(lambda m: abbr_to_placeholder[m.group()], text)

        # Split on sentence-ending punctuation followed by space and capital letter
        # or followed by newline
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n+', protected_text)

        # Restore abbreviations with a single regex sub per sentence
        restored_sentences = []
        for sentence in sentences:
            sentence = restore_re.sub(lambda m: placeholder_to_abbr[m.group()], sentence).strip()
            if len(sentence) > 10:
                restored_sentences.append(sentence)

        return restored_sentences

    def chunk_text(self, text: str, overlap_tokens: int = None) -> list[str]:
        """Chunk text into token-sized pieces with overlap.

        Args:
            text: Text to chunk
            overlap_tokens: Number of tokens to overlap between chunks (default: self.overlap_tokens)

        Returns:
            List of text chunks
        """
        if overlap_tokens is None:
            overlap_tokens = self.overlap_tokens

        # Split into sentences
        sentences = self.split_into_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk_sentences = []
        current_tokens = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self.count_tokens(sentence)

            # If a single sentence exceeds max_tokens, split it
            if sentence_tokens > self.max_tokens:
                # If we have accumulated sentences, save them first
                if current_chunk_sentences:
                    chunks.append(" ".join(current_chunk_sentences))
                    current_chunk_sentences = []
                    current_tokens = 0

                # Hard split the long sentence
                encoded = self.encoder.encode(sentence)
                for start_idx in range(0, len(encoded), self.max_tokens):
                    chunk_tokens = encoded[start_idx:start_idx + self.max_tokens]
                    chunk_text = self.encoder.decode(chunk_tokens)
                    chunks.append(chunk_text)

                i += 1
                continue

            # Check if adding this sentence would exceed max_tokens
            if current_tokens + sentence_tokens > self.max_tokens and current_chunk_sentences:
                # Save current chunk
                chunks.append(" ".join(current_chunk_sentences))

                # Start new chunk with overlap
                # Find sentences from the end that fit within overlap_tokens
                overlap_sentences = []
                overlap_token_count = 0
                for sent in reversed(current_chunk_sentences):
                    sent_tokens = self.count_tokens(sent)
                    if overlap_token_count + sent_tokens <= overlap_tokens:
                        overlap_sentences.insert(0, sent)
                        overlap_token_count += sent_tokens
                    else:
                        break

                current_chunk_sentences = overlap_sentences
                current_tokens = overlap_token_count

            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_tokens += sentence_tokens
            i += 1

        # Add final chunk if it exists and meets minimum token requirement
        # (or if it's the only/last chunk from a short document)
        if current_chunk_sentences:
            final_chunk_text = " ".join(current_chunk_sentences)
            final_tokens = self.count_tokens(final_chunk_text)

            # Include final chunk if:
            # - It meets min_tokens, OR
            # - It's the only chunk, OR
            # - We already have chunks and this is just leftover
            if final_tokens >= self.min_tokens or len(chunks) == 0:
                chunks.append(final_chunk_text)
            else:
                # Audit bug: a genuine short trailing sentence used to be
                # silently discarded here once a full chunk already
                # existed - confirmed by execution with a real closing
                # sentence that never appeared in any output chunk. Merge
                # it into the previous chunk instead of losing the source
                # text entirely (may push that chunk slightly past
                # max_tokens, which is preferable to dropping real content).
                chunks[-1] = f"{chunks[-1]} {final_chunk_text}"

        return chunks

    def chunk_document(self, doc: RawDocument) -> list[Chunk]:
        """Chunk a raw document into semantic chunks.

        Args:
            doc: RawDocument to chunk

        Returns:
            List of Chunk objects
        """
        all_chunks = []
        chunk_index = 0

        # Generate source hash for deterministic chunk IDs
        source_hash = hashlib.md5(
            f"{doc.ticker}{doc.filing_date}{doc.accession_number}".encode()
        ).hexdigest()[:8]

        # Process each section
        for section_name, section_text in doc.sections.items():
            if not section_text or not section_text.strip():
                continue

            # Chunk the section text
            text_chunks = self.chunk_text(section_text)

            # Create Chunk objects
            for text_chunk in text_chunks:
                chunk_id = f"{doc.ticker.lower()}_{doc.doc_type.lower().replace('-', '')}_{source_hash}_{chunk_index:04d}"

                chunk = Chunk(
                    chunk_id=chunk_id,
                    ticker=doc.ticker,
                    text=text_chunk,
                    token_count=self.count_tokens(text_chunk),
                    doc_type=doc.doc_type,
                    source=doc.source,
                    section=section_name,
                    date=doc.filing_date,
                    url=None,
                    chunk_index=chunk_index,
                    total_chunks=0  # Will be updated after processing all sections
                )

                all_chunks.append(chunk)
                chunk_index += 1

        # Update total_chunks for all chunks
        total = len(all_chunks)
        for chunk in all_chunks:
            chunk.total_chunks = total

        logger.info(f"Created {total} chunks from {doc.doc_type} for {doc.ticker}")
        return all_chunks

    def chunk_article(self, article: RawArticle) -> list[Chunk]:
        """Chunk a news article into semantic chunks.

        Args:
            article: RawArticle to chunk

        Returns:
            List of Chunk objects
        """
        # Combine title and content
        full_text = f"{article.title}. {article.content}"
        article_tokens = self.count_tokens(full_text)

        # Generate source hash from URL
        source_hash = hashlib.md5(article.url.encode()).hexdigest()[:8]

        # If article is short enough, return as single chunk
        if article_tokens <= self.max_tokens:
            chunk_id = f"{article.ticker.lower()}_news_{source_hash}_0000"

            chunk = Chunk(
                chunk_id=chunk_id,
                ticker=article.ticker,
                text=full_text,
                token_count=article_tokens,
                doc_type="news",
                source=article.source,
                section=None,
                date=article.published_date,
                url=article.url,
                chunk_index=0,
                total_chunks=1
            )

            return [chunk]

        # Article is long, chunk it
        text_chunks = self.chunk_text(full_text)
        chunks = []

        for i, text_chunk in enumerate(text_chunks):
            chunk_id = f"{article.ticker.lower()}_news_{source_hash}_{i:04d}"

            chunk = Chunk(
                chunk_id=chunk_id,
                ticker=article.ticker,
                text=text_chunk,
                token_count=self.count_tokens(text_chunk),
                doc_type="news",
                source=article.source,
                section=None,
                date=article.published_date,
                url=article.url,
                chunk_index=i,
                total_chunks=len(text_chunks)
            )

            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks from news article for {article.ticker}")
        return chunks
