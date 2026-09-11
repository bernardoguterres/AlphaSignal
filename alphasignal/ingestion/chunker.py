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

        self._normalize_token_config()

        # Initialize tiktoken encoder (cl100k_base - used by OpenAI embedding models)
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def _normalize_token_config(self):
        """Reject/normalize invalid chunking config combinations predictably.

        No validation existed before this - a misconfigured max_tokens<=0,
        min_tokens>max_tokens, or overlap_tokens>=max_tokens wouldn't error,
        it would just produce silently bizarre chunk boundaries. Every
        correction here is deterministic and logged, never chosen to tune
        retrieval quality.
        """
        if self.max_tokens <= 0:
            logger.warning(
                f"chunking.max_tokens={self.max_tokens} is invalid (must be > 0); "
                "falling back to 400"
            )
            self.max_tokens = 400

        if self.min_tokens < 0:
            logger.warning(
                f"chunking.min_tokens={self.min_tokens} is invalid (must be >= 0); "
                "clamping to 0"
            )
            self.min_tokens = 0
        if self.min_tokens > self.max_tokens:
            logger.warning(
                f"chunking.min_tokens={self.min_tokens} exceeds max_tokens="
                f"{self.max_tokens}; clamping min_tokens to max_tokens"
            )
            self.min_tokens = self.max_tokens

        if self.overlap_tokens < 0:
            logger.warning(
                f"chunking.overlap_tokens={self.overlap_tokens} is invalid "
                "(must be >= 0); clamping to 0"
            )
            self.overlap_tokens = 0
        if self.overlap_tokens >= self.max_tokens:
            logger.warning(
                f"chunking.overlap_tokens={self.overlap_tokens} >= max_tokens="
                f"{self.max_tokens}; clamping to max_tokens - 1"
            )
            self.overlap_tokens = max(0, self.max_tokens - 1)

        if self.target_tokens < 0:
            logger.warning(
                f"chunking.target_tokens={self.target_tokens} is invalid "
                "(must be >= 0); treating as unset (0)"
            )
            self.target_tokens = 0
        # target_tokens > max_tokens is not an error - chunk_text() already
        # caps the effective boundary at max_tokens via
        # min(target_tokens, max_tokens), so it degrades to "unused" rather
        # than needing correction here.

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
            "U.S.",
            "U.K.",
            "E.U.",
            "e.g.",
            "i.e.",
            "etc.",
            "Dr.",
            "Mr.",
            "Mrs.",
            "Ms.",
            "Jr.",
            "Sr.",
            "Inc.",
            "Corp.",
            "Ltd.",
            "Co.",
            "LLC",
        ]

        # Build placeholder map and a single regex for each direction
        abbr_to_placeholder = {
            abbr: f"__ABBR{i}__" for i, abbr in enumerate(abbreviations)
        }
        placeholder_to_abbr = {v: k for k, v in abbr_to_placeholder.items()}
        protect_re = re.compile("|".join(re.escape(a) for a in abbreviations))
        restore_re = re.compile("|".join(re.escape(p) for p in placeholder_to_abbr))

        protected_text = protect_re.sub(lambda m: abbr_to_placeholder[m.group()], text)

        # Split on sentence-ending punctuation followed by space and capital letter
        # or followed by newline
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n+", protected_text)

        # Restore abbreviations with a single regex sub per sentence
        restored_sentences = []
        for sentence in sentences:
            sentence = restore_re.sub(
                lambda m: placeholder_to_abbr[m.group()], sentence
            ).strip()
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

        # target_tokens is the *normal* chunk-boundary threshold (the size
        # we aim for); max_tokens remains the hard ceiling used only for
        # force-splitting a single oversized sentence. If target_tokens is
        # unset or configured >= max_tokens, it has no distinct effect and
        # boundaries fall back to max_tokens exactly as before (audit
        # finding, 2026-09-11: target_tokens was accepted from config but
        # never referenced anywhere in the boundary logic).
        boundary_tokens = (
            min(self.target_tokens, self.max_tokens)
            if self.target_tokens > 0
            else self.max_tokens
        )

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
                    chunk_tokens = encoded[start_idx : start_idx + self.max_tokens]
                    chunk_text = self.encoder.decode(chunk_tokens)
                    chunks.append(chunk_text)

                i += 1
                continue

            # Check if adding this sentence would exceed the normal boundary
            # (target_tokens, capped at max_tokens)
            if (
                current_tokens + sentence_tokens > boundary_tokens
                and current_chunk_sentences
            ):
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

    def document_source_prefix(self, doc: RawDocument) -> str:
        """Return the stable source-identity prefix chunk_ids for this
        document will share, independent of how many chunks (if any) it
        actually produces this run.

        Needed so a document that now yields ZERO chunks (all sections
        emptied out) can still have its *previous* chunks recognized as
        orphans and cleaned up - store_chunks() can only derive a prefix
        from chunk_ids it was actually given, which is nothing at all in
        that case (audit finding, 2026-09-11).
        """
        source_hash = hashlib.md5(
            f"{doc.ticker}{doc.filing_date}{doc.accession_number}".encode()
        ).hexdigest()[:8]
        return f"{doc.ticker.lower()}_{doc.doc_type.lower().replace('-', '')}_{source_hash}"

    def article_source_prefix(self, article: RawArticle) -> str:
        """Return the stable source-identity prefix chunk_ids for this
        article will share. See document_source_prefix() for why this is
        exposed independent of chunk_document/chunk_article's own output.
        """
        source_hash = hashlib.md5(article.url.encode()).hexdigest()[:8]
        return f"{article.ticker.lower()}_news_{source_hash}"

    def chunk_document(self, doc: RawDocument) -> list[Chunk]:
        """Chunk a raw document into semantic chunks.

        Args:
            doc: RawDocument to chunk

        Returns:
            List of Chunk objects
        """
        all_chunks = []
        chunk_index = 0

        source_hash_prefix = self.document_source_prefix(doc)

        # Process each section
        for section_name, section_text in doc.sections.items():
            if not section_text or not section_text.strip():
                continue

            # Chunk the section text
            text_chunks = self.chunk_text(section_text)

            # Create Chunk objects
            for text_chunk in text_chunks:
                chunk_id = f"{source_hash_prefix}_{chunk_index:04d}"

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
                    total_chunks=0,  # Will be updated after processing all sections
                    source_id=source_hash_prefix,
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

        source_hash_prefix = self.article_source_prefix(article)

        # If article is short enough, return as single chunk
        if article_tokens <= self.max_tokens:
            chunk_id = f"{source_hash_prefix}_0000"

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
                total_chunks=1,
                source_id=source_hash_prefix,
            )

            return [chunk]

        # Article is long, chunk it
        text_chunks = self.chunk_text(full_text)
        chunks = []

        for i, text_chunk in enumerate(text_chunks):
            chunk_id = f"{source_hash_prefix}_{i:04d}"

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
                total_chunks=len(text_chunks),
                source_id=source_hash_prefix,
            )

            chunks.append(chunk)

        logger.info(
            f"Created {len(chunks)} chunks from news article for {article.ticker}"
        )
        return chunks
