"""Sentiment extraction module."""

import json
import logging
import math
import re
from datetime import datetime, timedelta

from openai import OpenAI

from alphasignal.api.schemas import SentimentSignal
from alphasignal.generation import SentimentResult
from alphasignal.ingestion import Chunk

logger = logging.getLogger(__name__)


class SentimentExtractor:
    """Extracts sentiment from financial text using LLM."""

    SENTIMENT_PROMPT = """Analyse the sentiment of the following financial text excerpt.
Respond ONLY with a valid JSON object in exactly this format, no other text:
{
    "score": <float between -1.0 (very negative) and 1.0 (very positive)>,
    "confidence": <float between 0.0 and 1.0>,
    "key_positive": [<up to 3 short positive phrases from the text>],
    "key_negative": [<up to 3 short negative phrases from the text>],
    "summary": <one sentence summarising the sentiment and main topics>
}"""

    def __init__(self, config: dict):
        """Initialize sentiment extractor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        generation_config = config.get("generation", {})

        # Initialize OpenAI client
        self.client = OpenAI()

        # Store generation parameters
        self.model = generation_config.get("model", "gpt-5.6-luna")
        self.cache_hours = generation_config.get("sentiment_cache_hours", 24)

        # Cache: chunk_id -> (SentimentResult, timestamp)
        self._cache: dict[str, tuple[SentimentResult, datetime]] = {}

        logger.info(
            f"Initialized SentimentExtractor with cache_hours={self.cache_hours}"
        )

    def extract_sentiment(self, chunk: Chunk) -> SentimentResult:
        """Extract sentiment from a chunk.

        Args:
            chunk: Chunk to analyze

        Returns:
            SentimentResult with sentiment scores and analysis
        """
        # Check cache
        if chunk.chunk_id in self._cache:
            cached_result, cached_time = self._cache[chunk.chunk_id]
            age = datetime.now() - cached_time

            if age < timedelta(hours=self.cache_hours):
                logger.debug(f"Cache hit for chunk {chunk.chunk_id}")
                return cached_result

        logger.info(f"Extracting sentiment for chunk {chunk.chunk_id}")

        # Build prompt
        prompt = f"{self.SENTIMENT_PROMPT}\n\nText:\n{chunk.text}"

        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                # temperature=0 omitted: gpt-5.6-luna rejects any non-default
                # temperature (reasoning-tier restriction) - see generator.py's
                # same omission and the caller-facing note about losing
                # determinism as a result.
                max_completion_tokens=500,
            )

            # Extract response text
            response_text = response.choices[0].message.content or ""

            # Parse JSON
            sentiment_data = self._parse_sentiment_json(response_text)

            # Validate and clamp scores. NaN/Infinity must be rejected
            # BEFORE clamping, not clamped: Python's max()/min() don't
            # special-case NaN (every comparison against NaN is False), so
            # max(-1.0, min(1.0, nan)) silently evaluates to 1.0 - a
            # malformed/NaN score from the LLM provider would otherwise
            # become a false maximally-bullish signal instead of failing
            # safe (audit finding, 2026-07-14).
            raw_score = float(sentiment_data.get("score", 0.0))
            raw_confidence = float(sentiment_data.get("confidence", 0.0))
            if not (math.isfinite(raw_score) and math.isfinite(raw_confidence)):
                logger.error(
                    f"Non-finite sentiment score/confidence from provider for "
                    f"chunk {chunk.chunk_id}: score={raw_score}, confidence={raw_confidence} "
                    f"- treating as a provider malfunction, defaulting to neutral"
                )
                result = SentimentResult(
                    score=0.0,
                    confidence=0.0,
                    key_positive=[],
                    key_negative=[],
                    summary="Invalid score from provider",
                )
                self._cache[chunk.chunk_id] = (result, datetime.now())
                return result

            score = max(-1.0, min(1.0, raw_score))  # Clamp to [-1, 1]
            confidence = max(0.0, min(1.0, raw_confidence))  # Clamp to [0, 1]

            result = SentimentResult(
                score=score,
                confidence=confidence,
                key_positive=sentiment_data.get("key_positive", [])[:3],
                key_negative=sentiment_data.get("key_negative", [])[:3],
                summary=sentiment_data.get("summary", ""),
            )

            # Cache result
            self._cache[chunk.chunk_id] = (result, datetime.now())

            logger.debug(
                f"Extracted sentiment: score={score:.2f}, confidence={confidence:.2f}"
            )
            return result

        except Exception as e:
            logger.error(
                f"Error extracting sentiment for chunk {chunk.chunk_id}: {e}",
                exc_info=True,
            )
            # Return default result
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                key_positive=[],
                key_negative=[],
                summary="Parse error",
            )

    def extract_ticker_sentiment(
        self, ticker: str, chunks: list[Chunk]
    ) -> list[SentimentSignal]:
        """Extract sentiment signals for a ticker from chunks.

        Args:
            ticker: Ticker symbol
            chunks: List of chunks to analyze

        Returns:
            List of SentimentSignal objects sorted by date descending
        """
        logger.info(
            f"Extracting ticker sentiment for {ticker} from {len(chunks)} chunks"
        )

        # Sort by date descending and take most recent 10
        sorted_chunks = sorted(chunks, key=lambda c: c.date, reverse=True)
        recent_chunks = sorted_chunks[:10]

        # Extract sentiment for each chunk
        signals = []
        for chunk in recent_chunks:
            sentiment = self.extract_sentiment(chunk)

            signal = SentimentSignal(
                ticker=chunk.ticker,
                date=chunk.date,
                score=sentiment.score,
                confidence=sentiment.confidence,
                source=chunk.source,
                doc_type=chunk.doc_type,
                summary=sentiment.summary,
                key_positive=sentiment.key_positive,
                key_negative=sentiment.key_negative,
            )
            signals.append(signal)

        # Sort by date descending
        signals.sort(key=lambda s: s.date, reverse=True)

        logger.info(f"Extracted {len(signals)} sentiment signals for {ticker}")
        return signals

    def _parse_sentiment_json(self, response_text: str) -> dict:
        """Parse JSON from LLM response, with fallback handling.

        Args:
            response_text: Raw response text from LLM

        Returns:
            Parsed sentiment dictionary
        """
        try:
            # Try direct parse
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            # Try to extract JSON substring using regex
            json_match = re.search(r"\{[^}]+\}", response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            # Return default if all parsing fails
            logger.warning(f"Failed to parse sentiment JSON: {response_text[:100]}")
            return {
                "score": 0.0,
                "confidence": 0.0,
                "key_positive": [],
                "key_negative": [],
                "summary": "Parse error",
            }
