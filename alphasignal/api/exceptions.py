"""Custom exceptions for AlphaSignal API."""

from typing import Any


class AlphaSignalError(Exception):
    """Base exception for AlphaSignal errors.

    Attributes:
        message: Human-readable error message
        code: Machine-readable error code for client handling
        detail: Optional additional context or stack trace info
    """

    def __init__(
        self,
        message: str,
        code: str,
        detail: str | None = None,
        **kwargs: Any
    ):
        """Initialize AlphaSignalError.

        Args:
            message: Human-readable error message
            code: Error code (e.g., "OPENAI_ERROR", "INDEX_NOT_LOADED")
            detail: Optional additional error context
            **kwargs: Additional fields to include in error response
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.detail = detail
        self.extra = kwargs


class GenerationError(AlphaSignalError):
    """Error from OpenAI API calls (RAG generation, embeddings, sentiment).

    Named distinctly from `openai.OpenAIError` - the routes catch and
    re-raise the real SDK exception directly (see api/routes/query.py),
    so a same-named class here would just be a confusing landmine.
    """

    def __init__(self, message: str, detail: str | None = None):
        """Initialize GenerationError.

        Args:
            message: Error message from OpenAI or custom message
            detail: Optional API error details
        """
        super().__init__(message=message, code="OPENAI_ERROR", detail=detail)


class IndexNotLoadedError(AlphaSignalError):
    """Error when FAISS index is not loaded."""

    def __init__(self, message: str = "FAISS index not loaded", detail: str | None = None):
        """Initialize IndexNotLoadedError.

        Args:
            message: Error message
            detail: Optional details about why index isn't loaded
        """
        super().__init__(message=message, code="INDEX_NOT_LOADED", detail=detail)


class DatabaseError(AlphaSignalError):
    """Error from SQLite database operations."""

    def __init__(self, message: str, detail: str | None = None):
        """Initialize DatabaseError.

        Args:
            message: Error message
            detail: Optional database error details
        """
        super().__init__(message=message, code="DB_ERROR", detail=detail)


class IngestionError(AlphaSignalError):
    """Error during data ingestion."""

    def __init__(self, message: str, source: str, detail: str | None = None):
        """Initialize IngestionError.

        Args:
            message: Error message
            source: Ingestion source (e.g., "EDGAR", "RSS")
            detail: Optional error details
        """
        super().__init__(
            message=message,
            code="INGESTION_ERROR",
            detail=detail,
            source=source
        )
