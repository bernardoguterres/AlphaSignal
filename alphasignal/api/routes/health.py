"""Health check endpoint for AlphaSignal API."""

import logging
import time
from fastapi import APIRouter, Request
from alphasignal.api.schemas import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return health status of the AlphaSignal API."""
    # Check if we have vector store and metadata store (set on app.state.app_state at startup)
    faiss_loaded = False
    sqlite_connected = False
    chunks_indexed = 0
    uptime = 0.0

    if hasattr(request.app.state, "app_state"):
        app_state = request.app.state.app_state
        uptime = time.time() - app_state.start_time
        try:
            # Check vector store
            if (
                app_state.retriever.vector_store
                and app_state.retriever.vector_store.index is not None
            ):
                faiss_loaded = True

            # Check metadata store - sqlite_connected is only set True
            # AFTER count() actually succeeds, not before: a store object
            # being non-None doesn't mean it's actually reachable (e.g. a
            # locked/corrupted DB file), and setting the flag first meant a
            # failing count() left sqlite_connected=True (caught by the
            # except block below) even though the store had just failed.
            if app_state.retriever.metadata_store:
                chunks_indexed = app_state.retriever.metadata_store.count()
                sqlite_connected = True
        except Exception:
            logger.warning("Health check failed to read store status", exc_info=True)

    # Audit bug: status previously hardcoded "healthy" regardless of corpus
    # state - a fresh Railway deploy with zero ingested data (or a
    # store-connection failure) passed the healthcheck and looked identical
    # to a real, ready deployment. HTTP status code stays 200 in all three
    # cases deliberately, so Railway's healthcheck (which only looks at the
    # HTTP status, not this field) still passes and doesn't block a fresh
    # deploy before ingestion has happened - only the `status` field value
    # now reflects real readiness for callers that actually check it.
    if not sqlite_connected:
        status = "unhealthy"
    elif chunks_indexed == 0:
        status = "no_data"
    else:
        status = "healthy"

    return HealthResponse(
        status=status,
        version="0.1.0",
        faiss_index_loaded=faiss_loaded,
        sqlite_connected=sqlite_connected,
        chunks_indexed=chunks_indexed,
        uptime_seconds=uptime,
    )
