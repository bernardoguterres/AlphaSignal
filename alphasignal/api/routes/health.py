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

            # Check metadata store
            if app_state.retriever.metadata_store:
                sqlite_connected = True
                chunks_indexed = app_state.retriever.metadata_store.count()
        except Exception:
            logger.warning("Health check failed to read store status", exc_info=True)

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        faiss_index_loaded=faiss_loaded,
        sqlite_connected=sqlite_connected,
        chunks_indexed=chunks_indexed,
        uptime_seconds=uptime,
    )
