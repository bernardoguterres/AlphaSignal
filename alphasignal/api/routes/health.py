"""Health check endpoint for AlphaSignal API."""

import time
from fastapi import APIRouter, Request
from alphasignal.api.schemas import HealthResponse

router = APIRouter()

# Track startup time
startup_time = time.time()


@router.get("/", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return health status of the AlphaSignal API."""
    uptime = time.time() - startup_time

    # Check if we have vector store and metadata store (will be added by app)
    faiss_loaded = False
    sqlite_connected = False
    chunks_indexed = 0

    # Try to get pipeline from app state if available
    if hasattr(request.app.state, 'pipeline'):
        pipeline = request.app.state.pipeline
        try:
            # Check vector store
            if pipeline.vector_store and pipeline.vector_store.index is not None:
                faiss_loaded = True

            # Check metadata store
            if pipeline.metadata_store:
                sqlite_connected = True
                chunks_indexed = pipeline.metadata_store.count()
        except Exception:
            pass  # If stores aren't initialized, keep defaults

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        faiss_index_loaded=faiss_loaded,
        sqlite_connected=sqlite_connected,
        chunks_indexed=chunks_indexed,
        uptime_seconds=uptime
    )
