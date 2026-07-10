"""Metrics endpoint for AlphaSignal API."""

import time

from fastapi import APIRouter, Depends

from alphasignal.api.dependencies import get_app_state, get_metadata_store
from alphasignal.api.state import AppState
from alphasignal.store.metadata_store import MetadataStore

router = APIRouter()


@router.get("/")
def get_metrics(
    state: AppState = Depends(get_app_state),
    metadata_store: MetadataStore = Depends(get_metadata_store),
):
    """Get performance metrics and statistics.

    Args:
        state: Application state
        metadata_store: Metadata store instance

    Returns:
        Dictionary with all metrics
    """
    # Get metrics summary
    metrics = state.metrics_collector.get_summary()

    # Add additional system metrics
    uptime_seconds = time.time() - state.start_time
    chunks_indexed = metadata_store.count()

    # Combine all metrics
    return {
        **metrics,
        "system": {
            "uptime_seconds": int(uptime_seconds),
            "chunks_indexed": chunks_indexed,
        },
    }
