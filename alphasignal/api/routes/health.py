"""Health check endpoint for AlphaSignal API."""

import logging
import time
from pathlib import Path
from fastapi import APIRouter, Request
from alphasignal.api.schemas import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _count_verified_vectors(vector_hashes: dict, metadata_store) -> int:
    """Vectors the retriever will actually use: the stored hash must equal
    the chunk's CURRENT hash in SQLite (the same rule as dense search).
    Orphan vectors and vectors of since-changed text are not counted."""
    if not vector_hashes or metadata_store is None:
        return 0
    current = metadata_store.get_content_hashes()
    return sum(1 for cid, h in vector_hashes.items() if h and current.get(cid) == h)


_version_cache: dict = {}


def _versions(metadata_store, vector_store) -> dict:
    """Active corpus/index versions, recomputed only when the database file
    or the FAISS generation changes."""
    key = (
        getattr(metadata_store, "db_path", None)
        and Path(metadata_store.db_path).stat().st_mtime_ns,
        vector_store.generation,
        len(vector_store.chunk_ids),
    )
    if _version_cache.get("key") != key:
        info = metadata_store.version_info()
        index_digest = vector_store.content_digest()
        _version_cache.update(
            key=key,
            value={
                "corpus_version": info["corpus_version"],
                "content_digest": info["content_digest"],
                "index_digest": index_digest,
                "corpus_consistent": index_digest == info["content_digest"],
                "chunker_versions": info["chunker_versions"],
            },
        )
    return _version_cache["value"]


@router.get("/", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return health status of the AlphaSignal API."""
    # Check if we have vector store and metadata store (set on app.state.app_state at startup)
    faiss_loaded = False
    sqlite_connected = False
    chunks_indexed = 0
    uptime = 0.0
    vectors_total = 0
    vectors_verified = 0
    versions: dict = {}

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
                vs = app_state.retriever.vector_store
                vectors_total = int(vs.index.ntotal)
                vectors_verified = _count_verified_vectors(
                    vs.content_hashes, app_state.retriever.metadata_store
                )

            # Check metadata store - sqlite_connected is only set True
            # AFTER count() actually succeeds, not before: a store object
            # being non-None doesn't mean it's actually reachable (e.g. a
            # locked/corrupted DB file), and setting the flag first meant a
            # failing count() left sqlite_connected=True (caught by the
            # except block below) even though the store had just failed.
            if app_state.retriever.metadata_store:
                chunks_indexed = app_state.retriever.metadata_store.count()
                sqlite_connected = True
            if faiss_loaded and sqlite_connected:
                versions = _versions(
                    app_state.retriever.metadata_store,
                    app_state.retriever.vector_store,
                )
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
    elif vectors_verified == 0 or versions.get("corpus_consistent") is False:
        # Vectors with no recorded content hash are excluded from dense
        # retrieval, so every query would silently run BM25-only; and an
        # index that does not hold exactly the stored chunks belongs to
        # another corpus version.
        status = "degraded"
    else:
        status = "healthy"
    dense_mode = (
        "unavailable"
        if vectors_verified == 0
        else "full" if vectors_verified >= chunks_indexed else "partial"
    )

    return HealthResponse(
        status=status,
        version="0.1.0",
        faiss_index_loaded=faiss_loaded,
        sqlite_connected=sqlite_connected,
        chunks_indexed=chunks_indexed,
        uptime_seconds=uptime,
        vectors_total=vectors_total,
        vectors_verified=vectors_verified,
        dense_retrieval=dense_mode,
        **versions,
    )
