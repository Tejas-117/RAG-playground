"""SQLite boundaries for immutable ranked retrieval results."""

import math
import sqlite3
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid5

from backend.db.connection import connect
from backend.pipeline.configs import DistanceMetric
from backend.retrieval.models import HydratedVectorSearchHit

# Deterministic IDs make a run's one allowed retrieval result stable across retries.
_RETRIEVAL_RESULT_NAMESPACE = UUID("5c56a770-b87f-4f20-8a67-d4280303964a")


class InvalidRetrievalResultError(ValueError):
    """Report malformed ranking, counts, metrics, or timing before persistence."""


class RetrievalArtifactMismatchError(RuntimeError):
    """Report a run, vector index, or chunk from incompatible artifacts."""


def save_retrieval_result(
    pipeline_run_id: str,
    vector_index_id: str,
    requested_top_k: int,
    distance_metric: DistanceMetric | str,
    duration_ms: int,
    hits: tuple[HydratedVectorSearchHit, ...],
) -> dict[str, Any]:
    """Validate and atomically persist one result with all ranked chunk links.

    Args:
        pipeline_run_id: Stable run that owns this non-reusable result.
        vector_index_id: Exact ready index searched for the run.
        requested_top_k: Maximum result count requested by retrieval config.
        distance_metric: Raw-distance semantics used by the vector index.
        duration_ms: Non-negative retrieval stage wall-clock duration.
        hits: Ranked hydrated hits produced from the searched index's chunk set.

    Returns:
        Materialized retrieval summary and ordered lightweight ranked hits.

    Raises:
        InvalidRetrievalResultError: If counts, ranks, values, or metric are invalid.
        RetrievalArtifactMismatchError: If run, index, or chunks are incompatible.
        sqlite3.IntegrityError: If this run already owns a retrieval result.
    """
    # Use the shared connection-aware writer so standalone and pipeline writes agree.
    with connect() as connection:
        result_id = insert_retrieval_result(
            connection,
            pipeline_run_id,
            vector_index_id,
            requested_top_k,
            distance_metric,
            duration_ms,
            hits,
        )

    return get_retrieval_result(result_id)


def insert_retrieval_result(
    connection: sqlite3.Connection,
    pipeline_run_id: str,
    vector_index_id: str,
    requested_top_k: int,
    distance_metric: DistanceMetric | str,
    duration_ms: int,
    hits: tuple[HydratedVectorSearchHit, ...],
) -> str:
    """Insert one validated retrieval result using an existing transaction.

    Args:
        connection: Active SQLite transaction that owns the complete operation.
        pipeline_run_id: Stable run that owns this non-reusable result.
        vector_index_id: Exact ready index searched for the run.
        requested_top_k: Maximum result count requested by retrieval config.
        distance_metric: Raw-distance semantics used by the vector index.
        duration_ms: Non-negative retrieval stage wall-clock duration.
        hits: Ranked hydrated hits produced from the searched index's chunk set.

    Returns:
        Deterministic identifier of the inserted retrieval result.

    Raises:
        InvalidRetrievalResultError: If counts, ranks, values, or metric are invalid.
        RetrievalArtifactMismatchError: If run, index, or chunks are incompatible.
        sqlite3.IntegrityError: If relational constraints reject the result.
    """
    normalized_metric = _validate_result_values(
        requested_top_k,
        distance_metric,
        duration_ms,
        hits,
    )
    result_id = str(
        uuid5(
            _RETRIEVAL_RESULT_NAMESPACE,
            f"retrieval-result:{pipeline_run_id}",
        )
    )
    created_at = _utc_timestamp()

    # Validate ownership before making either the parent or ranked children visible.
    chunk_set_id = _validate_artifact_ownership(
        connection,
        pipeline_run_id,
        vector_index_id,
        normalized_metric,
    )
    _validate_hit_ownership(connection, chunk_set_id, hits)
    connection.execute(
        """
        INSERT INTO retrieval_result (
            id, pipeline_run_id, vector_index_id, requested_top_k,
            returned_count, distance_metric, duration_ms, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result_id,
            pipeline_run_id,
            vector_index_id,
            requested_top_k,
            len(hits),
            normalized_metric,
            duration_ms,
            created_at,
        ),
    )

    # Persist only stable references and raw score data in retrieval order.
    for hit in hits:
        connection.execute(
            """
            INSERT INTO retrieved_chunk (
                retrieval_result_id, rank, chunk_id, raw_distance
            ) VALUES (?, ?, ?, ?)
            """,
            (result_id, hit.rank, hit.chunk_id, hit.raw_distance),
        )

    return result_id


def get_retrieval_result(retrieval_result_id: str) -> dict[str, Any]:
    """Load one persisted retrieval result and its ranked chunk references.

    Args:
        retrieval_result_id: Stable retrieval result identifier to load.

    Returns:
        Materialized result summary and ordered lightweight hits.

    Raises:
        LookupError: If the requested retrieval result does not exist.
    """
    # Read parent and ranked children from one consistent database connection.
    with connect() as connection:
        result_row = connection.execute(
            "SELECT * FROM retrieval_result WHERE id = ?",
            (retrieval_result_id,),
        ).fetchone()

        # Keep an unknown result distinct from a valid result with no hits.
        if result_row is None:
            raise LookupError(retrieval_result_id)

        hit_rows = connection.execute(
            """
            SELECT rank, chunk_id, raw_distance
            FROM retrieved_chunk
            WHERE retrieval_result_id = ?
            ORDER BY rank
            """,
            (retrieval_result_id,),
        ).fetchall()

    return {
        "id": result_row["id"],
        "pipeline_run_id": result_row["pipeline_run_id"],
        "vector_index_id": result_row["vector_index_id"],
        "requested_top_k": result_row["requested_top_k"],
        "returned_count": result_row["returned_count"],
        "distance_metric": result_row["distance_metric"],
        "duration_ms": result_row["duration_ms"],
        "created_at": result_row["created_at"],
        "hits": [dict(row) for row in hit_rows],
    }


def _validate_result_values(
    requested_top_k: int,
    distance_metric: DistanceMetric | str,
    duration_ms: int,
    hits: tuple[HydratedVectorSearchHit, ...],
) -> str:
    """Validate result-level values and return a normalized metric string.

    Args:
        requested_top_k: Configured maximum number of returned chunks.
        distance_metric: Enum or string describing raw distance semantics.
        duration_ms: Retrieval wall-clock duration in milliseconds.
        hits: Ranked hydrated chunks proposed for persistence.

    Returns:
        Validated string representation of the distance metric.

    Raises:
        InvalidRetrievalResultError: If any result-level invariant is violated.
    """
    # Reject booleans explicitly because Python otherwise treats them as integers.
    if (
        isinstance(requested_top_k, bool)
        or not isinstance(requested_top_k, int)
        or requested_top_k <= 0
    ):
        raise InvalidRetrievalResultError("requested_top_k must be positive.")

    # Stage duration must be a concrete non-negative integer measurement.
    if (
        isinstance(duration_ms, bool)
        or not isinstance(duration_ms, int)
        or duration_ms < 0
    ):
        raise InvalidRetrievalResultError("duration_ms must be non-negative.")

    normalized_metric = (
        distance_metric.value
        if isinstance(distance_metric, DistanceMetric)
        else distance_metric
    )
    allowed_metrics = {metric.value for metric in DistanceMetric}

    # Preserve only a metric whose raw score semantics the application recognizes.
    if normalized_metric not in allowed_metrics:
        raise InvalidRetrievalResultError("distance_metric is unsupported.")

    # A vector store may return fewer than requested, but never more than requested.
    if len(hits) > requested_top_k:
        raise InvalidRetrievalResultError(
            "The returned hit count exceeds requested_top_k."
        )

    expected_ranks = list(range(1, len(hits) + 1))
    actual_ranks = [hit.rank for hit in hits]

    # Contiguous one-based ranks preserve the exact vector-search order.
    if actual_ranks != expected_ranks:
        raise InvalidRetrievalResultError(
            "Retrieval hit ranks must be contiguous and one-based."
        )

    chunk_ids = [hit.chunk_id for hit in hits]

    # Duplicate chunk references would make the result ranking ambiguous.
    if len(chunk_ids) != len(set(chunk_ids)):
        raise InvalidRetrievalResultError(
            "Retrieval hits must contain unique chunk identifiers."
        )

    # SQLite REAL values must not persist NaN or infinity as meaningful distances.
    if any(not math.isfinite(hit.raw_distance) for hit in hits):
        raise InvalidRetrievalResultError("Retrieval hit distances must be finite.")

    return normalized_metric


def _validate_artifact_ownership(
    connection: sqlite3.Connection,
    pipeline_run_id: str,
    vector_index_id: str,
    distance_metric: str,
) -> str:
    """Confirm the run and ready index share the same chunk-set provenance.

    Args:
        connection: Active transaction used for validation and later inserts.
        pipeline_run_id: Run that must already reference the selected artifacts.
        vector_index_id: Ready index that retrieval searched.
        distance_metric: Normalized raw-distance metric claimed by the result.

    Returns:
        Exact chunk-set identifier shared by the run and vector index.

    Raises:
        RetrievalArtifactMismatchError: If any artifact relationship differs.
    """
    row = connection.execute(
        """
        SELECT pipeline_run.chunk_set_id AS run_chunk_set_id,
               pipeline_run.vector_index_id AS run_vector_index_id,
               vector_index.chunk_set_id AS index_chunk_set_id,
               vector_index.distance_metric AS index_distance_metric
        FROM pipeline_run
        JOIN vector_index ON vector_index.id = ? AND vector_index.status = 'ready'
        WHERE pipeline_run.id = ?
        """,
        (vector_index_id, pipeline_run_id),
    ).fetchone()

    # An unknown run or non-ready index cannot own a valid retrieval result.
    if row is None:
        raise RetrievalArtifactMismatchError(
            "The pipeline run or ready vector index does not exist."
        )

    # The result must use the exact index already attached to the immutable run.
    if row["run_vector_index_id"] != vector_index_id:
        raise RetrievalArtifactMismatchError(
            "The vector index does not belong to the pipeline run."
        )

    # Both artifacts must resolve to one exact reusable chunk set.
    if (
        row["run_chunk_set_id"] is None
        or row["run_chunk_set_id"] != row["index_chunk_set_id"]
    ):
        raise RetrievalArtifactMismatchError(
            "The run and vector index use different chunk sets."
        )

    # Raw distances must be labeled with the metric used by the searched index.
    if row["index_distance_metric"] != distance_metric:
        raise RetrievalArtifactMismatchError(
            "The result metric does not match the vector index."
        )

    return row["index_chunk_set_id"]


def _validate_hit_ownership(
    connection: sqlite3.Connection,
    chunk_set_id: str,
    hits: tuple[HydratedVectorSearchHit, ...],
) -> None:
    """Confirm every retrieved chunk belongs to the index's exact chunk set.

    Args:
        connection: Active transaction used for validation and later inserts.
        chunk_set_id: Chunk artifact from which the vector index was built.
        hits: Ranked hydrated chunks proposed for persistence.

    Returns:
        None when every stable chunk identifier belongs to the expected set.

    Raises:
        RetrievalArtifactMismatchError: If a hit is missing or belongs elsewhere.
    """
    # Empty search results are valid and require no dynamic SQL placeholders.
    if not hits:
        return

    chunk_ids = tuple(hit.chunk_id for hit in hits)
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = connection.execute(
        f"""
        SELECT id FROM chunk
        WHERE chunk_set_id = ? AND id IN ({placeholders})
        """,
        (chunk_set_id, *chunk_ids),
    ).fetchall()
    owned_chunk_ids = {row["id"] for row in rows}

    # A scoped count mismatch covers both nonexistent and foreign-set chunk IDs.
    if owned_chunk_ids != set(chunk_ids):
        raise RetrievalArtifactMismatchError(
            "One or more retrieval hits do not belong to the vector index's chunk set."
        )


def _utc_timestamp() -> str:
    """Create the unambiguous UTC timestamp stored with retrieval results.

    Args:
        None.

    Returns:
        ISO-8601 UTC timestamp ending in ``Z``.
    """
    # Match the timestamp representation used by pipeline-run persistence.
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
