"""SQLite persistence boundaries for immutable pipeline-run execution state."""

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from backend.db.connection import connect
from backend.pipeline.configs import PipelineConfig


class CorpusNotFoundError(LookupError):
    """Report that a run references a corpus that does not exist."""


class RunNotFoundError(LookupError):
    """Report that a lifecycle update references an unknown run."""


class InvalidRunStateError(RuntimeError):
    """Report an attempted pipeline-run lifecycle transition from the wrong state."""


class ChunkSetNotReadyError(RuntimeError):
    """Report that run completion references a chunk set that is not ready."""


def _utc_timestamp() -> str:
    """Create the UTC timestamp format used by the SQLite schema.

    Args:
        None.

    Returns:
        An ISO-8601 UTC timestamp ending in ``Z``.
    """
    # Use timezone-aware values so persisted lifecycle times are unambiguous.
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _canonical_json(value: Any) -> str:
    """Serialize configuration and error details deterministically.

    Args:
        value: JSON-compatible value that should be persisted.

    Returns:
        Compact JSON text with stable key ordering.
    """
    # Stable JSON keeps immutable snapshots and tests independent of dict insertion order.
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def create_pending_run(
    corpus_id: str,
    question: str,
    configuration: PipelineConfig,
) -> dict[str, Any]:
    """Persist one validated run before its first pipeline stage starts.

    Args:
        corpus_id: Stable identifier of the immutable corpus selected for the run.
        question: Normalized non-empty question submitted by the user.
        configuration: Typed configuration with all backend defaults resolved.

    Returns:
        Materialized pending run with its immutable configuration snapshot.

    Raises:
        CorpusNotFoundError: If the selected corpus does not exist.
    """
    run_id = str(uuid4())
    created_at = _utc_timestamp()
    effective_config_json = _canonical_json(configuration.model_dump(mode="json"))

    # Verify corpus existence and create the pending run in one transaction.
    with connect() as connection:
        corpus = connection.execute(
            "SELECT id FROM corpus WHERE id = ?",
            (corpus_id,),
        ).fetchone()

        # Return a domain error instead of exposing a foreign-key failure.
        if corpus is None:
            raise CorpusNotFoundError(corpus_id)

        connection.execute(
            """
            INSERT INTO pipeline_run (
                id, corpus_id, chunk_set_id, question, effective_config_json,
                status, chunk_set_reused, created_at, started_at,
                completed_at, duration_ms, error_code, error_details_json
            ) VALUES (?, ?, NULL, ?, ?, 'pending', NULL, ?, NULL, NULL, NULL, NULL, NULL)
            """,
            (run_id, corpus_id, question, effective_config_json, created_at),
        )

    return get_run(run_id)


def mark_run_running(run_id: str) -> dict[str, Any]:
    """Move one pending run into active execution.

    Args:
        run_id: Stable identifier of the pending run to start.

    Returns:
        Materialized running state after the transition.

    Raises:
        InvalidRunStateError: If the run is absent or no longer pending.
    """
    started_at = _utc_timestamp()

    # Conditional update prevents terminal or concurrently started runs being overwritten.
    with connect() as connection:
        cursor = connection.execute(
            """
            UPDATE pipeline_run
            SET status = 'running', started_at = ?
            WHERE id = ? AND status = 'pending'
            """,
            (started_at, run_id),
        )

        # A zero-row update means the requested lifecycle transition is invalid.
        if cursor.rowcount != 1:
            raise InvalidRunStateError(
                f"Pipeline run '{run_id}' cannot transition to running."
            )

    return get_run(run_id)


def complete_run(
    run_id: str,
    chunk_set_id: str,
    chunk_set_reused: bool,
    duration_ms: int,
) -> dict[str, Any]:
    """Complete a running run with its exact ready chunk artifact.

    Args:
        run_id: Stable identifier of the running pipeline execution.
        chunk_set_id: Ready reusable chunk artifact selected by the executor.
        chunk_set_reused: Whether this execution reused an existing artifact.
        duration_ms: Non-negative elapsed execution time in milliseconds.

    Returns:
        Materialized completed run and compact chunking summary.

    Raises:
        ChunkSetNotReadyError: If the supplied artifact is absent or not ready.
        InvalidRunStateError: If the run is absent or no longer running.
    """
    completed_at = _utc_timestamp()

    # Validate the artifact and lifecycle transition in one database transaction.
    with connect() as connection:
        chunk_set = connection.execute(
            "SELECT id FROM chunk_set WHERE id = ? AND status = 'ready'",
            (chunk_set_id,),
        ).fetchone()

        # Never attach a partial or failed reusable artifact to a completed run.
        if chunk_set is None:
            raise ChunkSetNotReadyError(chunk_set_id)

        cursor = connection.execute(
            """
            UPDATE pipeline_run
            SET chunk_set_id = ?, status = 'completed', chunk_set_reused = ?,
                completed_at = ?, duration_ms = ?, error_code = NULL,
                error_details_json = NULL
            WHERE id = ? AND status = 'running'
            """,
            (
                chunk_set_id,
                int(chunk_set_reused),
                completed_at,
                duration_ms,
                run_id,
            ),
        )

        # Protect completed and failed history from accidental later mutation.
        if cursor.rowcount != 1:
            raise InvalidRunStateError(
                f"Pipeline run '{run_id}' cannot transition to completed."
            )

    return get_run(run_id)


def fail_run(
    run_id: str,
    error_code: str,
    error_details: dict[str, Any],
    duration_ms: int,
) -> dict[str, Any]:
    """Record a safe terminal failure for one running pipeline execution.

    Args:
        run_id: Stable identifier of the running pipeline execution.
        error_code: Machine-readable failure category.
        error_details: Safe JSON details without raw exceptions or traces.
        duration_ms: Non-negative elapsed execution time in milliseconds.

    Returns:
        Materialized failed run.

    Raises:
        InvalidRunStateError: If the run is absent or no longer running.
    """
    completed_at = _utc_timestamp()
    error_details_json = _canonical_json(error_details)

    # Conditional update preserves immutable terminal states under retries or races.
    with connect() as connection:
        cursor = connection.execute(
            """
            UPDATE pipeline_run
            SET status = 'failed', completed_at = ?, duration_ms = ?,
                error_code = ?, error_details_json = ?
            WHERE id = ? AND status = 'running'
            """,
            (
                completed_at,
                duration_ms,
                error_code,
                error_details_json,
                run_id,
            ),
        )

        # Never rewrite an already-terminal historical run.
        if cursor.rowcount != 1:
            raise InvalidRunStateError(
                f"Pipeline run '{run_id}' cannot transition to failed."
            )

    return get_run(run_id)


def get_run(run_id: str) -> dict[str, Any]:
    """Load one run together with its linked chunk-set summary.

    Args:
        run_id: Stable identifier of the pipeline run to materialize.

    Returns:
        JSON-friendly run dictionary with optional chunking details.

    Raises:
        RunNotFoundError: If the run does not exist.
    """
    # Join the reusable artifact so transport code never performs persistence queries.
    with connect() as connection:
        row = connection.execute(
            """
            SELECT pipeline_run.id, pipeline_run.corpus_id,
                   pipeline_run.chunk_set_id, pipeline_run.question,
                   pipeline_run.effective_config_json, pipeline_run.status,
                   pipeline_run.chunk_set_reused, pipeline_run.created_at,
                   pipeline_run.started_at, pipeline_run.completed_at,
                   pipeline_run.duration_ms, pipeline_run.error_code,
                   pipeline_run.error_details_json, chunk_set.status AS chunk_set_status,
                   chunk_set.chunk_count
            FROM pipeline_run
            LEFT JOIN chunk_set ON chunk_set.id = pipeline_run.chunk_set_id
            WHERE pipeline_run.id = ?
            """,
            (run_id,),
        ).fetchone()

    # Keep absence distinct from an invalid lifecycle transition.
    if row is None:
        raise RunNotFoundError(run_id)

    chunking: dict[str, Any] | None = None

    # Completed runs expose only compact artifact metadata, never complete chunk text.
    if row["chunk_set_id"] is not None:
        chunking = {
            "chunk_set_id": row["chunk_set_id"],
            "status": row["chunk_set_status"],
            "chunk_count": row["chunk_count"],
            "reused": bool(row["chunk_set_reused"]),
        }

    return {
        "id": row["id"],
        "corpus_id": row["corpus_id"],
        "question": row["question"],
        "configuration": json.loads(row["effective_config_json"]),
        "status": row["status"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
        "duration_ms": row["duration_ms"],
        "error_code": row["error_code"],
        "error_details": json.loads(row["error_details_json"])
        if row["error_details_json"] is not None
        else None,
        "chunking": chunking,
    }
