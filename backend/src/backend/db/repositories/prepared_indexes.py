"""SQLite persistence boundaries for named prepared vector indexes."""

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from backend.db.connection import connect
from backend.pipeline.configs import PreparationConfig


class PreparedIndexNotFoundError(LookupError):
    """Report that a preparation lifecycle operation references an unknown row."""


class PreparedIndexCorpusNotFoundError(LookupError):
    """Report that a prepared index references a corpus that does not exist."""


class InvalidPreparedIndexStateError(RuntimeError):
    """Report an invalid prepared-index lifecycle transition."""


def _utc_timestamp() -> str:
    """Create the UTC timestamp format used by prepared-index records.

    Args:
        None.

    Returns:
        ISO-8601 UTC timestamp ending in ``Z``.
    """
    # Timezone-aware values keep queue ordering and lifecycle timestamps unambiguous.
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _canonical_json(value: Any) -> str:
    """Serialize one JSON-compatible value in deterministic key order.

    Args:
        value: Configuration or safe error details to persist.

    Returns:
        Compact canonical JSON text.
    """
    # Stable serialization preserves immutable snapshots and deterministic tests.
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def create_pending_prepared_index(
    name: str,
    corpus_id: str,
    configuration: PreparationConfig,
) -> dict[str, Any]:
    """Persist one named preparation request before asynchronous execution.

    Args:
        name: Trimmed user-facing name; duplicates are intentionally allowed.
        corpus_id: Stable immutable corpus selected for preparation.
        configuration: Resolved chunking and embedding configuration snapshot.

    Returns:
        Materialized pending prepared index suitable for polling.

    Raises:
        PreparedIndexCorpusNotFoundError: If the selected corpus is unknown.
    """
    prepared_index_id = str(uuid4())
    created_at = _utc_timestamp()
    effective_config_json = _canonical_json(configuration.model_dump(mode="json"))

    # Validate the parent and enqueue the request in the same transaction.
    with connect() as connection:
        corpus = connection.execute(
            "SELECT id FROM corpus WHERE id = ?",
            (corpus_id,),
        ).fetchone()

        # Raise a domain error instead of exposing a foreign-key constraint failure.
        if corpus is None:
            raise PreparedIndexCorpusNotFoundError(corpus_id)

        connection.execute(
            """
            INSERT INTO prepared_index (
                id, name, corpus_id, chunk_set_id, vector_index_id,
                effective_config_json, status, current_stage,
                chunk_set_reused, vector_index_reused,
                chunking_duration_ms, embedding_duration_ms,
                created_at, started_at, completed_at, duration_ms,
                error_code, error_details_json
            ) VALUES (
                ?, ?, ?, NULL, NULL, ?, 'pending', NULL,
                NULL, NULL, NULL, NULL, ?, NULL, NULL, NULL, NULL, NULL
            )
            """,
            (
                prepared_index_id,
                name,
                corpus_id,
                effective_config_json,
                created_at,
            ),
        )

    return get_prepared_index(prepared_index_id)


def claim_pending_prepared_index(prepared_index_id: str) -> dict[str, Any]:
    """Atomically claim one known pending preparation request.

    Args:
        prepared_index_id: Stable identifier selected by the shared queue.

    Returns:
        Materialized running prepared index.

    Raises:
        InvalidPreparedIndexStateError: If the row is no longer pending.
    """
    started_at = _utc_timestamp()

    # Compare-and-set prevents two workers from claiming the same durable request.
    with connect() as connection:
        cursor = connection.execute(
            """
            UPDATE prepared_index
            SET status = 'running', current_stage = 'chunking', started_at = ?
            WHERE id = ? AND status = 'pending'
            """,
            (started_at, prepared_index_id),
        )

        # Exactly one pending row must move to running for a valid claim.
        if cursor.rowcount != 1:
            raise InvalidPreparedIndexStateError(
                f"Prepared index '{prepared_index_id}' could not be claimed."
            )

    return get_prepared_index(prepared_index_id)


def claim_next_pending_prepared_index() -> dict[str, Any] | None:
    """Atomically claim the oldest prepared-index request.

    Args:
        None.

    Returns:
        Materialized running prepared index, or ``None`` for an empty queue.
    """
    started_at = _utc_timestamp()

    # An immediate transaction serializes selection and update for direct callers.
    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT id FROM prepared_index
            WHERE status = 'pending'
            ORDER BY created_at, id
            LIMIT 1
            """
        ).fetchone()

        # Preserve the queue when no preparation request is waiting.
        if row is None:
            return None

        cursor = connection.execute(
            """
            UPDATE prepared_index
            SET status = 'running', current_stage = 'chunking', started_at = ?
            WHERE id = ? AND status = 'pending'
            """,
            (started_at, row["id"]),
        )

        # A serialized claim must update exactly the selected pending row.
        if cursor.rowcount != 1:
            raise InvalidPreparedIndexStateError(
                f"Prepared index '{row['id']}' could not be claimed."
            )

    return get_prepared_index(row["id"])


def get_prepared_index_execution_input(
    prepared_index_id: str,
) -> dict[str, Any]:
    """Load the immutable inputs for one already-claimed preparation request.

    Args:
        prepared_index_id: Stable running prepared-index identifier.

    Returns:
        Corpus identifier and validated preparation configuration.

    Raises:
        PreparedIndexNotFoundError: If the identifier is unknown.
        InvalidPreparedIndexStateError: If the request is not running.
    """
    # Read only the durable instructions required by the preparation executor.
    with connect() as connection:
        row = connection.execute(
            """
            SELECT corpus_id, effective_config_json, status
            FROM prepared_index WHERE id = ?
            """,
            (prepared_index_id,),
        ).fetchone()

    # Distinguish an unknown identifier from an invalid lifecycle state.
    if row is None:
        raise PreparedIndexNotFoundError(prepared_index_id)

    # Executors may only operate on work atomically claimed by the queue.
    if row["status"] != "running":
        raise InvalidPreparedIndexStateError(
            f"Prepared index '{prepared_index_id}' is not running."
        )

    return {
        "corpus_id": row["corpus_id"],
        "configuration": PreparationConfig.model_validate_json(
            row["effective_config_json"]
        ),
    }


def record_prepared_chunking_result(
    prepared_index_id: str,
    chunk_set_id: str,
    reused: bool,
    duration_ms: int,
) -> None:
    """Attach a ready chunk set and advance preparation to embedding.

    Args:
        prepared_index_id: Stable running preparation identifier.
        chunk_set_id: Exact ready chunk artifact selected by the stage.
        reused: Whether the artifact existed before this request.
        duration_ms: Time spent resolving the chunking stage.

    Returns:
        None. The request remains running in the embedding stage.

    Raises:
        InvalidPreparedIndexStateError: If state or artifact readiness is invalid.
    """
    # Validate artifact ownership and transition the request in one transaction.
    with connect() as connection:
        prepared_index = connection.execute(
            """
            SELECT corpus_id, status, current_stage
            FROM prepared_index WHERE id = ?
            """,
            (prepared_index_id,),
        ).fetchone()
        chunk_set = connection.execute(
            """
            SELECT corpus_id, status FROM chunk_set WHERE id = ?
            """,
            (chunk_set_id,),
        ).fetchone()

        # Only a claimed chunking-stage request may advance.
        if (
            prepared_index is None
            or prepared_index["status"] != "running"
            or prepared_index["current_stage"] != "chunking"
        ):
            raise InvalidPreparedIndexStateError(
                f"Prepared index '{prepared_index_id}' is not chunking."
            )

        # The selected ready chunk set must belong to the request's exact corpus.
        if (
            chunk_set is None
            or chunk_set["status"] != "ready"
            or chunk_set["corpus_id"] != prepared_index["corpus_id"]
        ):
            raise InvalidPreparedIndexStateError(
                "The prepared index cannot use the selected chunk set."
            )

        connection.execute(
            """
            UPDATE prepared_index
            SET chunk_set_id = ?, chunk_set_reused = ?,
                chunking_duration_ms = ?, current_stage = 'embedding'
            WHERE id = ?
            """,
            (chunk_set_id, int(reused), duration_ms, prepared_index_id),
        )


def complete_prepared_index(
    prepared_index_id: str,
    vector_index_id: str,
    reused: bool,
    embedding_duration_ms: int,
    total_duration_ms: int,
) -> dict[str, Any]:
    """Attach the compatible vector artifact and mark preparation ready.

    Args:
        prepared_index_id: Stable running preparation identifier.
        vector_index_id: Exact ready technical vector-index artifact.
        reused: Whether the vector index existed before this request.
        embedding_duration_ms: Time spent resolving the embedding stage.
        total_duration_ms: Total preparation execution time.

    Returns:
        Materialized ready prepared index.

    Raises:
        InvalidPreparedIndexStateError: If state or artifact lineage is invalid.
    """
    completed_at = _utc_timestamp()

    # Validate the vector index against the attached chunk set before completion.
    with connect() as connection:
        prepared_index = connection.execute(
            """
            SELECT status, current_stage, chunk_set_id
            FROM prepared_index WHERE id = ?
            """,
            (prepared_index_id,),
        ).fetchone()
        vector_index = connection.execute(
            """
            SELECT chunk_set_id, status FROM vector_index WHERE id = ?
            """,
            (vector_index_id,),
        ).fetchone()

        # Completion is only legal after the chunking result has been recorded.
        if (
            prepared_index is None
            or prepared_index["status"] != "running"
            or prepared_index["current_stage"] != "embedding"
        ):
            raise InvalidPreparedIndexStateError(
                f"Prepared index '{prepared_index_id}' is not embedding."
            )

        # Prevent a user-facing reference from linking unrelated artifact lineage.
        if (
            vector_index is None
            or vector_index["status"] != "ready"
            or vector_index["chunk_set_id"] != prepared_index["chunk_set_id"]
        ):
            raise InvalidPreparedIndexStateError(
                "The prepared index cannot use the selected vector index."
            )

        connection.execute(
            """
            UPDATE prepared_index
            SET vector_index_id = ?, vector_index_reused = ?,
                embedding_duration_ms = ?, status = 'ready',
                current_stage = NULL, completed_at = ?, duration_ms = ?
            WHERE id = ?
            """,
            (
                vector_index_id,
                int(reused),
                embedding_duration_ms,
                completed_at,
                total_duration_ms,
                prepared_index_id,
            ),
        )

    return get_prepared_index(prepared_index_id)


def fail_prepared_index(
    prepared_index_id: str,
    error_code: str,
    error_details: dict[str, Any],
    duration_ms: int,
) -> dict[str, Any]:
    """Persist a safe terminal failure for one running preparation request.

    Args:
        prepared_index_id: Stable running preparation identifier.
        error_code: Machine-readable public failure category.
        error_details: Safe structured details including stage and message.
        duration_ms: Total execution time before failure.

    Returns:
        Materialized failed prepared index.

    Raises:
        InvalidPreparedIndexStateError: If the request is not running.
    """
    completed_at = _utc_timestamp()

    # Keep any completed chunking artifact attached while terminalizing the request.
    with connect() as connection:
        cursor = connection.execute(
            """
            UPDATE prepared_index
            SET status = 'failed', current_stage = NULL,
                completed_at = ?, duration_ms = ?,
                error_code = ?, error_details_json = ?
            WHERE id = ? AND status = 'running'
            """,
            (
                completed_at,
                duration_ms,
                error_code,
                _canonical_json(error_details),
                prepared_index_id,
            ),
        )

        # Never overwrite an already-terminal or unclaimed request.
        if cursor.rowcount != 1:
            raise InvalidPreparedIndexStateError(
                f"Prepared index '{prepared_index_id}' cannot be failed."
            )

    return get_prepared_index(prepared_index_id)


def fail_interrupted_prepared_indexes() -> int:
    """Fail preparation work abandoned by a previous backend process.

    Args:
        None.

    Returns:
        Number of stale running requests moved to a terminal failed state.
    """
    completed_at = _utc_timestamp()

    # Capture the active stage so clients receive an accurate restart failure.
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id, current_stage FROM prepared_index WHERE status = 'running'
            """
        ).fetchall()

        # Persist one safe failure for each request abandoned during shutdown.
        for row in rows:
            stage = row["current_stage"] or "chunking"
            details = _canonical_json(
                {
                    "stage": stage,
                    "message": (
                        "The backend stopped while this index was being prepared."
                    ),
                }
            )
            connection.execute(
                """
                UPDATE prepared_index
                SET status = 'failed', current_stage = NULL,
                    completed_at = ?, duration_ms = 0,
                    error_code = 'preparation_interrupted',
                    error_details_json = ?
                WHERE id = ? AND status = 'running'
                """,
                (completed_at, details, row["id"]),
            )

    return len(rows)


def get_prepared_index(prepared_index_id: str) -> dict[str, Any]:
    """Load one named prepared index with reusable artifact summaries.

    Args:
        prepared_index_id: Stable user-facing prepared-index identifier.

    Returns:
        Materialized lifecycle record and artifact summaries.

    Raises:
        PreparedIndexNotFoundError: If the identifier is unknown.
    """
    # Join summaries without loading every chunk or any vector coordinates.
    with connect() as connection:
        row = connection.execute(
            _PREPARED_INDEX_SELECT + " WHERE prepared_index.id = ?",
            (prepared_index_id,),
        ).fetchone()

    # Keep not-found handling explicit at repository and API boundaries.
    if row is None:
        raise PreparedIndexNotFoundError(prepared_index_id)

    return _materialize_prepared_index(row)


def list_prepared_indexes(status: str | None = None) -> list[dict[str, Any]]:
    """List named prepared indexes newest first, optionally by status.

    Args:
        status: Optional exact lifecycle status selected by the caller.

    Returns:
        Materialized prepared-index summaries ordered by creation and ID.
    """
    # Use a bound filter when requested and otherwise retain every lifecycle state.
    if status is None:
        query = _PREPARED_INDEX_SELECT + (
            " ORDER BY prepared_index.created_at DESC, prepared_index.id DESC"
        )
        parameters: tuple[str, ...] = ()
    else:
        query = _PREPARED_INDEX_SELECT + (
            " WHERE prepared_index.status = ?"
            " ORDER BY prepared_index.created_at DESC, prepared_index.id DESC"
        )
        parameters = (status,)

    # List views load only relational summaries and immutable configuration JSON.
    with connect() as connection:
        rows = connection.execute(query, parameters).fetchall()

    return [_materialize_prepared_index(row) for row in rows]


# One shared select keeps detail and list response fields identical.
_PREPARED_INDEX_SELECT = """
    SELECT prepared_index.*,
           chunk_set.chunk_count,
           vector_index.vector_count,
           vector_index.dimensions
    FROM prepared_index
    LEFT JOIN chunk_set ON chunk_set.id = prepared_index.chunk_set_id
    LEFT JOIN vector_index ON vector_index.id = prepared_index.vector_index_id
"""


def _materialize_prepared_index(row: Any) -> dict[str, Any]:
    """Convert one joined SQLite row into the API-facing dictionary shape.

    Args:
        row: Joined prepared-index and artifact-summary row.

    Returns:
        JSON-friendly dictionary with configuration, stages, and safe error.
    """
    configuration = json.loads(row["effective_config_json"])
    error_details = (
        json.loads(row["error_details_json"])
        if row["error_details_json"] is not None
        else None
    )

    # Separate the standard error fields from optional structured details.
    error = None
    if error_details is not None:
        error = {
            "code": row["error_code"],
            "message": error_details.pop(
                "message",
                "The prepared index could not be completed.",
            ),
            "stage": error_details.pop("stage", None),
            "details": error_details,
        }

    return {
        "id": row["id"],
        "name": row["name"],
        "corpus_id": row["corpus_id"],
        "configuration": configuration,
        "status": row["status"],
        "current_stage": row["current_stage"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
        "duration_ms": row["duration_ms"],
        "chunking": {
            "chunk_set_id": row["chunk_set_id"],
            "chunk_count": row["chunk_count"],
            "reused": (
                bool(row["chunk_set_reused"])
                if row["chunk_set_reused"] is not None
                else None
            ),
            "duration_ms": row["chunking_duration_ms"],
        },
        "embedding": {
            "vector_index_id": row["vector_index_id"],
            "vector_count": row["vector_count"],
            "dimensions": row["dimensions"],
            "reused": (
                bool(row["vector_index_reused"])
                if row["vector_index_reused"] is not None
                else None
            ),
            "duration_ms": row["embedding_duration_ms"],
        },
        "error": error,
    }
