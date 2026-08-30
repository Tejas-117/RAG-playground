"""SQLite persistence boundaries for queued immutable pipeline runs."""

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from backend.db.connection import connect
from backend.db.repositories.generation_results import (
    get_generation_result_for_run,
    insert_generation_result,
)
from backend.db.repositories.retrieval_results import (
    RetrievalArtifactMismatchError,
    get_hydrated_retrieval_result_for_run,
    insert_retrieval_result,
)
from backend.generation.models import GenerationServiceResult
from backend.pipeline.configs import (
    DistanceMetric,
    GenerationConfig,
    PipelineConfig,
)
from backend.retrieval.models import HydratedVectorSearchHit


class CorpusNotFoundError(LookupError):
    """Report that a run references a corpus that does not exist."""


class RunNotFoundError(LookupError):
    """Report that a lifecycle operation references an unknown run."""


class InvalidRunStateError(RuntimeError):
    """Report a pipeline-run transition from an incompatible state or stage."""


class ChunkSetNotReadyError(RuntimeError):
    """Report that a stage references a chunk set that is not ready."""


class VectorIndexNotReadyError(RuntimeError):
    """Report that run completion references a vector index that is not ready."""


class VectorIndexArtifactMismatchError(RuntimeError):
    """Report a ready vector index built from a different run chunk set."""


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
    # Stable JSON keeps snapshots and tests independent of dictionary insertion order.
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def create_pending_run(
    corpus_id: str,
    question: str,
    configuration: PipelineConfig,
) -> dict[str, Any]:
    """Persist one validated run before the background worker claims it.

    Args:
        corpus_id: Stable identifier of the immutable corpus selected for the run.
        question: Normalized non-empty question submitted by the user.
        configuration: Typed configuration with every backend default resolved.

    Returns:
        Materialized pending run with its immutable configuration snapshot.

    Raises:
        CorpusNotFoundError: If the selected corpus does not exist.
    """
    run_id = str(uuid4())
    created_at = _utc_timestamp()
    effective_config_json = _canonical_json(configuration.model_dump(mode="json"))

    # Validate the corpus and enqueue the run in one transaction.
    with connect() as connection:
        corpus = connection.execute(
            "SELECT id FROM corpus WHERE id = ?", (corpus_id,)
        ).fetchone()

        # Surface a domain error instead of leaking a foreign-key failure.
        if corpus is None:
            raise CorpusNotFoundError(corpus_id)

        connection.execute(
            """
            INSERT INTO pipeline_run (
                id, corpus_id, chunk_set_id, vector_index_id, question,
                effective_config_json, status, current_stage,
                chunk_set_reused, vector_index_reused,
                chunking_duration_ms, embedding_duration_ms,
                retrieval_duration_ms, generation_duration_ms,
                created_at, started_at, completed_at, duration_ms,
                error_code, error_details_json
            ) VALUES (
                ?, ?, NULL, NULL, ?, ?, 'pending', NULL,
                NULL, NULL, NULL, NULL, NULL, NULL, ?, NULL, NULL, NULL, NULL, NULL
            )
            """,
            (run_id, corpus_id, question, effective_config_json, created_at),
        )

    return get_run(run_id)


def claim_next_pending_run() -> dict[str, Any] | None:
    """Atomically claim the oldest queued run for background execution.

    Args:
        None.

    Returns:
        Claimed running run, or ``None`` when the queue is empty.
    """
    started_at = _utc_timestamp()

    # An immediate transaction serializes competing workers before queue selection.
    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT id FROM pipeline_run
            WHERE status = 'pending'
            ORDER BY created_at, id
            LIMIT 1
            """
        ).fetchone()

        # Leave the queue unchanged when there is no work to claim.
        if row is None:
            return None

        run_id = row["id"]
        cursor = connection.execute(
            """
            UPDATE pipeline_run
            SET status = 'running', current_stage = 'chunking', started_at = ?
            WHERE id = ? AND status = 'pending'
            """,
            (started_at, run_id),
        )

        # A serialized claim should update exactly the selected pending row.
        if cursor.rowcount != 1:
            raise InvalidRunStateError(f"Pipeline run '{run_id}' could not be claimed.")

    return get_run(run_id)


def fail_interrupted_runs() -> int:
    """Fail runs abandoned while the previous backend process was executing.

    Args:
        None.

    Returns:
        Number of stale running rows moved to a terminal failed state.
    """
    completed_at = _utc_timestamp()

    # Capture each active stage so polling clients receive an accurate failure stage.
    with connect() as connection:
        rows = connection.execute(
            "SELECT id, current_stage FROM pipeline_run WHERE status = 'running'"
        ).fetchall()

        # Persist one safe terminal failure for every process-abandoned run.
        for row in rows:
            stage = row["current_stage"] or "chunking"
            error_details = _canonical_json(
                {
                    "stage": stage,
                    "message": "The backend stopped while this run was executing.",
                }
            )
            connection.execute(
                """
                UPDATE pipeline_run
                SET status = 'failed', current_stage = NULL,
                    completed_at = ?, duration_ms = 0,
                    error_code = 'run_interrupted', error_details_json = ?
                WHERE id = ? AND status = 'running'
                """,
                (completed_at, error_details, row["id"]),
            )

    return len(rows)


def get_run_execution_input(run_id: str) -> dict[str, Any]:
    """Load the immutable inputs required by the background executor.

    Args:
        run_id: Stable identifier of the claimed running run.

    Returns:
        Corpus ID, question, and validated effective pipeline configuration.

    Raises:
        RunNotFoundError: If the requested run does not exist.
        InvalidRunStateError: If the run is not actively executing.
    """
    # Read the complete snapshot so the worker never depends on request memory.
    with connect() as connection:
        row = connection.execute(
            """
            SELECT corpus_id, question, effective_config_json, status
            FROM pipeline_run WHERE id = ?
            """,
            (run_id,),
        ).fetchone()

    # Keep absence distinct from an invalid lifecycle transition.
    if row is None:
        raise RunNotFoundError(run_id)

    # Only a successfully claimed row may enter the pipeline executor.
    if row["status"] != "running":
        raise InvalidRunStateError(f"Pipeline run '{run_id}' is not running.")

    return {
        "corpus_id": row["corpus_id"],
        "question": row["question"],
        "configuration": PipelineConfig.model_validate_json(
            row["effective_config_json"]
        ),
    }


def record_chunking_result(
    run_id: str,
    chunk_set_id: str,
    reused: bool,
    duration_ms: int,
) -> dict[str, Any]:
    """Attach a ready chunk artifact and advance a running run to embedding.

    Args:
        run_id: Stable identifier of the running pipeline execution.
        chunk_set_id: Ready reusable chunk artifact produced by chunking.
        reused: Whether the chunk artifact already existed.
        duration_ms: Time this run spent resolving the chunking stage.

    Returns:
        Materialized running run now positioned at embedding.

    Raises:
        ChunkSetNotReadyError: If the supplied chunk artifact is not ready.
        InvalidRunStateError: If the run is not actively chunking.
    """
    # Validate the upstream artifact and stage transition together.
    with connect() as connection:
        chunk_set = connection.execute(
            "SELECT id FROM chunk_set WHERE id = ? AND status = 'ready'",
            (chunk_set_id,),
        ).fetchone()

        # Never expose a partial chunk set to the embedding stage.
        if chunk_set is None:
            raise ChunkSetNotReadyError(chunk_set_id)

        cursor = connection.execute(
            """
            UPDATE pipeline_run
            SET chunk_set_id = ?, chunk_set_reused = ?,
                chunking_duration_ms = ?, current_stage = 'embedding'
            WHERE id = ? AND status = 'running' AND current_stage = 'chunking'
            """,
            (chunk_set_id, int(reused), duration_ms, run_id),
        )

        # Protect terminal or concurrently changed run history.
        if cursor.rowcount != 1:
            raise InvalidRunStateError(
                f"Pipeline run '{run_id}' cannot advance to embedding."
            )

    return get_run(run_id)


def record_embedding_result(
    run_id: str,
    vector_index_id: str,
    vector_index_reused: bool,
    embedding_duration_ms: int,
) -> dict[str, Any]:
    """Attach a ready compatible vector index and advance to retrieval.

    Args:
        run_id: Stable identifier of the running pipeline execution.
        vector_index_id: Ready reusable vector index selected by the executor.
        vector_index_reused: Whether this execution reused the index.
        embedding_duration_ms: Time spent resolving the embedding stage.

    Returns:
        Materialized running run now positioned at retrieval.

    Raises:
        VectorIndexNotReadyError: If the vector index is absent or not ready.
        VectorIndexArtifactMismatchError: If the index uses another chunk set.
        InvalidRunStateError: If the run is not actively embedding.
    """
    # Validate compatibility before exposing the index to query-time retrieval.
    with connect() as connection:
        vector_index = connection.execute(
            """
            SELECT id, chunk_set_id FROM vector_index
            WHERE id = ? AND status = 'ready'
            """,
            (vector_index_id,),
        ).fetchone()

        # Retrieval may only search a fully materialized vector collection.
        if vector_index is None:
            raise VectorIndexNotReadyError(vector_index_id)

        run = connection.execute(
            """
            SELECT status, current_stage, chunk_set_id
            FROM pipeline_run WHERE id = ?
            """,
            (run_id,),
        ).fetchone()

        # Protect terminal, missing, or concurrently changed run history.
        if (
            run is None
            or run["status"] != "running"
            or run["current_stage"] != "embedding"
        ):
            raise InvalidRunStateError(
                f"Pipeline run '{run_id}' cannot advance to retrieval."
            )

        # An index cannot be attached when it was built from another chunk artifact.
        if run["chunk_set_id"] != vector_index["chunk_set_id"]:
            raise VectorIndexArtifactMismatchError(vector_index_id)

        cursor = connection.execute(
            """
            UPDATE pipeline_run
            SET vector_index_id = ?, vector_index_reused = ?,
                embedding_duration_ms = ?, current_stage = 'retrieval'
            WHERE id = ? AND status = 'running' AND current_stage = 'embedding'
            """,
            (
                vector_index_id,
                int(vector_index_reused),
                embedding_duration_ms,
                run_id,
            ),
        )

        # A concurrent lifecycle change must not leave an index partially attached.
        if cursor.rowcount != 1:
            raise InvalidRunStateError(
                f"Pipeline run '{run_id}' cannot advance to retrieval."
            )

    return get_run(run_id)


def record_retrieval_result(
    run_id: str,
    vector_index_id: str,
    requested_top_k: int,
    distance_metric: DistanceMetric | str,
    hits: tuple[HydratedVectorSearchHit, ...],
    retrieval_duration_ms: int,
) -> str:
    """Persist ranked retrieval output and advance its run to generation.

    Args:
        run_id: Stable identifier of the running pipeline execution.
        vector_index_id: Exact ready vector index searched by retrieval.
        requested_top_k: Maximum result count from the immutable run config.
        distance_metric: Raw-distance semantics of the searched vector index.
        hits: Ranked hydrated chunks returned by retrieval.
        retrieval_duration_ms: Time spent resolving the retrieval stage.

    Returns:
        Stable identifier of the persisted immutable retrieval result.

    Raises:
        InvalidRunStateError: If the run is not actively retrieving.
        RetrievalArtifactMismatchError: If the run references another index.
        InvalidRetrievalResultError: If retrieval result values are malformed.
        sqlite3.IntegrityError: If relational persistence rejects any write.
    """
    # One transaction prevents partial ranked output or a false stage transition.
    with connect() as connection:
        run = connection.execute(
            """
            SELECT status, current_stage, vector_index_id
            FROM pipeline_run WHERE id = ?
            """,
            (run_id,),
        ).fetchone()

        # Only the active retrieval stage may create the run's immutable result.
        if (
            run is None
            or run["status"] != "running"
            or run["current_stage"] != "retrieval"
        ):
            raise InvalidRunStateError(
                f"Pipeline run '{run_id}' cannot advance to generation."
            )

        # Reject an executor result produced from any index other than the attached one.
        if run["vector_index_id"] != vector_index_id:
            raise RetrievalArtifactMismatchError(
                "The searched vector index does not belong to the pipeline run."
            )

        retrieval_result_id = insert_retrieval_result(
            connection,
            run_id,
            vector_index_id,
            requested_top_k,
            distance_metric,
            retrieval_duration_ms,
            hits,
        )
        cursor = connection.execute(
            """
            UPDATE pipeline_run
            SET retrieval_duration_ms = ?, current_stage = 'generation'
            WHERE id = ? AND status = 'running' AND current_stage = 'retrieval'
            """,
            (retrieval_duration_ms, run_id),
        )

        # Raising here rolls back both retrieval rows and the lifecycle transition.
        if cursor.rowcount != 1:
            raise InvalidRunStateError(
                f"Pipeline run '{run_id}' cannot advance to generation."
            )

    return retrieval_result_id


def complete_run_with_generation(
    run_id: str,
    retrieval_result_id: str,
    generation_config: GenerationConfig,
    generation_result: GenerationServiceResult,
    generation_duration_ms: int,
    total_duration_ms: int,
) -> dict[str, Any]:
    """Persist one generated answer and complete its run atomically.

    Args:
        run_id: Stable identifier of the running pipeline execution.
        retrieval_result_id: Exact persisted retrieval output used for the prompt.
        generation_config: Immutable effective provider and sampling configuration.
        generation_result: Answer plus exact prompt/provider provenance.
        generation_duration_ms: Time spent resolving the generation stage.
        total_duration_ms: Total background pipeline duration through generation.

    Returns:
        Materialized completed run retaining all successful stage artifacts.

    Raises:
        InvalidRunStateError: If the run is not actively generating.
        GenerationArtifactMismatchError: If retrieval belongs to another run.
        InvalidGenerationResultError: If generated values are malformed.
        sqlite3.IntegrityError: If relational persistence rejects any write.
    """
    completed_at = _utc_timestamp()

    # One transaction prevents a partial answer or a falsely completed run.
    with connect() as connection:
        run = connection.execute(
            """
            SELECT status, current_stage FROM pipeline_run WHERE id = ?
            """,
            (run_id,),
        ).fetchone()

        # Only the active generation stage may create the run's immutable answer.
        if (
            run is None
            or run["status"] != "running"
            or run["current_stage"] != "generation"
        ):
            raise InvalidRunStateError(
                f"Pipeline run '{run_id}' cannot transition to completed."
            )

        insert_generation_result(
            connection,
            run_id,
            retrieval_result_id,
            generation_config,
            generation_result,
            generation_duration_ms,
        )
        cursor = connection.execute(
            """
            UPDATE pipeline_run
            SET generation_duration_ms = ?, status = 'completed',
                current_stage = NULL, completed_at = ?, duration_ms = ?,
                error_code = NULL, error_details_json = NULL
            WHERE id = ? AND status = 'running' AND current_stage = 'generation'
            """,
            (generation_duration_ms, completed_at, total_duration_ms, run_id),
        )

        # Raising here rolls back both the answer and lifecycle completion.
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
        duration_ms: Non-negative total execution time in milliseconds.

    Returns:
        Materialized failed run retaining any successful upstream artifact.

    Raises:
        InvalidRunStateError: If the run is absent or no longer running.
    """
    completed_at = _utc_timestamp()
    error_details_json = _canonical_json(error_details)

    # Conditional update protects immutable terminal history from later retries.
    with connect() as connection:
        cursor = connection.execute(
            """
            UPDATE pipeline_run
            SET status = 'failed', current_stage = NULL,
                completed_at = ?, duration_ms = ?,
                error_code = ?, error_details_json = ?
            WHERE id = ? AND status = 'running'
            """,
            (completed_at, duration_ms, error_code, error_details_json, run_id),
        )

        # Never rewrite an already-terminal historical run.
        if cursor.rowcount != 1:
            raise InvalidRunStateError(
                f"Pipeline run '{run_id}' cannot transition to failed."
            )

    return get_run(run_id)


def get_run(run_id: str) -> dict[str, Any]:
    """Load one run with stage state and compact reusable-artifact summaries.

    Args:
        run_id: Stable identifier of the pipeline run to materialize.

    Returns:
        JSON-friendly run dictionary suitable for API polling.

    Raises:
        RunNotFoundError: If the run does not exist.
    """
    # Join both reusable artifacts so API code does not perform persistence queries.
    with connect() as connection:
        row = connection.execute(
            """
            SELECT pipeline_run.*,
                   chunk_set.status AS chunk_set_status,
                   chunk_set.chunk_count,
                   vector_index.status AS vector_index_status,
                   vector_index.vector_count,
                   vector_index.dimensions,
                   vector_index.provider AS index_provider,
                   vector_index.model AS index_model,
                   vector_index.distance_metric AS index_distance_metric
            FROM pipeline_run
            LEFT JOIN chunk_set ON chunk_set.id = pipeline_run.chunk_set_id
            LEFT JOIN vector_index
                ON vector_index.id = pipeline_run.vector_index_id
            WHERE pipeline_run.id = ?
            """,
            (run_id,),
        ).fetchone()

    # Keep a missing run distinct from every lifecycle state.
    if row is None:
        raise RunNotFoundError(run_id)

    retrieval_result = get_hydrated_retrieval_result_for_run(run_id)
    generation_result = get_generation_result_for_run(run_id)
    configuration = json.loads(row["effective_config_json"])
    error_details = (
        json.loads(row["error_details_json"])
        if row["error_details_json"] is not None
        else None
    )
    failed_stage = error_details.get("stage") if error_details else None

    # Derive chunking state from the persisted stage and attached artifact.
    if row["chunk_set_id"] is not None:
        chunking_status = "completed"
    elif row["status"] == "running" and row["current_stage"] == "chunking":
        chunking_status = "running"
    elif row["status"] == "failed" and failed_stage == "chunking":
        chunking_status = "failed"
    else:
        chunking_status = "pending"

    # Embedding can start only after chunking has attached its ready artifact.
    if row["vector_index_id"] is not None:
        embedding_status = "completed"
    elif row["status"] == "running" and row["current_stage"] == "embedding":
        embedding_status = "running"
    elif row["status"] == "failed" and failed_stage == "embedding":
        embedding_status = "failed"
    else:
        embedding_status = "pending"

    # A persisted retrieval parent proves the complete ranked result was committed.
    if retrieval_result is not None:
        retrieval_status = "completed"
    elif row["status"] == "running" and row["current_stage"] == "retrieval":
        retrieval_status = "running"
    elif row["status"] == "failed" and failed_stage == "retrieval":
        retrieval_status = "failed"
    else:
        retrieval_status = "pending"

    # Generation completes only when its immutable answer is present.
    if generation_result is not None:
        generation_status = "completed"
    elif row["status"] == "running" and row["current_stage"] == "generation":
        generation_status = "running"
    elif row["status"] == "failed" and failed_stage == "generation":
        generation_status = "failed"
    else:
        generation_status = "pending"

    error: dict[str, Any] | None = None

    # Materialize one safe polling error without exposing serialized storage details.
    if row["error_code"] is not None:
        error = {
            "code": row["error_code"],
            "message": error_details.get("message", "The pipeline run failed."),
            "stage": failed_stage,
            "details": {
                key: value
                for key, value in (error_details or {}).items()
                if key not in {"message", "stage"}
            },
        }

    embedding_configuration = configuration["embedding"]
    return {
        "id": row["id"],
        "corpus_id": row["corpus_id"],
        "question": row["question"],
        "configuration": configuration,
        "status": row["status"],
        "current_stage": row["current_stage"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
        "duration_ms": row["duration_ms"],
        "chunking": {
            "status": chunking_status,
            "chunk_set_id": row["chunk_set_id"],
            "chunk_count": row["chunk_count"],
            "reused": bool(row["chunk_set_reused"])
            if row["chunk_set_reused"] is not None
            else None,
            "duration_ms": row["chunking_duration_ms"],
        },
        "embedding": {
            "status": embedding_status,
            "vector_index_id": row["vector_index_id"],
            "vector_count": row["vector_count"],
            "dimensions": row["dimensions"],
            "provider": row["index_provider"] or embedding_configuration["provider"],
            "model": row["index_model"] or embedding_configuration["model"],
            "distance_metric": row["index_distance_metric"]
            or embedding_configuration["distance_metric"],
            "reused": bool(row["vector_index_reused"])
            if row["vector_index_reused"] is not None
            else None,
            "duration_ms": row["embedding_duration_ms"],
        },
        "retrieval": {
            "status": retrieval_status,
            "result_id": retrieval_result["id"] if retrieval_result else None,
            "requested_top_k": configuration["retrieval"]["top_k"],
            "returned_count": retrieval_result["returned_count"]
            if retrieval_result
            else None,
            "distance_metric": retrieval_result["distance_metric"]
            if retrieval_result
            else embedding_configuration["distance_metric"],
            "duration_ms": row["retrieval_duration_ms"],
            "chunks": retrieval_result["hits"] if retrieval_result else [],
        },
        "generation": {
            "status": generation_status,
            "result_id": generation_result["id"] if generation_result else None,
            "retrieval_result_id": generation_result["retrieval_result_id"]
            if generation_result
            else None,
            "provider": generation_result["provider"]
            if generation_result
            else configuration["generation"]["provider"],
            "model": generation_result["model"]
            if generation_result
            else configuration["generation"]["model"],
            "provider_model": generation_result["provider_model"]
            if generation_result
            else None,
            "answer": generation_result["answer"] if generation_result else None,
            "finish_reason": generation_result["finish_reason"]
            if generation_result
            else None,
            "prompt_template_version": generation_result["prompt_template_version"]
            if generation_result
            else None,
            "provider_policy_version": generation_result["provider_policy_version"]
            if generation_result
            else None,
            "prompt_tokens": generation_result["prompt_tokens"]
            if generation_result
            else None,
            "completion_tokens": generation_result["completion_tokens"]
            if generation_result
            else None,
            "total_tokens": generation_result["total_tokens"]
            if generation_result
            else None,
            "provider_called": generation_result["provider_called"]
            if generation_result
            else None,
            "context_chunks": generation_result["context_chunks"]
            if generation_result
            else [],
            "duration_ms": row["generation_duration_ms"],
        },
        "error": error,
    }
