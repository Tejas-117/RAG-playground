"""SQLite persistence for dataset-wide benchmark runs and example executions."""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from backend.db.connection import connect
from backend.generation.models import GenerationServiceResult
from backend.pipeline.configs import ExperimentConfig, PipelineConfig, PreparationConfig
from backend.retrieval.models import HydratedVectorSearchHit


class BenchmarkRunNotFoundError(LookupError):
    """Report an unknown benchmark-run identifier."""


class BenchmarkInputNotFoundError(LookupError):
    """Report an unknown prepared index or evaluation dataset."""


class BenchmarkInputMismatchError(ValueError):
    """Report incompatible corpus lineage or a non-ready prepared index."""


class InvalidBenchmarkStateError(RuntimeError):
    """Report a benchmark or example lifecycle transition from the wrong state."""


def _utc_timestamp() -> str:
    """Return the unambiguous UTC timestamp stored for benchmark lifecycle events."""
    # Match the second-level ISO representation used by other durable work queues.
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace(
            "+00:00",
            "Z",
        )
    )


def _canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible snapshot with deterministic key ordering."""
    # Compact stable JSON makes saved effective configurations directly comparable.
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def resolve_benchmark_configuration(
    prepared_index_id: str,
    dataset_id: str,
    configuration: ExperimentConfig,
) -> tuple[str, str, PipelineConfig]:
    """Resolve immutable lineage and merge preparation with query-time settings.

    Args:
        prepared_index_id: Ready named index selected by the user.
        dataset_id: Immutable evaluation dataset selected for the benchmark.
        configuration: Retrieval, generation, and future evaluation settings.

    Returns:
        Corpus ID, technical vector-index ID, and full effective configuration.

    Raises:
        BenchmarkInputNotFoundError: If either selected resource is unknown.
        BenchmarkInputMismatchError: If resources are incompatible or not ready.
    """
    # Resolve both selected resources together so corpus compatibility is authoritative.
    with connect() as connection:
        prepared_index = connection.execute(
            """
            SELECT corpus_id, vector_index_id, effective_config_json, status
            FROM prepared_index WHERE id = ?
            """,
            (prepared_index_id,),
        ).fetchone()
        dataset = connection.execute(
            "SELECT corpus_id FROM evaluation_dataset WHERE id = ?",
            (dataset_id,),
        ).fetchone()

    # Keep an unknown identity distinct from an incompatible known resource.
    if prepared_index is None or dataset is None:
        raise BenchmarkInputNotFoundError(prepared_index_id, dataset_id)

    # Benchmarks may only search a completed named preparation artifact.
    if prepared_index["status"] != "ready" or prepared_index["vector_index_id"] is None:
        raise BenchmarkInputMismatchError("The selected prepared index is not ready.")

    # Stable corpus identity is the compatibility boundary for labels and chunks.
    if prepared_index["corpus_id"] != dataset["corpus_id"]:
        raise BenchmarkInputMismatchError(
            "The selected dataset and prepared index use different corpora."
        )

    preparation = PreparationConfig.model_validate_json(
        prepared_index["effective_config_json"]
    )
    effective_configuration = PipelineConfig(
        chunking=preparation.chunking,
        embedding=preparation.embedding,
        retrieval=configuration.retrieval,
        generation=configuration.generation,
        evaluation=configuration.evaluation,
    )
    return (
        prepared_index["corpus_id"],
        prepared_index["vector_index_id"],
        effective_configuration,
    )


def create_pending_benchmark_run(
    prepared_index_id: str,
    dataset_id: str,
    configuration: PipelineConfig,
) -> dict[str, Any]:
    """Create one pending benchmark and all ordered example execution records.

    Args:
        prepared_index_id: Ready named index selected for the benchmark.
        dataset_id: Immutable dataset containing every benchmark question.
        configuration: Fully resolved immutable pipeline configuration snapshot.

    Returns:
        Materialized pending benchmark suitable for polling.
    """
    benchmark_run_id = str(uuid4())
    created_at = _utc_timestamp()

    # Insert the aggregate and every child atomically so queue work is never partial.
    with connect() as connection:
        prepared_index = connection.execute(
            """
            SELECT corpus_id, vector_index_id, status
            FROM prepared_index WHERE id = ?
            """,
            (prepared_index_id,),
        ).fetchone()
        dataset = connection.execute(
            "SELECT corpus_id FROM evaluation_dataset WHERE id = ?",
            (dataset_id,),
        ).fetchone()
        examples = connection.execute(
            """
            SELECT id, ordinal FROM evaluation_example
            WHERE dataset_id = ? ORDER BY ordinal
            """,
            (dataset_id,),
        ).fetchall()

        # Recheck launch-time lineage inside the insertion transaction.
        if prepared_index is None or dataset is None:
            raise BenchmarkInputNotFoundError(prepared_index_id, dataset_id)

        if (
            prepared_index["status"] != "ready"
            or prepared_index["vector_index_id"] is None
            or prepared_index["corpus_id"] != dataset["corpus_id"]
        ):
            raise BenchmarkInputMismatchError(
                "The selected dataset and prepared index are not compatible."
            )

        # Imported datasets are non-empty by contract, but protect the queue invariant.
        if not examples:
            raise BenchmarkInputMismatchError(
                "The selected evaluation dataset does not contain any examples."
            )

        connection.execute(
            """
            INSERT INTO benchmark_run (
                id, prepared_index_id, dataset_id, corpus_id, vector_index_id,
                effective_config_json, status, current_stage,
                current_example_id, total_examples, completed_examples,
                created_at, started_at, completed_at, duration_ms,
                error_code, error_details_json
            ) VALUES (
                ?, ?, ?, ?, ?, ?, 'pending', NULL, NULL, ?, 0,
                ?, NULL, NULL, NULL, NULL, NULL
            )
            """,
            (
                benchmark_run_id,
                prepared_index_id,
                dataset_id,
                prepared_index["corpus_id"],
                prepared_index["vector_index_id"],
                _canonical_json(configuration.model_dump(mode="json")),
                len(examples),
                created_at,
            ),
        )

        # Child rows preserve dataset order while remaining internal to one run.
        for example in examples:
            connection.execute(
                """
                INSERT INTO benchmark_example_run (
                    id, benchmark_run_id, evaluation_example_id, ordinal,
                    status, current_stage
                ) VALUES (?, ?, ?, ?, 'pending', NULL)
                """,
                (
                    str(uuid4()),
                    benchmark_run_id,
                    example["id"],
                    example["ordinal"],
                ),
            )

    return get_benchmark_run(benchmark_run_id)


def claim_next_pending_benchmark_run() -> dict[str, Any] | None:
    """Claim the oldest pending benchmark directly for tests or dedicated workers."""
    started_at = _utc_timestamp()

    # An immediate transaction makes oldest-first selection and transition atomic.
    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT id FROM benchmark_run WHERE status = 'pending'
            ORDER BY created_at, id LIMIT 1
            """
        ).fetchone()

        # An empty queue has no lifecycle side effect.
        if row is None:
            return None

        cursor = connection.execute(
            """
            UPDATE benchmark_run
            SET status = 'running', current_stage = 'retrieval', started_at = ?
            WHERE id = ? AND status = 'pending'
            """,
            (started_at, row["id"]),
        )

        # A serialized claim must transition exactly the selected row.
        if cursor.rowcount != 1:
            raise InvalidBenchmarkStateError(
                f"Benchmark run '{row['id']}' could not be claimed."
            )

    return get_benchmark_run(row["id"])


def get_benchmark_execution_input(benchmark_run_id: str) -> dict[str, Any]:
    """Load the immutable index, configuration, and ordered questions for execution."""
    # Read all durable executor inputs instead of relying on request memory.
    with connect() as connection:
        run = connection.execute(
            """
            SELECT status, vector_index_id, effective_config_json
            FROM benchmark_run WHERE id = ?
            """,
            (benchmark_run_id,),
        ).fetchone()

        # Only work atomically claimed by the queue may be executed.
        if run is None:
            raise BenchmarkRunNotFoundError(benchmark_run_id)

        if run["status"] != "running":
            raise InvalidBenchmarkStateError(
                f"Benchmark run '{benchmark_run_id}' is not running."
            )

        vector_index = connection.execute(
            "SELECT * FROM vector_index WHERE id = ? AND status = 'ready'",
            (run["vector_index_id"],),
        ).fetchone()
        examples = connection.execute(
            """
            SELECT benchmark_example_run.id AS example_run_id,
                   benchmark_example_run.ordinal,
                   evaluation_example.id AS example_id,
                   evaluation_example.question,
                   evaluation_example.reference_answer
            FROM benchmark_example_run
            JOIN evaluation_example
              ON evaluation_example.id = benchmark_example_run.evaluation_example_id
            WHERE benchmark_example_run.benchmark_run_id = ?
            ORDER BY benchmark_example_run.ordinal
            """,
            (benchmark_run_id,),
        ).fetchall()

    # A deleted or incomplete technical artifact cannot be searched safely.
    if vector_index is None:
        raise BenchmarkInputMismatchError("The benchmark vector index is not ready.")

    vector_artifact = dict(vector_index)
    vector_artifact["configuration"] = json.loads(
        vector_artifact.pop("embedding_config_json")
    )
    return {
        "configuration": PipelineConfig.model_validate_json(
            run["effective_config_json"]
        ),
        "vector_index": vector_artifact,
        "examples": [dict(example) for example in examples],
    }


def start_benchmark_example(
    benchmark_run_id: str,
    example_run_id: str,
    example_id: str,
) -> None:
    """Move one pending child into retrieval and expose it on the parent run."""
    started_at = _utc_timestamp()

    # Update child and parent in one transaction so progress cannot disagree.
    with connect() as connection:
        cursor = connection.execute(
            """
            UPDATE benchmark_example_run
            SET status = 'running', current_stage = 'retrieval', started_at = ?
            WHERE id = ? AND benchmark_run_id = ? AND status = 'pending'
            """,
            (started_at, example_run_id, benchmark_run_id),
        )

        # Re-execution must not overwrite a completed or failed example.
        if cursor.rowcount != 1:
            raise InvalidBenchmarkStateError(
                f"Benchmark example '{example_run_id}' could not start."
            )

        connection.execute(
            """
            UPDATE benchmark_run
            SET current_stage = 'retrieval', current_example_id = ?
            WHERE id = ? AND status = 'running'
            """,
            (example_id, benchmark_run_id),
        )


def record_benchmark_retrieval(
    benchmark_run_id: str,
    example_run_id: str,
    vector_index_id: str,
    requested_top_k: int,
    distance_metric: str,
    hits: tuple[HydratedVectorSearchHit, ...],
    duration_ms: int,
) -> str:
    """Persist ranked hits and advance one example execution to generation."""
    retrieval_result_id = str(uuid4())
    created_at = _utc_timestamp()

    # Retrieval parent, ranked hits, and lifecycle transition are one transaction.
    with connect() as connection:
        run = connection.execute(
            """
            SELECT vector_index_id FROM benchmark_run
            WHERE id = ? AND status = 'running'
            """,
            (benchmark_run_id,),
        ).fetchone()
        example = connection.execute(
            """
            SELECT status, current_stage FROM benchmark_example_run
            WHERE id = ? AND benchmark_run_id = ?
            """,
            (example_run_id, benchmark_run_id),
        ).fetchone()
        vector_index = connection.execute(
            """
            SELECT chunk_set_id, distance_metric FROM vector_index
            WHERE id = ? AND status = 'ready'
            """,
            (vector_index_id,),
        ).fetchone()

        # Only the active example may save output from the run's exact index.
        if (
            run is None
            or run["vector_index_id"] != vector_index_id
            or example is None
            or example["status"] != "running"
            or example["current_stage"] != "retrieval"
            or vector_index is None
            or vector_index["distance_metric"] != distance_metric
        ):
            raise InvalidBenchmarkStateError("Benchmark retrieval state is invalid.")

        # Result counts and ranks must preserve the vector search contract exactly.
        if len(hits) > requested_top_k or [hit.rank for hit in hits] != list(
            range(1, len(hits) + 1)
        ):
            raise InvalidBenchmarkStateError("Benchmark retrieval ranking is invalid.")

        hit_ids = tuple(hit.chunk_id for hit in hits)

        # Every returned chunk must belong to the exact prepared vector lineage.
        if hit_ids:
            placeholders = ",".join("?" for _ in hit_ids)
            rows = connection.execute(
                f"""
                SELECT id FROM chunk
                WHERE chunk_set_id = ? AND id IN ({placeholders})
                """,
                (vector_index["chunk_set_id"], *hit_ids),
            ).fetchall()

            if {row["id"] for row in rows} != set(hit_ids):
                raise InvalidBenchmarkStateError(
                    "A retrieved chunk does not belong to the benchmark index."
                )

        connection.execute(
            """
            INSERT INTO benchmark_retrieval_result (
                id, example_run_id, vector_index_id, requested_top_k,
                returned_count, distance_metric, duration_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                retrieval_result_id,
                example_run_id,
                vector_index_id,
                requested_top_k,
                len(hits),
                distance_metric,
                duration_ms,
                created_at,
            ),
        )

        # Preserve the vector store's exact ranking and raw score semantics.
        for hit in hits:
            connection.execute(
                """
                INSERT INTO benchmark_retrieved_chunk (
                    retrieval_result_id, rank, chunk_id, raw_distance
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    retrieval_result_id,
                    hit.rank,
                    hit.chunk_id,
                    hit.raw_distance,
                ),
            )

        connection.execute(
            """
            UPDATE benchmark_example_run
            SET retrieval_duration_ms = ?, current_stage = 'generation'
            WHERE id = ?
            """,
            (duration_ms, example_run_id),
        )
        connection.execute(
            """
            UPDATE benchmark_run SET current_stage = 'generation' WHERE id = ?
            """,
            (benchmark_run_id,),
        )

    return retrieval_result_id


def complete_benchmark_example(
    benchmark_run_id: str,
    example_run_id: str,
    retrieval_result_id: str,
    configuration: PipelineConfig,
    result: GenerationServiceResult,
    generation_duration_ms: int,
    example_duration_ms: int,
) -> None:
    """Persist one answer and atomically complete its child execution."""
    generation_result_id = str(uuid4())
    completed_at = _utc_timestamp()
    response = result.response

    # Answer, prompt context, child completion, and parent progress move together.
    with connect() as connection:
        retrieval_rows = connection.execute(
            """
            SELECT rank, chunk_id FROM benchmark_retrieved_chunk
            WHERE retrieval_result_id = ? ORDER BY rank
            """,
            (retrieval_result_id,),
        ).fetchall()
        selected_rows = retrieval_rows[: len(result.context_chunk_ids)]

        # Prompt provenance must be an exact prefix of the persisted ranking.
        if tuple(row["chunk_id"] for row in selected_rows) != result.context_chunk_ids:
            raise InvalidBenchmarkStateError(
                "Generation context does not match persisted retrieval order."
            )

        connection.execute(
            """
            INSERT INTO benchmark_generation_result (
                id, example_run_id, retrieval_result_id, provider, model,
                provider_model, prompt_template_version, provider_policy_version,
                generation_config_json, answer_text, finish_reason,
                prompt_tokens, completion_tokens, total_tokens,
                provider_request_id, system_fingerprint, provider_called,
                duration_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                generation_result_id,
                example_run_id,
                retrieval_result_id,
                configuration.generation.provider,
                configuration.generation.model,
                response.provider_model,
                result.prompt_template_version,
                result.provider_policy_version,
                _canonical_json(configuration.generation.model_dump(mode="json")),
                response.answer_text,
                response.finish_reason,
                response.prompt_tokens,
                response.completion_tokens,
                response.total_tokens,
                response.provider_request_id,
                response.system_fingerprint,
                int(result.provider_called),
                generation_duration_ms,
                completed_at,
            ),
        )

        # Record the exact retrieval ranks consumed by the prompt builder.
        for ordinal, retrieval_row in enumerate(selected_rows, start=1):
            connection.execute(
                """
                INSERT INTO benchmark_generation_context_chunk (
                    generation_result_id, ordinal,
                    retrieval_result_id, retrieval_rank
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    generation_result_id,
                    ordinal,
                    retrieval_result_id,
                    retrieval_row["rank"],
                ),
            )

        cursor = connection.execute(
            """
            UPDATE benchmark_example_run
            SET status = 'completed', current_stage = NULL,
                generation_duration_ms = ?, completed_at = ?, duration_ms = ?
            WHERE id = ? AND status = 'running' AND current_stage = 'generation'
            """,
            (
                generation_duration_ms,
                completed_at,
                example_duration_ms,
                example_run_id,
            ),
        )

        # Failed compare-and-set rolls back every answer-related row above.
        if cursor.rowcount != 1:
            raise InvalidBenchmarkStateError(
                f"Benchmark example '{example_run_id}' could not complete."
            )

        connection.execute(
            """
            UPDATE benchmark_run
            SET completed_examples = completed_examples + 1
            WHERE id = ? AND status = 'running'
            """,
            (benchmark_run_id,),
        )


def complete_benchmark_run(benchmark_run_id: str, duration_ms: int) -> dict[str, Any]:
    """Mark a run completed after every child example has completed successfully."""
    completed_at = _utc_timestamp()

    # The count guard prevents a benchmark from completing with pending children.
    with connect() as connection:
        cursor = connection.execute(
            """
            UPDATE benchmark_run
            SET status = 'completed', current_stage = NULL,
                current_example_id = NULL, completed_at = ?, duration_ms = ?
            WHERE id = ? AND status = 'running'
              AND completed_examples = total_examples
            """,
            (completed_at, duration_ms, benchmark_run_id),
        )

        # Only a fully processed active run may become immutable completed history.
        if cursor.rowcount != 1:
            raise InvalidBenchmarkStateError(
                f"Benchmark run '{benchmark_run_id}' could not complete."
            )

    return get_benchmark_run(benchmark_run_id)


def fail_benchmark_run(
    benchmark_run_id: str,
    example_run_id: str | None,
    error_code: str,
    error_details: dict[str, Any],
    run_duration_ms: int,
    example_duration_ms: int,
) -> dict[str, Any]:
    """Fail the active example and parent while retaining earlier completed output.

    Args:
        benchmark_run_id: Stable parent benchmark identifier.
        example_run_id: Active child execution, if one had started.
        error_code: Safe machine-readable failure category.
        error_details: Safe structured stage and message information.
        run_duration_ms: Total benchmark duration before failure.
        example_duration_ms: Active example duration before failure.

    Returns:
        Materialized failed benchmark retaining earlier completed outputs.
    """
    completed_at = _utc_timestamp()
    encoded_details = _canonical_json(error_details)

    # Terminalize the active child and aggregate without touching completed siblings.
    with connect() as connection:
        if example_run_id is not None:
            connection.execute(
                """
                UPDATE benchmark_example_run
                SET status = 'failed', current_stage = NULL,
                    completed_at = ?, duration_ms = ?,
                    error_code = ?, error_details_json = ?
                WHERE id = ? AND status = 'running'
                """,
                (
                    completed_at,
                    example_duration_ms,
                    error_code,
                    encoded_details,
                    example_run_id,
                ),
            )

        cursor = connection.execute(
            """
            UPDATE benchmark_run
            SET status = 'failed', current_stage = NULL,
                current_example_id = NULL, completed_at = ?, duration_ms = ?,
                error_code = ?, error_details_json = ?
            WHERE id = ? AND status = 'running'
            """,
            (
                completed_at,
                run_duration_ms,
                error_code,
                encoded_details,
                benchmark_run_id,
            ),
        )

        # Never overwrite terminal benchmark history.
        if cursor.rowcount != 1:
            raise InvalidBenchmarkStateError(
                f"Benchmark run '{benchmark_run_id}' could not fail."
            )

    return get_benchmark_run(benchmark_run_id)


def fail_interrupted_benchmark_runs() -> int:
    """Fail benchmark work abandoned by an earlier backend process."""
    completed_at = _utc_timestamp()

    # Recovery updates all active aggregates and children in one transaction.
    with connect() as connection:
        rows = connection.execute(
            "SELECT id, current_stage FROM benchmark_run WHERE status = 'running'"
        ).fetchall()

        # Each abandoned run receives the same safe restart-specific category.
        for row in rows:
            details = _canonical_json(
                {
                    "stage": row["current_stage"],
                    "message": "The backend stopped while this benchmark was running.",
                }
            )
            connection.execute(
                """
                UPDATE benchmark_example_run
                SET status = 'failed', current_stage = NULL,
                    completed_at = ?, duration_ms = 0,
                    error_code = 'run_interrupted', error_details_json = ?
                WHERE benchmark_run_id = ? AND status = 'running'
                """,
                (completed_at, details, row["id"]),
            )
            connection.execute(
                """
                UPDATE benchmark_run
                SET status = 'failed', current_stage = NULL,
                    current_example_id = NULL, completed_at = ?, duration_ms = 0,
                    error_code = 'run_interrupted', error_details_json = ?
                WHERE id = ?
                """,
                (completed_at, details, row["id"]),
            )

    return len(rows)


def list_benchmark_runs() -> list[dict[str, Any]]:
    """Return benchmark summaries newest first without loading question results."""
    # Join user-facing resource names so inventory clients need no extra requests.
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT benchmark_run.*, prepared_index.name AS prepared_index_name,
                   evaluation_dataset.name AS dataset_name
            FROM benchmark_run
            JOIN prepared_index ON prepared_index.id = benchmark_run.prepared_index_id
            JOIN evaluation_dataset ON evaluation_dataset.id = benchmark_run.dataset_id
            ORDER BY benchmark_run.created_at DESC, benchmark_run.id DESC
            """
        ).fetchall()

    return [_benchmark_summary_from_row(row) for row in rows]


def get_benchmark_run(benchmark_run_id: str) -> dict[str, Any]:
    """Return one benchmark with ordered question, retrieval, and answer results."""
    # Load the aggregate and each normalized result type consistently.
    with connect() as connection:
        run = connection.execute(
            """
            SELECT benchmark_run.*, prepared_index.name AS prepared_index_name,
                   evaluation_dataset.name AS dataset_name
            FROM benchmark_run
            JOIN prepared_index ON prepared_index.id = benchmark_run.prepared_index_id
            JOIN evaluation_dataset ON evaluation_dataset.id = benchmark_run.dataset_id
            WHERE benchmark_run.id = ?
            """,
            (benchmark_run_id,),
        ).fetchone()

        # Unknown run identity is distinct from a valid run with no completed examples.
        if run is None:
            raise BenchmarkRunNotFoundError(benchmark_run_id)

        examples = connection.execute(
            """
            SELECT benchmark_example_run.*, evaluation_example.question,
                   evaluation_example.reference_answer
            FROM benchmark_example_run
            JOIN evaluation_example
              ON evaluation_example.id = benchmark_example_run.evaluation_example_id
            WHERE benchmark_example_run.benchmark_run_id = ?
            ORDER BY benchmark_example_run.ordinal
            """,
            (benchmark_run_id,),
        ).fetchall()

        result_examples = [
            _materialize_example(connection, example) for example in examples
        ]

    response = _benchmark_summary_from_row(run)
    response["configuration"] = json.loads(run["effective_config_json"])
    response["examples"] = result_examples
    response["error"] = _materialize_error(run)
    return response


def _benchmark_summary_from_row(row: sqlite3.Row) -> dict[str, Any]:
    """Convert one joined benchmark row into a compact response dictionary."""
    return {
        "id": row["id"],
        "prepared_index_id": row["prepared_index_id"],
        "prepared_index_name": row["prepared_index_name"],
        "dataset_id": row["dataset_id"],
        "dataset_name": row["dataset_name"],
        "corpus_id": row["corpus_id"],
        "vector_index_id": row["vector_index_id"],
        "status": row["status"],
        "current_stage": row["current_stage"],
        "current_example_id": row["current_example_id"],
        "total_examples": row["total_examples"],
        "completed_examples": row["completed_examples"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
        "duration_ms": row["duration_ms"],
    }


def _materialize_error(row: sqlite3.Row) -> dict[str, Any] | None:
    """Decode a safe persisted benchmark or example error when one exists."""
    # Non-failed lifecycle rows intentionally expose no error object.
    if row["error_code"] is None:
        return None

    details = json.loads(row["error_details_json"] or "{}")
    return {
        "code": row["error_code"],
        "message": details.pop("message", "The benchmark could not be completed."),
        "stage": details.pop("stage", None),
        "details": details,
    }


def _materialize_example(
    connection: sqlite3.Connection,
    example: sqlite3.Row,
) -> dict[str, Any]:
    """Hydrate one child execution with ranked chunks and its generated answer."""
    retrieval = connection.execute(
        "SELECT * FROM benchmark_retrieval_result WHERE example_run_id = ?",
        (example["id"],),
    ).fetchone()
    generation = connection.execute(
        "SELECT * FROM benchmark_generation_result WHERE example_run_id = ?",
        (example["id"],),
    ).fetchone()
    retrieval_response = None

    # Hydrate persisted chunk provenance only after retrieval has completed.
    if retrieval is not None:
        hit_rows = connection.execute(
            """
            SELECT benchmark_retrieved_chunk.rank,
                   benchmark_retrieved_chunk.chunk_id,
                   benchmark_retrieved_chunk.raw_distance,
                   chunk.source_document_id, document.original_filename,
                   chunk.ordinal, chunk.text, chunk.character_start_offset,
                   chunk.character_end_offset, chunk.token_start_offset,
                   chunk.token_end_offset, chunk.page_start, chunk.page_end,
                   chunk.section_path_json, chunk.source_metadata_json
            FROM benchmark_retrieved_chunk
            JOIN chunk ON chunk.id = benchmark_retrieved_chunk.chunk_id
            JOIN document ON document.id = chunk.source_document_id
            WHERE benchmark_retrieved_chunk.retrieval_result_id = ?
            ORDER BY benchmark_retrieved_chunk.rank
            """,
            (retrieval["id"],),
        ).fetchall()
        hits: list[dict[str, Any]] = []

        # Decode structured chunk fields while preserving text and exact rank.
        for hit_row in hit_rows:
            hit = dict(hit_row)
            hit["section_path"] = json.loads(hit.pop("section_path_json") or "null")
            hit["source_metadata"] = json.loads(hit.pop("source_metadata_json") or "{}")
            hits.append(hit)

        retrieval_response = {
            "result_id": retrieval["id"],
            "requested_top_k": retrieval["requested_top_k"],
            "returned_count": retrieval["returned_count"],
            "distance_metric": retrieval["distance_metric"],
            "duration_ms": retrieval["duration_ms"],
            "chunks": hits,
        }

    generation_response = None

    # Materialize answer provenance independently from retrieval availability.
    if generation is not None:
        context_rows = connection.execute(
            """
            SELECT benchmark_generation_context_chunk.ordinal,
                   benchmark_generation_context_chunk.retrieval_rank,
                   benchmark_retrieved_chunk.chunk_id
            FROM benchmark_generation_context_chunk
            JOIN benchmark_retrieved_chunk
              ON benchmark_retrieved_chunk.retrieval_result_id =
                    benchmark_generation_context_chunk.retrieval_result_id
             AND benchmark_retrieved_chunk.rank =
                    benchmark_generation_context_chunk.retrieval_rank
            WHERE benchmark_generation_context_chunk.generation_result_id = ?
            ORDER BY benchmark_generation_context_chunk.ordinal
            """,
            (generation["id"],),
        ).fetchall()
        generation_response = {
            "result_id": generation["id"],
            "retrieval_result_id": generation["retrieval_result_id"],
            "provider": generation["provider"],
            "model": generation["model"],
            "provider_model": generation["provider_model"],
            "answer": generation["answer_text"],
            "finish_reason": generation["finish_reason"],
            "prompt_template_version": generation["prompt_template_version"],
            "provider_policy_version": generation["provider_policy_version"],
            "prompt_tokens": generation["prompt_tokens"],
            "completion_tokens": generation["completion_tokens"],
            "total_tokens": generation["total_tokens"],
            "provider_called": bool(generation["provider_called"]),
            "duration_ms": generation["duration_ms"],
            "context_chunks": [dict(row) for row in context_rows],
        }

    return {
        "id": example["id"],
        "example_id": example["evaluation_example_id"],
        "ordinal": example["ordinal"],
        "question": example["question"],
        "reference_answer": example["reference_answer"],
        "status": example["status"],
        "current_stage": example["current_stage"],
        "started_at": example["started_at"],
        "completed_at": example["completed_at"],
        "duration_ms": example["duration_ms"],
        "retrieval": retrieval_response,
        "generation": generation_response,
        "error": _materialize_error(example),
    }
