"""SQLite boundaries for immutable generated answers and prompt context links."""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid5

from backend.db.connection import connect
from backend.generation.models import GenerationServiceResult
from backend.pipeline.configs import GenerationConfig

# Deterministic IDs keep the one allowed answer for a run stable across retries.
_GENERATION_RESULT_NAMESPACE = UUID("a84d2414-7aef-4b36-82cf-f19b9f95f4a8")


class InvalidGenerationResultError(ValueError):
    """Report malformed answer, usage, timing, or context provenance."""


class GenerationArtifactMismatchError(RuntimeError):
    """Report a generation result linked to another run or retrieval result."""


def insert_generation_result(
    connection: sqlite3.Connection,
    pipeline_run_id: str,
    retrieval_result_id: str,
    generation_config: GenerationConfig,
    result: GenerationServiceResult,
    duration_ms: int,
) -> str:
    """Insert one validated answer using an existing lifecycle transaction.

    Args:
        connection: Active SQLite transaction owning the complete run transition.
        pipeline_run_id: Stable run that owns the generated answer.
        retrieval_result_id: Exact persisted retrieval result used as context.
        generation_config: Immutable effective generation settings for the run.
        result: Generated answer and exact context/provider provenance.
        duration_ms: Non-negative generation-stage wall-clock duration.

    Returns:
        Deterministic generation-result identifier.

    Raises:
        InvalidGenerationResultError: If values or context ordering are malformed.
        GenerationArtifactMismatchError: If upstream ownership is incompatible.
        sqlite3.IntegrityError: If relational persistence rejects any write.
    """
    _validate_result_values(result, duration_ms)
    retrieval_rows = _validate_artifact_ownership(
        connection,
        pipeline_run_id,
        retrieval_result_id,
        result.context_chunk_ids,
    )
    result_id = str(
        uuid5(
            _GENERATION_RESULT_NAMESPACE,
            f"generation-result:{pipeline_run_id}",
        )
    )
    response = result.response
    created_at = _utc_timestamp()
    connection.execute(
        """
        INSERT INTO generation_result (
            id, pipeline_run_id, retrieval_result_id, provider, model,
            provider_model, prompt_template_version, provider_policy_version,
            generation_config_json, answer_text, finish_reason,
            prompt_tokens, completion_tokens, total_tokens,
            provider_request_id, system_fingerprint, provider_called,
            duration_ms, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result_id,
            pipeline_run_id,
            retrieval_result_id,
            generation_config.provider,
            generation_config.model,
            response.provider_model,
            result.prompt_template_version,
            result.provider_policy_version,
            _canonical_json(generation_config.model_dump(mode="json")),
            response.answer_text,
            response.finish_reason,
            response.prompt_tokens,
            response.completion_tokens,
            response.total_tokens,
            response.provider_request_id,
            response.system_fingerprint,
            int(result.provider_called),
            duration_ms,
            created_at,
        ),
    )

    # Persist the exact prefix of ranked retrieval hits included in the prompt.
    for ordinal, retrieval_row in enumerate(retrieval_rows, start=1):
        connection.execute(
            """
            INSERT INTO generation_context_chunk (
                generation_result_id, ordinal,
                retrieval_result_id, retrieval_rank
            ) VALUES (?, ?, ?, ?)
            """,
            (
                result_id,
                ordinal,
                retrieval_result_id,
                retrieval_row["rank"],
            ),
        )

    return result_id


def get_generation_result_for_run(pipeline_run_id: str) -> dict[str, Any] | None:
    """Load one run's generated answer and exact context references.

    Args:
        pipeline_run_id: Stable run identifier whose answer should be loaded.

    Returns:
        Materialized generation result, or ``None`` before generation succeeds.
    """
    # Read the answer and ordered context links from one consistent connection.
    with connect() as connection:
        row = connection.execute(
            "SELECT * FROM generation_result WHERE pipeline_run_id = ?",
            (pipeline_run_id,),
        ).fetchone()

        # Missing generation is a normal pending, active, or failed-stage state.
        if row is None:
            return None

        context_rows = connection.execute(
            """
            SELECT generation_context_chunk.ordinal,
                   generation_context_chunk.retrieval_rank,
                   retrieved_chunk.chunk_id
            FROM generation_context_chunk
            JOIN retrieved_chunk
              ON retrieved_chunk.retrieval_result_id =
                    generation_context_chunk.retrieval_result_id
             AND retrieved_chunk.rank =
                    generation_context_chunk.retrieval_rank
            WHERE generation_context_chunk.generation_result_id = ?
            ORDER BY generation_context_chunk.ordinal
            """,
            (row["id"],),
        ).fetchall()

    return {
        "id": row["id"],
        "retrieval_result_id": row["retrieval_result_id"],
        "provider": row["provider"],
        "model": row["model"],
        "provider_model": row["provider_model"],
        "prompt_template_version": row["prompt_template_version"],
        "provider_policy_version": row["provider_policy_version"],
        "answer": row["answer_text"],
        "finish_reason": row["finish_reason"],
        "prompt_tokens": row["prompt_tokens"],
        "completion_tokens": row["completion_tokens"],
        "total_tokens": row["total_tokens"],
        "provider_request_id": row["provider_request_id"],
        "system_fingerprint": row["system_fingerprint"],
        "provider_called": bool(row["provider_called"]),
        "duration_ms": row["duration_ms"],
        "created_at": row["created_at"],
        "context_chunks": [dict(context_row) for context_row in context_rows],
    }


def _validate_result_values(
    result: GenerationServiceResult,
    duration_ms: int,
) -> None:
    """Validate answer-level values before any generation rows are written.

    Args:
        result: Generated answer and provenance proposed for persistence.
        duration_ms: Generation-stage wall-clock duration.

    Returns:
        None. Valid values remain unchanged.

    Raises:
        InvalidGenerationResultError: If any persisted value is malformed.
    """
    response = result.response

    # Stage duration must be a concrete non-negative integer measurement.
    if (
        isinstance(duration_ms, bool)
        or not isinstance(duration_ms, int)
        or duration_ms < 0
    ):
        raise InvalidGenerationResultError("duration_ms must be non-negative.")

    # Required text fields must remain meaningful after whitespace normalization.
    required_text = (
        response.answer_text,
        response.finish_reason,
        result.prompt_template_version,
        result.provider_policy_version,
    )
    if any(not isinstance(value, str) or not value.strip() for value in required_text):
        raise InvalidGenerationResultError(
            "Generation answer and policy values must not be blank."
        )

    # Exact context links cannot contain duplicate chunk identifiers.
    if len(result.context_chunk_ids) != len(set(result.context_chunk_ids)):
        raise InvalidGenerationResultError(
            "Generation context chunk identifiers must be unique."
        )

    # Every optional usage field must be a non-negative integer when present.
    for value in (
        response.prompt_tokens,
        response.completion_tokens,
        response.total_tokens,
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise InvalidGenerationResultError(
                "Generation token usage must be non-negative."
            )


def _validate_artifact_ownership(
    connection: sqlite3.Connection,
    pipeline_run_id: str,
    retrieval_result_id: str,
    context_chunk_ids: tuple[str, ...],
) -> list[sqlite3.Row]:
    """Validate retrieval ownership and return its exact selected rank prefix.

    Args:
        connection: Active transaction used for ownership reads and later writes.
        pipeline_run_id: Run expected to own the retrieval result.
        retrieval_result_id: Retrieval result expected to supply generation context.
        context_chunk_ids: Ordered chunk IDs included by the prompt builder.

    Returns:
        Ordered retrieval rows corresponding exactly to the context identifiers.

    Raises:
        GenerationArtifactMismatchError: If run/retrieval ownership is incompatible.
        InvalidGenerationResultError: If context is not an exact retrieval-rank prefix.
    """
    retrieval = connection.execute(
        """
        SELECT pipeline_run_id FROM retrieval_result WHERE id = ?
        """,
        (retrieval_result_id,),
    ).fetchone()

    # Generation may use only the retrieval result persisted for the same run.
    if retrieval is None or retrieval["pipeline_run_id"] != pipeline_run_id:
        raise GenerationArtifactMismatchError(
            "The retrieval result does not belong to the pipeline run."
        )

    retrieval_rows = connection.execute(
        """
        SELECT rank, chunk_id FROM retrieved_chunk
        WHERE retrieval_result_id = ?
        ORDER BY rank
        """,
        (retrieval_result_id,),
    ).fetchall()
    selected_rows = retrieval_rows[: len(context_chunk_ids)]
    selected_ids = tuple(row["chunk_id"] for row in selected_rows)

    # Prompt packing must preserve a contiguous prefix of nearest-neighbor ranking.
    if selected_ids != context_chunk_ids:
        raise InvalidGenerationResultError(
            "Generation context must preserve retrieval-rank order."
        )

    return list(selected_rows)


def _utc_timestamp() -> str:
    """Create the UTC timestamp format used by generation persistence.

    Args:
        None.

    Returns:
        ISO-8601 UTC timestamp ending in ``Z``.
    """
    # Timezone-aware values preserve unambiguous provider-stage history.
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _canonical_json(value: Any) -> str:
    """Serialize generation configuration deterministically.

    Args:
        value: JSON-compatible configuration value to persist.

    Returns:
        Compact JSON with stable key ordering.
    """
    # Canonical settings make historical generation provenance easy to compare.
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
