"""Durable, independently repeatable scoring of saved benchmark retrieval."""

import json
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

from backend.db.connection import connect
from backend.evaluation.retrieval import METRICS, score_question


class EvaluationNotFoundError(LookupError):
    """Report an unknown benchmark run or retrieval-evaluation attempt."""


class EvaluationRunStateError(ValueError):
    """Report a benchmark whose lifecycle state does not permit evaluation."""


def _now() -> str:
    """Create a timestamp for evaluation lifecycle writes.

    Args:
        None.

    Returns:
        An ISO-8601 UTC timestamp ending in ``Z``.
    """
    # Store second-level UTC timestamps consistently with the other durable queues.
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _row_dict(
    row: sqlite3.Row,
    include_questions: bool = False,
) -> dict[str, object]:
    """Decode one stored attempt into its repository response shape.

    Args:
        row: SQLite row containing one retrieval-evaluation attempt.
        include_questions: Whether to initialize the detail-only question collection.

    Returns:
        Attempt data with decoded configuration, aggregates, and structured error.
    """
    # Convert the row before replacing persisted JSON columns with decoded values.
    result = dict(row)
    result["configuration"] = json.loads(result.pop("config_json"))
    result["aggregates"] = json.loads(result.pop("aggregate_json") or "{}")
    # Expose the two nullable error columns as one structured API-facing value.
    if result["error_code"]:
        result["error"] = {
            "code": result.pop("error_code"),
            "message": result.pop("error_message"),
        }
    else:
        result.pop("error_code")
        result.pop("error_message")
        result["error"] = None
    # Question data is populated by the detail reader, not the summary reader.
    if include_questions:
        result["questions"] = []
    return result


def create_evaluation(run_id: str, metrics: list[str] | None = None) -> dict:
    """Queue one scoring attempt using explicit or saved metric selections.

    Args:
        run_id: Stable identifier of the completed benchmark to score.
        metrics: Optional retrieval metric override; ``None`` uses the run snapshot.

    Returns:
        The newly persisted pending evaluation with an empty question collection.

    Raises:
        EvaluationNotFoundError: If the benchmark run does not exist.
        EvaluationRunStateError: If the benchmark has not completed successfully.
        ValueError: If a metric is unsupported or appears more than once.
    """
    # Validate the parent and save the attempt atomically on one connection.
    with connect() as connection:
        # Only immutable completed benchmark output can be scored.
        run = connection.execute(
            "SELECT status, effective_config_json FROM benchmark_run WHERE id = ?",
            (run_id,),
        ).fetchone()
        if run is None:
            raise EvaluationNotFoundError(run_id)

        # Failed, pending, and running benchmarks do not have immutable full results.
        if run["status"] != "completed":
            raise EvaluationRunStateError(
                "Only completed benchmark runs can be evaluated."
            )

        saved = json.loads(run["effective_config_json"])["evaluation"]
        selected = saved["retrieval_metrics"] if metrics is None else metrics

        # A set removes duplicates; unequal lengths therefore indicate repetition.
        if len(selected) != len(set(selected)) or any(
            metric not in METRICS for metric in selected
        ):
            raise ValueError("Unknown or duplicate retrieval metric selection.")
        attempt_id = str(uuid4())

        # Preserve answer selections for provenance while this release scores retrieval.
        snapshot = {
            "retrieval_metrics": selected,
            "answer_metrics": saved["answer_metrics"],
        }
        connection.execute(
            """INSERT INTO retrieval_evaluation
               (id, benchmark_run_id, status, config_json, created_at)
               VALUES (?, ?, 'pending', ?, ?)""",
            (attempt_id, run_id, json.dumps(snapshot), _now()),
        )
    return get_evaluation(run_id, attempt_id)


def enqueue_saved_evaluation(connection, run_id: str, config_json: str) -> None:
    """Queue the saved retrieval selection during benchmark completion.

    Args:
        connection: Active benchmark-completion transaction.
        run_id: Stable identifier of the benchmark that just completed.
        config_json: Immutable effective pipeline configuration stored on the run.

    Returns:
        None. A pending attempt is inserted only when retrieval metrics are selected.
    """
    # Read evaluation selections from the exact configuration used by the benchmark.
    selected = json.loads(config_json)["evaluation"]
    # Empty retrieval selection intentionally has no automatic scoring attempt.
    if selected["retrieval_metrics"]:
        connection.execute(
            """INSERT INTO retrieval_evaluation
               (id, benchmark_run_id, status, config_json, created_at)
               VALUES (?, ?, 'pending', ?, ?)""",
            (str(uuid4()), run_id, json.dumps(selected), _now()),
        )


def list_evaluations(run_id: str) -> list[dict]:
    """Return newest evaluation attempts for a known benchmark.

    Args:
        run_id: Stable benchmark identifier whose attempt history is requested.

    Returns:
        Compact evaluation attempts ordered newest first.

    Raises:
        EvaluationNotFoundError: If the benchmark run does not exist.
    """
    # Verify the parent separately so an empty history differs from an unknown run.
    with connect() as connection:
        if (
            connection.execute(
                "SELECT 1 FROM benchmark_run WHERE id = ?", (run_id,)
            ).fetchone()
            is None
        ):
            raise EvaluationNotFoundError(run_id)
        rows = connection.execute(
            """SELECT * FROM retrieval_evaluation WHERE benchmark_run_id = ?
               ORDER BY created_at DESC, rowid DESC""",
            (run_id,),
        ).fetchall()
    # Decode every persisted JSON snapshot before crossing the repository boundary.
    return [_row_dict(row) for row in rows]


def get_evaluation(run_id: str, attempt_id: str) -> dict:
    """Return one attempt with saved per-question scores and evidence.

    Args:
        run_id: Stable identifier of the attempt's parent benchmark.
        attempt_id: Stable identifier of the requested evaluation attempt.

    Returns:
        Evaluation detail containing ordered question outcomes.

    Raises:
        EvaluationNotFoundError: If the attempt does not belong to the benchmark.
    """
    # Load the attempt and its normalized question rows from one database snapshot.
    with connect() as connection:
        row = connection.execute(
            "SELECT * FROM retrieval_evaluation WHERE id = ? AND benchmark_run_id = ?",
            (attempt_id, run_id),
        ).fetchone()
        if row is None:
            raise EvaluationNotFoundError(attempt_id)

        # Preserve dataset order so evidence aligns with benchmark question order.
        questions = connection.execute(
            """SELECT example_id, ordinal, result_json FROM retrieval_evaluation_question
               WHERE evaluation_id = ? ORDER BY ordinal""",
            (attempt_id,),
        ).fetchall()
    result = _row_dict(row, include_questions=True)

    # Merge each question's identity columns with its JSON scoring evidence.
    result["questions"] = [
        {
            "example_id": row["example_id"],
            "ordinal": row["ordinal"],
            **json.loads(row["result_json"]),
        }
        for row in questions
    ]
    return result


def latest_evaluation(connection, run_id: str) -> dict | None:
    """Read the latest compact attempt within a caller-owned transaction.

    Args:
        connection: Active SQLite connection used by a run summary query.
        run_id: Stable benchmark identifier whose latest attempt is requested.

    Returns:
        Decoded latest attempt, or ``None`` when the run has no evaluations.
    """
    # Break timestamp ties with insertion order because timestamps use seconds.
    row = connection.execute(
        """SELECT * FROM retrieval_evaluation WHERE benchmark_run_id = ?
           ORDER BY created_at DESC, rowid DESC LIMIT 1""",
        (run_id,),
    ).fetchone()
    return _row_dict(row) if row else None


def execute_evaluation(attempt_id: str) -> dict:
    """Score saved rankings and persist all question outcomes atomically.

    Args:
        attempt_id: Stable identifier of an evaluation already claimed as running.

    Returns:
        Completed evaluation including aggregates and per-question evidence.

    Raises:
        EvaluationNotFoundError: If the attempt is missing or is not running.
    """
    # Keep all question results and the terminal transition in one transaction.
    with connect() as connection:
        attempt = connection.execute(
            "SELECT * FROM retrieval_evaluation WHERE id = ? AND status = 'running'",
            (attempt_id,),
        ).fetchone()
        if attempt is None:
            raise EvaluationNotFoundError(attempt_id)

        # The attempt snapshot determines metrics; the run snapshot determines K.
        config = json.loads(attempt["config_json"])

        run = connection.execute(
            "SELECT effective_config_json FROM benchmark_run WHERE id = ?",
            (attempt["benchmark_run_id"],),
        ).fetchone()
        top_k = json.loads(run["effective_config_json"])["retrieval"]["top_k"]
        
        examples = connection.execute(
            """SELECT evaluation_example_id, ordinal FROM benchmark_example_run
               WHERE benchmark_run_id = ? ORDER BY ordinal""",
            (attempt["benchmark_run_id"],),
        ).fetchall()
        totals = {metric: 0.0 for metric in config["retrieval_metrics"]}
        eligible = 0

        # Each question is scored solely from persisted labels and saved chunk ranks.
        for example in examples:
            # Resolve distinct ground-truth documents saved during dataset import.
            labels = {
                row["document_id"]
                for row in connection.execute(
                    """SELECT document_id FROM evaluation_example_relevant_document
                   WHERE example_id = ?""",
                    (example["evaluation_example_id"],),
                )
            }
            # Retain chunk-level ranking, including repeated source documents.
            hits = [
                row["source_document_id"]
                for row in connection.execute(
                    """SELECT chunk.source_document_id FROM benchmark_example_run
                   JOIN benchmark_retrieval_result AS result
                     ON result.example_run_id = benchmark_example_run.id
                   JOIN benchmark_retrieved_chunk AS hit
                     ON hit.retrieval_result_id = result.id
                   JOIN chunk ON chunk.id = hit.chunk_id
                   WHERE benchmark_example_run.benchmark_run_id = ?
                     AND benchmark_example_run.evaluation_example_id = ?
                   ORDER BY hit.rank""",
                    (attempt["benchmark_run_id"], example["evaluation_example_id"]),
                )
            ]
            outcome = score_question(labels, hits, top_k)
            outcome["ranked_document_ids"] = hits[:top_k]

            # Persist only the metrics selected for this attempt.
            if outcome["scores"] is not None:
                eligible += 1

                # filter the required metrics from the outcome
                outcome["scores"] = {
                    metric: outcome["scores"][metric]
                    for metric in config["retrieval_metrics"]
                }
                # Accumulate eligible question scores for the final macro average.
                for metric, value in outcome["scores"].items():
                    totals[metric] += value

            # Save scored and skipped questions alike so coverage is auditable.
            connection.execute(
                """INSERT INTO retrieval_evaluation_question
                   (evaluation_id, example_id, ordinal, result_json) VALUES (?, ?, ?, ?)""",
                (
                    attempt_id,
                    example["evaluation_example_id"],
                    example["ordinal"],
                    json.dumps(outcome),
                ),
            )
        # Avoid division by zero by returning null metrics with zero eligible questions.
        aggregates = {
            metric: totals[metric] / eligible if eligible else None for metric in totals
        }
        connection.execute(
            """UPDATE retrieval_evaluation SET status = 'completed',
               aggregate_json = ?, eligible_count = ?, total_count = ?, completed_at = ?
               WHERE id = ?""",
            (json.dumps(aggregates), eligible, len(examples), _now(), attempt_id),
        )
    return get_evaluation(attempt["benchmark_run_id"], attempt_id)


def fail_evaluation(attempt_id: str) -> None:
    """Record a safe evaluation failure without changing its benchmark.

    Args:
        attempt_id: Stable identifier of the running attempt that failed.

    Returns:
        None. The running attempt is terminalized when it still exists.
    """
    # Update only a running attempt so terminal history cannot be overwritten.
    with connect() as connection:
        connection.execute(
            """UPDATE retrieval_evaluation SET status = 'failed',
               error_code = 'evaluation_failed',
               error_message = 'Saved retrieval results could not be scored.',
               completed_at = ? WHERE id = ? AND status = 'running'""",
            (_now(), attempt_id),
        )


def fail_interrupted_evaluations() -> int:
    """Terminalize attempts abandoned during a backend restart.

    Args:
        None.

    Returns:
        Number of running attempts changed to failed.
    """
    # A new process cannot resume work held by the previous process's thread.
    with connect() as connection:
        cursor = connection.execute(
            """UPDATE retrieval_evaluation SET status = 'failed',
               error_code = 'evaluation_interrupted',
               error_message = 'The backend stopped during evaluation.', completed_at = ?
               WHERE status = 'running'""",
            (_now(),),
        )
    return cursor.rowcount
