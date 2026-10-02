"""Durable, independently repeatable scoring of saved benchmark outputs."""

import json
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

from backend.db.connection import connect
from backend.evaluation.answer import (
    ANSWER_METRICS,
    AnswerJudge,
    AnswerJudgeError,
    AnswerJudgeInput,
    GroqAnswerJudge,
    evaluator_snapshot,
)
from backend.evaluation.retrieval import METRICS, score_question


class EvaluationNotFoundError(LookupError):
    """Report an unknown benchmark run or evaluation attempt."""


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
    result["coverage"] = json.loads(result.pop("coverage_json") or "{}")
    result["has_errors"] = bool(result["has_errors"])
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


def create_evaluation(
    run_id: str,
    metrics: list[str] | None = None,
    answer_metrics: list[str] | None = None,
) -> dict:
    """Queue one scoring attempt using explicit or saved metric selections.

    Args:
        run_id: Stable identifier of the completed benchmark to score.
        metrics: Optional retrieval metric override; ``None`` uses the run snapshot.
        answer_metrics: Optional answer metric override; ``None`` uses the run snapshot.

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
        selected_answers = (
            saved["answer_metrics"] if answer_metrics is None else answer_metrics
        )

        # A set removes duplicates; unequal lengths therefore indicate repetition.
        if len(selected) != len(set(selected)) or any(
            metric not in METRICS for metric in selected
        ):
            raise ValueError("Unknown or duplicate retrieval metric selection.")
        # Answer selections use the same duplicate and supported-value guarantees.
        if len(selected_answers) != len(set(selected_answers)) or any(
            metric not in ANSWER_METRICS for metric in selected_answers
        ):
            raise ValueError("Unknown or duplicate answer metric selection.")
        if not selected and not selected_answers:
            raise ValueError("At least one evaluation metric must be selected.")
        attempt_id = str(uuid4())

        # Preserve answer selections for provenance while this release scores retrieval.
        snapshot = {
            "retrieval_metrics": selected,
            "answer_metrics": selected_answers,
            "evaluator": evaluator_snapshot() if selected_answers else None,
        }
        connection.execute(
            """INSERT INTO retrieval_evaluation
               (id, benchmark_run_id, status, config_json, created_at)
               VALUES (?, ?, 'pending', ?, ?)""",
            (attempt_id, run_id, json.dumps(snapshot), _now()),
        )
    return get_evaluation(run_id, attempt_id)


def enqueue_saved_evaluation(connection, run_id: str, config_json: str) -> None:
    """Queue saved metric selections during benchmark completion.

    Args:
        connection: Active benchmark-completion transaction.
        run_id: Stable identifier of the benchmark that just completed.
        config_json: Immutable effective pipeline configuration stored on the run.

    Returns:
        None. A pending attempt is inserted when any evaluation metric is selected.
    """
    # Read evaluation selections from the exact configuration used by the benchmark.
    selected = json.loads(config_json)["evaluation"]
    # Queue when either deterministic retrieval or judged answer metrics are selected.
    if selected["retrieval_metrics"] or selected["answer_metrics"]:
        snapshot = dict(selected)
        snapshot["evaluator"] = (
            evaluator_snapshot() if selected["answer_metrics"] else None
        )
        connection.execute(
            """INSERT INTO retrieval_evaluation
               (id, benchmark_run_id, status, config_json, created_at)
               VALUES (?, ?, 'pending', ?, ?)""",
            (str(uuid4()), run_id, json.dumps(snapshot), _now()),
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


def _load_question_inputs(run_id: str) -> list[dict[str, object]]:
    """Load immutable retrieval and generation inputs for an evaluation.

    Args:
        run_id: Completed benchmark whose saved outputs are being evaluated.

    Returns:
        Ordered question inputs including labels, hits, answer, and exact context.
    """
    with connect() as connection:
        rows = connection.execute(
            """SELECT child.evaluation_example_id, child.ordinal,
                      example.question, example.reference_answer,
                      generation.id AS generation_result_id,
                      generation.answer_text
               FROM benchmark_example_run AS child
               JOIN evaluation_example AS example
                 ON example.id = child.evaluation_example_id
               LEFT JOIN benchmark_generation_result AS generation
                 ON generation.example_run_id = child.id
               WHERE child.benchmark_run_id = ? ORDER BY child.ordinal""",
            (run_id,),
        ).fetchall()
        questions: list[dict[str, object]] = []

        # Hydrate evidence in dataset order while preserving exact persisted ranks.
        for row in rows:
            labels = {
                label["document_id"]
                for label in connection.execute(
                    """SELECT document_id FROM evaluation_example_relevant_document
                       WHERE example_id = ?""",
                    (row["evaluation_example_id"],),
                )
            }
            hits = [
                hit["source_document_id"]
                for hit in connection.execute(
                    """SELECT chunk.source_document_id
                       FROM benchmark_example_run AS child
                       JOIN benchmark_retrieval_result AS result
                         ON result.example_run_id = child.id
                       JOIN benchmark_retrieved_chunk AS hit
                         ON hit.retrieval_result_id = result.id
                       JOIN chunk ON chunk.id = hit.chunk_id
                       WHERE child.benchmark_run_id = ?
                         AND child.evaluation_example_id = ?
                       ORDER BY hit.rank""",
                    (run_id, row["evaluation_example_id"]),
                )
            ]
            context = []

            # Context is empty only when no generation result was persisted.
            if row["generation_result_id"] is not None:
                context = [
                    item["text"]
                    for item in connection.execute(
                        """SELECT chunk.text
                           FROM benchmark_generation_context_chunk AS context
                           JOIN benchmark_retrieved_chunk AS hit
                             ON hit.retrieval_result_id = context.retrieval_result_id
                            AND hit.rank = context.retrieval_rank
                           JOIN chunk ON chunk.id = hit.chunk_id
                           WHERE context.generation_result_id = ?
                           ORDER BY context.ordinal""",
                        (row["generation_result_id"],),
                    )
                ]

            questions.append(
                {
                    "example_id": row["evaluation_example_id"],
                    "ordinal": row["ordinal"],
                    "question": row["question"],
                    "reference_answer": row["reference_answer"],
                    "answer": row["answer_text"],
                    "labels": labels,
                    "hits": hits,
                    "context": context,
                }
            )
    return questions


def _persist_question(
    attempt_id: str,
    example_id: str,
    ordinal: int,
    outcome: dict[str, object],
) -> None:
    """Persist one completed question independently from provider work.

    Args:
        attempt_id: Parent evaluation attempt identifier.
        example_id: Immutable dataset example identifier.
        ordinal: Dataset display order.
        outcome: Retrieval and answer evaluation evidence to serialize.

    Returns:
        None. The question result is committed before the next provider call.
    """
    # Upsert permits restart-safe execution without replacing already saved work.
    with connect() as connection:
        connection.execute(
            """INSERT OR IGNORE INTO retrieval_evaluation_question
               (evaluation_id, example_id, ordinal, result_json) VALUES (?, ?, ?, ?)""",
            (attempt_id, example_id, ordinal, json.dumps(outcome)),
        )


def execute_evaluation(
    attempt_id: str,
    answer_judge: AnswerJudge | None = None,
) -> dict:
    """Score saved retrieval and generated answers without rerunning the pipeline.

    Args:
        attempt_id: Stable identifier of an evaluation already claimed as running.
        answer_judge: Optional deterministic or production judge implementation.

    Returns:
        Completed evaluation including aggregates and per-question evidence.

    Raises:
        EvaluationNotFoundError: If the attempt is missing or is not running.
    """
    # Read attempt metadata in a short transaction before any provider request.
    with connect() as connection:
        attempt = connection.execute(
            "SELECT * FROM retrieval_evaluation WHERE id = ? AND status = 'running'",
            (attempt_id,),
        ).fetchone()
        if attempt is None:
            raise EvaluationNotFoundError(attempt_id)

        config = json.loads(attempt["config_json"])
        run = connection.execute(
            "SELECT effective_config_json FROM benchmark_run WHERE id = ?",
            (attempt["benchmark_run_id"],),
        ).fetchone()
        top_k = json.loads(run["effective_config_json"])["retrieval"]["top_k"]

    questions = _load_question_inputs(attempt["benchmark_run_id"])
    selected_retrieval = config["retrieval_metrics"]
    selected_answers = config["answer_metrics"]
    all_metrics = [*selected_retrieval, *selected_answers]
    totals = {metric: 0.0 for metric in all_metrics}
    coverage = {
        metric: {
            "total": len(questions),
            "eligible": 0,
            "scored": 0,
            "skipped": 0,
            "error": 0,
        }
        for metric in all_metrics
    }
    judge = answer_judge or (GroqAnswerJudge() if selected_answers else None)
    has_errors = False

    # Score each question and commit it before beginning another paid request.
    for question in questions:
        outcome = score_question(question["labels"], question["hits"], top_k)
        outcome["ranked_document_ids"] = question["hits"][:top_k]
        outcome["answer_scores"] = {}
        outcome["answer_skips"] = {}
        outcome["answer_error"] = None
        outcome["judge"] = None

        # Filter deterministic retrieval scores and update per-metric coverage.
        if outcome["scores"] is None:
            for metric in selected_retrieval:
                coverage[metric]["skipped"] += 1
        else:
            outcome["scores"] = {
                metric: outcome["scores"][metric] for metric in selected_retrieval
            }
            for metric, value in outcome["scores"].items():
                coverage[metric]["eligible"] += 1
                coverage[metric]["scored"] += 1
                totals[metric] += value

        eligible_answers = []

        # A persisted generated answer is required by every answer metric.
        for metric in selected_answers:
            skip_reason = None
            if question["answer"] is None:
                skip_reason = "no_generated_answer"
            elif metric == "answer_correctness" and not question["reference_answer"]:
                skip_reason = "no_reference_answer"

            if skip_reason:
                outcome["answer_skips"][metric] = skip_reason
                coverage[metric]["skipped"] += 1
            else:
                eligible_answers.append(metric)
                coverage[metric]["eligible"] += 1

        # One call evaluates all eligible metrics for this question.
        if eligible_answers:
            try:
                judged = judge.judge(
                    AnswerJudgeInput(
                        question=question["question"],
                        answer=question["answer"],
                        context_chunks=tuple(question["context"]),
                        reference_answer=question["reference_answer"],
                        metrics=tuple(eligible_answers),
                    )
                )
                outcome["answer_scores"] = judged.results
                outcome["judge"] = {
                    "duration_ms": judged.duration_ms,
                    "prompt_tokens": judged.prompt_tokens,
                    "completion_tokens": judged.completion_tokens,
                    "total_tokens": judged.total_tokens,
                    "provider_request_id": judged.provider_request_id,
                    "provider_model": judged.provider_model,
                }
                for metric, result in judged.results.items():
                    coverage[metric]["scored"] += 1
                    totals[metric] += result["score"]
            except AnswerJudgeError as error:
                # One provider failure applies to all metrics in this combined request.
                has_errors = True
                outcome["answer_error"] = {
                    "code": error.code,
                    "message": error.message,
                    "metrics": eligible_answers,
                }
                for metric in eligible_answers:
                    coverage[metric]["error"] += 1

        _persist_question(
            attempt_id,
            question["example_id"],
            question["ordinal"],
            outcome,
        )

    # Macro averages include successful scores only; coverage makes exclusions clear.
    aggregates = {
        metric: totals[metric] / coverage[metric]["scored"]
        if coverage[metric]["scored"]
        else None
        for metric in totals
    }
    total_eligible = sum(item["eligible"] for item in coverage.values())
    total_scored = sum(item["scored"] for item in coverage.values())
    terminal_status = "failed" if total_eligible and not total_scored else "completed"
    terminal_error = (
        (
            "evaluation_failed",
            "No eligible evaluation metric could be scored.",
        )
        if terminal_status == "failed"
        else (None, None)
    )

    # Terminalize in a final short transaction without modifying benchmark results.
    with connect() as connection:
        connection.execute(
            """UPDATE retrieval_evaluation SET status = ?, aggregate_json = ?,
               coverage_json = ?, has_errors = ?, eligible_count = ?, total_count = ?,
               error_code = ?, error_message = ?, completed_at = ?
               WHERE id = ?""",
            (
                terminal_status,
                json.dumps(aggregates),
                json.dumps(coverage),
                int(has_errors),
                max((item["eligible"] for item in coverage.values()), default=0),
                len(questions),
                terminal_error[0],
                terminal_error[1],
                _now(),
                attempt_id,
            ),
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
               error_message = 'Saved benchmark outputs could not be scored.',
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
