"""Deterministic offline coverage for saved retrieval scoring and attempt lifecycle."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from test_benchmark_runs import _experiment_configuration, _seed_ready_inputs

from backend.api.routers.runs import EvaluationRequest, create_run_evaluation
from backend.db.connection import connect
from backend.db.repositories.benchmark_runs import (
    create_pending_benchmark_run,
    resolve_benchmark_configuration,
)
from backend.db.repositories.retrieval_evaluations import (
    create_evaluation,
    execute_evaluation,
    fail_interrupted_evaluations,
    get_evaluation,
)
from backend.db.repositories.work_queue import claim_next_pending_work_item
from backend.evaluation.answer import AnswerJudgeInput, AnswerJudgeResult
from backend.evaluation.retrieval import score_question
from backend.generation.models import (
    GenerationProviderResponse,
    GenerationServiceResult,
)
from backend.pipeline.benchmark_execution import BenchmarkExecutor
from backend.pipeline.configs import ExperimentConfig


class FakeAnswerJudge:
    """Return deterministic answer judgments while recording exact inputs."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        """Configure an optional deterministic provider failure.

        Args:
            fail_on_call: One-based call number that should fail when provided.

        Returns:
            None. Calls are retained for assertions.
        """
        # Mutable call history belongs only to this isolated test fake.
        self.calls: list[AnswerJudgeInput] = []
        self.fail_on_call = fail_on_call

    def judge(self, judge_input: AnswerJudgeInput) -> AnswerJudgeResult:
        """Score all requested metrics in one deterministic operation.

        Args:
            judge_input: Exact persisted material supplied by the executor.

        Returns:
            Fixed valid scores and provenance for every requested metric.
        """
        self.calls.append(judge_input)

        # The optional failure supports partial-result coverage without network calls.
        if self.fail_on_call == len(self.calls):
            from backend.evaluation.answer import AnswerJudgeError

            raise AnswerJudgeError("evaluator_unavailable", "Evaluator unavailable.")

        return AnswerJudgeResult(
            results={
                metric: {
                    "rubric_score": 3,
                    "score": 0.75,
                    "rationale": "Deterministic test rationale.",
                    "evidence_ranks": [],
                }
                for metric in judge_input.metrics
            },
            duration_ms=5,
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
            provider_request_id=f"judge-{len(self.calls)}",
            provider_model="openai/gpt-oss-20b",
        )


def _complete_generated_benchmark(
    answer_metrics: list[str] | None = None,
) -> str:
    """Create and execute a benchmark with deterministic generated answers.

    Returns:
        Stable identifier of the completed benchmark.
    """
    _seed_ready_inputs()
    experiment_data = _experiment_configuration().model_dump()

    # Override saved selections only when a test is exercising automatic judging.
    if answer_metrics is not None:
        experiment_data["evaluation"]["retrieval_metrics"] = []
        experiment_data["evaluation"]["answer_metrics"] = answer_metrics

    _, _, config = resolve_benchmark_configuration(
        "prepared-index-1",
        "dataset-1",
        ExperimentConfig.model_validate(experiment_data),
    )
    run = create_pending_benchmark_run("prepared-index-1", "dataset-1", config)
    claimed = claim_next_pending_work_item()
    assert claimed == {"kind": "benchmark_run", "id": run["id"]}

    def retrieve(*args: Any, **kwargs: Any) -> tuple[()]:
        """Return empty saved retrieval for the deterministic answer fixture."""
        return ()

    def generate(*args: Any, **kwargs: Any) -> GenerationServiceResult:
        """Return one offline generated answer without provider access."""
        return GenerationServiceResult(
            response=GenerationProviderResponse(
                answer_text="Saved generated answer.",
                provider_model="fake-model",
                finish_reason="stop",
            ),
            context_chunk_ids=(),
            prompt_template_version="test-prompt-v1",
            provider_policy_version="test-policy-v1",
            provider_called=False,
        )

    BenchmarkExecutor(
        chunk_retriever=retrieve,
        answer_generator=generate,
    ).execute(run["id"])
    return run["id"]


def test_question_scores_keep_chunk_rank_and_deduplicate_recall() -> None:
    """Repeated source documents count once while retaining original MRR rank.

    Returns:
        None. Assertions verify metric formulas, empty rankings, and missing labels.
    """
    result = score_question({"a", "b"}, ["x", "a", "a", "b"], 3)
    assert result["scores"] == {"hit_rate_at_k": 1.0, "recall_at_k": 0.5, "mrr": 0.5}
    assert result["matching_ranks"] == [2, 3]
    assert score_question({"a"}, [], 3)["scores"]["mrr"] == 0.0
    assert (
        score_question(set(), ["a"], 3)["skip_reason"] == "no_resolved_document_labels"
    )


def test_attempts_score_saved_results_and_survive_restart() -> None:
    """Verify independent scoring, re-evaluation, and restart recovery.

    Returns:
        None. Assertions verify durable lifecycle and benchmark isolation.
    """
    with (
        TemporaryDirectory() as directory,
        patch("backend.db.connection.DATABASE_PATH", Path(directory) / "test.sqlite3"),
    ):
        _seed_ready_inputs()
        _, _, config = resolve_benchmark_configuration(
            "prepared-index-1", "dataset-1", _experiment_configuration()
        )
        run = create_pending_benchmark_run("prepared-index-1", "dataset-1", config)

        # API rejects a pending benchmark with a stable structured error.
        with pytest.raises(HTTPException) as error:
            asyncio.run(create_run_evaluation(run["id"]))
        assert error.value.status_code == 409
        assert error.value.detail["code"] == "run_not_completed"

        # Simulate immutable completed saved output with one labelled question.
        with connect() as connection:
            connection.execute(
                """INSERT INTO document
                   (id, corpus_id, original_filename, storage_path, size_bytes,
                    content_sha256, uploaded_at)
                   VALUES ('doc-a', 'corpus-1', 'a.txt', '/tmp/eval-a.txt', 1, 'a', '2026-09-01')"""
            )
            connection.execute(
                """INSERT INTO evaluation_example_relevant_document
                   (example_id, document_id) VALUES ('example-0', 'doc-a')"""
            )
            # Two saved chunks from the same document prove recall deduplication.
            for ordinal in range(2):
                connection.execute(
                    """INSERT INTO chunk
                       (id, chunk_set_id, source_document_id, ordinal, text)
                       VALUES (?, 'chunk-set-1', 'doc-a', ?, 'saved')""",
                    (f"chunk-{ordinal}", ordinal),
                )
            connection.execute(
                """INSERT INTO benchmark_retrieval_result
                   (id, example_run_id, vector_index_id, requested_top_k,
                    returned_count, distance_metric, duration_ms, created_at)
                   VALUES ('result-0', ?, 'vector-index-1', 3, 2,
                           'cosine', 1, '2026-09-01')""",
                (run["examples"][0]["id"],),
            )
            for rank in range(1, 3):
                connection.execute(
                    """INSERT INTO benchmark_retrieved_chunk
                       (retrieval_result_id, rank, chunk_id, raw_distance)
                       VALUES ('result-0', ?, ?, ?)""",
                    (rank, f"chunk-{rank - 1}", float(rank)),
                )
            connection.execute(
                "UPDATE benchmark_run SET status = 'completed', completed_examples = 2 WHERE id = ?",
                (run["id"],),
            )
        first = create_evaluation(run["id"])
        assert first["configuration"]["retrieval_metrics"] == ["hit_rate_at_k"]
        assert claim_next_pending_work_item() == {
            "kind": "retrieval_evaluation",
            "id": first["id"],
        }
        completed = execute_evaluation(first["id"])
        assert completed["status"] == "completed"
        assert completed["eligible_count"] == 1
        assert completed["aggregates"] == {"hit_rate_at_k": 1.0}
        assert len(completed["questions"]) == 2
        assert completed["questions"][0]["matching_ranks"] == [1, 2]
        assert completed["questions"][1]["skip_reason"]

        # A second selection creates separate history; interrupted recovery
        # changes only that attempt, preserving the completed benchmark.
        second = create_evaluation(run["id"], ["mrr"])
        claim_next_pending_work_item()
        assert fail_interrupted_evaluations() == 1
        assert get_evaluation(run["id"], second["id"])["error"]["code"] == (
            "evaluation_interrupted"
        )
        assert get_evaluation(run["id"], first["id"])["status"] == "completed"
        with connect() as connection:
            status = connection.execute(
                "SELECT status FROM benchmark_run WHERE id = ?", (run["id"],)
            ).fetchone()["status"]
        assert status == "completed"


def test_zero_eligible_questions_have_null_aggregates() -> None:
    """Verify an unlabelled dataset reports coverage without division by zero.

    Returns:
        None. Assertions verify null aggregates and zero eligible coverage.
    """
    with (
        TemporaryDirectory() as directory,
        patch("backend.db.connection.DATABASE_PATH", Path(directory) / "empty.sqlite3"),
    ):
        _seed_ready_inputs()
        _, _, config = resolve_benchmark_configuration(
            "prepared-index-1", "dataset-1", _experiment_configuration()
        )
        run = create_pending_benchmark_run("prepared-index-1", "dataset-1", config)
        with connect() as connection:
            connection.execute(
                "UPDATE benchmark_run SET status = 'completed', completed_examples = 2 WHERE id = ?",
                (run["id"],),
            )
        attempt = create_evaluation(run["id"], ["mrr", "recall_at_k"])
        claim_next_pending_work_item()
        result = execute_evaluation(attempt["id"])
        assert result["eligible_count"] == 0
        assert result["total_count"] == 2
        assert result["aggregates"] == {"mrr": None, "recall_at_k": None}


def test_answer_metrics_use_one_call_per_question_and_saved_outputs() -> None:
    """Verify combined judge calls, answer aggregates, and request provenance.

    Returns:
        None. Assertions cover saved inputs without retrieval or generation reruns.
    """
    with (
        TemporaryDirectory() as directory,
        patch(
            "backend.db.connection.DATABASE_PATH", Path(directory) / "answer.sqlite3"
        ),
    ):
        run_id = _complete_generated_benchmark()
        automatic = claim_next_pending_work_item()
        assert automatic is not None
        execute_evaluation(automatic["id"])
        attempt = create_evaluation(
            run_id,
            [],
            ["groundedness", "answer_relevance", "answer_correctness"],
        )
        assert claim_next_pending_work_item() == {
            "kind": "retrieval_evaluation",
            "id": attempt["id"],
        }
        judge = FakeAnswerJudge()
        result = execute_evaluation(attempt["id"], judge)

        # All three eligible metrics share one request for each persisted question.
        assert len(judge.calls) == 2
        assert judge.calls[0].question == "Question 0?"
        assert judge.calls[0].answer == "Saved generated answer."
        assert judge.calls[0].reference_answer == "Answer 0"
        assert judge.calls[0].metrics == (
            "groundedness",
            "answer_relevance",
            "answer_correctness",
        )
        assert result["aggregates"] == {
            "groundedness": 0.75,
            "answer_relevance": 0.75,
            "answer_correctness": 0.75,
        }
        assert result["coverage"]["answer_correctness"]["scored"] == 2
        assert result["questions"][0]["judge"]["provider_request_id"] == "judge-1"


def test_answer_evaluation_keeps_partial_results_after_question_failure() -> None:
    """Verify one judge failure retains earlier scores and completes with errors.

    Returns:
        None. Assertions cover partial aggregates and structured question errors.
    """
    with (
        TemporaryDirectory() as directory,
        patch(
            "backend.db.connection.DATABASE_PATH", Path(directory) / "partial.sqlite3"
        ),
    ):
        run_id = _complete_generated_benchmark()
        automatic = claim_next_pending_work_item()
        assert automatic is not None
        execute_evaluation(automatic["id"])
        attempt = asyncio.run(
            create_run_evaluation(
                run_id,
                EvaluationRequest(
                    retrieval_metrics=[],
                    answer_metrics=["answer_relevance"],
                ),
            )
        )
        claim_next_pending_work_item()
        result = execute_evaluation(attempt.id, FakeAnswerJudge(fail_on_call=2))

        assert result["status"] == "completed"
        assert result["has_errors"] is True
        assert result["aggregates"] == {"answer_relevance": 0.75}
        assert result["coverage"]["answer_relevance"] == {
            "total": 2,
            "eligible": 2,
            "scored": 1,
            "skipped": 0,
            "error": 1,
        }
        assert result["questions"][1]["answer_error"]["code"] == (
            "evaluator_unavailable"
        )


def test_answer_only_selection_is_enqueued_and_missing_reference_is_skipped() -> None:
    """Verify automatic answer evaluation and correctness eligibility coverage.

    Returns:
        None. Assertions cover answer-only queueing and per-question reference skips.
    """
    with (
        TemporaryDirectory() as directory,
        patch("backend.db.connection.DATABASE_PATH", Path(directory) / "auto.sqlite3"),
    ):
        run_id = _complete_generated_benchmark(["answer_correctness"])
        with connect() as connection:
            connection.execute(
                "UPDATE evaluation_example SET reference_answer = NULL WHERE id = ?",
                ("example-1",),
            )
        claimed = claim_next_pending_work_item()
        assert claimed is not None
        judge = FakeAnswerJudge()
        result = execute_evaluation(claimed["id"], judge)

        assert result["benchmark_run_id"] == run_id
        assert len(judge.calls) == 1
        assert result["aggregates"] == {"answer_correctness": 0.75}
        assert result["coverage"]["answer_correctness"] == {
            "total": 2,
            "eligible": 1,
            "scored": 1,
            "skipped": 1,
            "error": 0,
        }
        assert result["questions"][1]["answer_skips"] == {
            "answer_correctness": "no_reference_answer"
        }
