"""Deterministic offline coverage for saved retrieval scoring and attempt lifecycle."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from test_benchmark_runs import _experiment_configuration, _seed_ready_inputs

from backend.api.routers.runs import create_run_evaluation
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
from backend.evaluation.retrieval import score_question


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
