"""Offline tests for dataset-wide benchmark persistence and execution."""

import asyncio
import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from backend.api.routers.runs import (
    BenchmarkRunCreateRequest,
    BenchmarkRunResponse,
    _add_benchmark_stage_statuses,
    create_benchmark_run,
    read_benchmark_runs,
)
from backend.db.connection import connect
from backend.db.repositories.benchmark_runs import (
    BenchmarkInputMismatchError,
    create_pending_benchmark_run,
    list_benchmark_runs,
    resolve_benchmark_configuration,
)
from backend.db.repositories.work_queue import claim_next_pending_work_item
from backend.generation.models import (
    GenerationProviderResponse,
    GenerationServiceResult,
)
from backend.pipeline.benchmark_execution import BenchmarkExecutor
from backend.pipeline.configs import ExperimentConfig


def _experiment_configuration() -> ExperimentConfig:
    """Return deterministic query-time settings for benchmark tests."""
    # Use catalog-compatible identifiers even though stage providers are mocked.
    return ExperimentConfig.model_validate(
        {
            "retrieval": {"top_k": 3},
            "generation": {
                "provider": "groq",
                "model": "openai/gpt-oss-20b",
                "temperature": 0.2,
                "max_output_tokens": 100,
            },
            "evaluation": {
                "retrieval_metrics": ["hit_rate_at_k"],
                "answer_metrics": [],
            },
        }
    )


def _seed_ready_inputs() -> None:
    """Seed one ready index and a two-question dataset sharing one corpus."""
    preparation = {
        "chunking": {
            "strategy": "recursive",
            "chunk_size_tokens": 800,
            "chunk_overlap_tokens": 100,
        },
        "embedding": {
            "provider": "ollama",
            "model": "nomic-embed-text",
            "distance_metric": "cosine",
        },
    }

    # Explicit column lists keep the fixture stable as unrelated tables evolve.
    with connect() as connection:
        connection.execute(
            "INSERT INTO corpus VALUES (?, ?, ?, ?, ?)",
            ("corpus-1", "Docs", None, "2026-09-01T00:00:00Z", "2026-09-01T00:00:00Z"),
        )
        connection.execute(
            """
            INSERT INTO chunk_set (
                id, corpus_id, fingerprint, chunking_config_json,
                chunker_name, chunker_version, status, chunk_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'ready', 1, ?)
            """,
            (
                "chunk-set-1",
                "corpus-1",
                "chunk-fingerprint",
                json.dumps(preparation["chunking"]),
                "recursive",
                "1",
                "2026-09-01T00:00:01Z",
            ),
        )
        connection.execute(
            """
            INSERT INTO vector_index (
                id, chunk_set_id, fingerprint, embedding_config_json,
                provider, model, dimensions, distance_metric,
                input_policy_version, indexer_name, indexer_version,
                collection_name, status, vector_count, created_at,
                started_at, completed_at, duration_ms
            ) VALUES (?, ?, ?, ?, ?, ?, 3, 'cosine', ?, ?, ?, ?, 'ready', 1, ?, ?, ?, 1)
            """,
            (
                "vector-index-1",
                "chunk-set-1",
                "vector-fingerprint",
                json.dumps(preparation["embedding"]),
                "ollama",
                "nomic-embed-text",
                "raw-v1",
                "chroma",
                "1",
                "test-collection",
                "2026-09-01T00:00:02Z",
                "2026-09-01T00:00:02Z",
                "2026-09-01T00:00:02Z",
            ),
        )
        connection.execute(
            """
            INSERT INTO prepared_index (
                id, name, corpus_id, chunk_set_id, vector_index_id,
                effective_config_json, status, current_stage,
                chunk_set_reused, vector_index_reused,
                chunking_duration_ms, embedding_duration_ms,
                created_at, started_at, completed_at, duration_ms
            ) VALUES (?, ?, ?, ?, ?, ?, 'ready', NULL, 1, 1, 0, 0, ?, ?, ?, 0)
            """,
            (
                "prepared-index-1",
                "Baseline",
                "corpus-1",
                "chunk-set-1",
                "vector-index-1",
                json.dumps(preparation),
                "2026-09-01T00:00:03Z",
                "2026-09-01T00:00:03Z",
                "2026-09-01T00:00:03Z",
            ),
        )
        connection.execute(
            """
            INSERT INTO evaluation_dataset
                (id, name, corpus_id, source_filename, source_sha256, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "dataset-1",
                "Questions",
                "corpus-1",
                "questions.json",
                "a" * 64,
                "2026-09-01T00:00:04Z",
            ),
        )

        # Two rows verify that one benchmark owns ordered internal executions.
        for ordinal in range(2):
            connection.execute(
                """
                INSERT INTO evaluation_example
                    (id, dataset_id, ordinal, question, reference_answer)
                VALUES (?, 'dataset-1', ?, ?, ?)
                """,
                (
                    f"example-{ordinal}",
                    ordinal,
                    f"Question {ordinal}?",
                    f"Answer {ordinal}",
                ),
            )


def test_benchmark_reuses_ready_index_for_all_dataset_examples() -> None:
    """Verify one benchmark executes every question without preparation stages."""
    database_directory = TemporaryDirectory()
    database_path = Path(database_directory.name) / "benchmark.sqlite3"

    # Redirect all repository connections to a disposable isolated database.
    with patch("backend.db.connection.DATABASE_PATH", database_path):
        _seed_ready_inputs()
        experiment = _experiment_configuration()
        _, _, effective = resolve_benchmark_configuration(
            "prepared-index-1",
            "dataset-1",
            experiment,
        )
        pending = create_pending_benchmark_run(
            "prepared-index-1",
            "dataset-1",
            effective,
        )
        claimed = claim_next_pending_work_item()

        # The shared FIFO queue must route benchmarks to the query-time executor.
        assert claimed is not None
        assert claimed["kind"] == "benchmark_run"

        # Deterministic fakes avoid embedding, Chroma, tokenizer, and paid API calls.
        def retrieve(*args: Any, **kwargs: Any) -> tuple[()]:
            """Return no ranked context for one test question."""
            return ()

        def generate(*args: Any, **kwargs: Any) -> GenerationServiceResult:
            """Return one deterministic answer requiring no prompt context."""
            return GenerationServiceResult(
                response=GenerationProviderResponse(
                    answer_text="Deterministic answer",
                    provider_model="test-model",
                    finish_reason="stop",
                ),
                context_chunk_ids=(),
                prompt_template_version="test-prompt-v1",
                provider_policy_version="test-provider-v1",
                provider_called=False,
            )

        executor = BenchmarkExecutor(
            chunk_retriever=retrieve,
            answer_generator=generate,
        )
        completed = executor.execute(claimed["id"])

        # The aggregate is one run while each dataset example remains inspectable.
        assert pending["total_examples"] == 2
        assert completed["status"] == "completed"
        assert completed["completed_examples"] == 2
        assert [example["question"] for example in completed["examples"]] == [
            "Question 0?",
            "Question 1?",
        ]
        assert all(
            example["status"] == "completed" for example in completed["examples"]
        )
        assert all(
            example["generation"]["answer"] == "Deterministic answer"
            for example in completed["examples"]
        )
        assert completed["vector_index_id"] == "vector-index-1"
        assert completed["latest_evaluation"]["status"] == "pending"
        assert claim_next_pending_work_item()["kind"] == "retrieval_evaluation"

        # The complete repository shape must satisfy the public polling contract.
        response = BenchmarkRunResponse.model_validate(
            _add_benchmark_stage_statuses(completed)
        )
        assert response.examples[0].generation is not None

        # List results retain successful-stage coverage without hydrating questions.
        summary = list_benchmark_runs()[0]
        assert "examples" not in summary
        assert summary["configuration"] == completed["configuration"]
        assert summary["metrics"]["retrieval_result_count"] == 2
        assert summary["metrics"]["generation_result_count"] == 2
        assert summary["metrics"]["prompt_tokens"] is None
        assert summary["metrics"]["prompt_token_result_count"] == 0

        # Mixed unknown and genuine zero usage must remain distinguishable.
        with connect() as connection:
            connection.execute(
                """UPDATE benchmark_generation_result
                   SET prompt_tokens = 0, completion_tokens = 7, duration_ms = 20
                   WHERE example_run_id = ?""",
                (completed["examples"][0]["id"],),
            )
            connection.execute(
                """UPDATE benchmark_generation_result SET duration_ms = 40
                   WHERE example_run_id = ?""",
                (completed["examples"][1]["id"],),
            )
        metrics = list_benchmark_runs()[0]["metrics"]
        assert metrics["prompt_tokens"] == 0
        assert metrics["completion_tokens"] == 7
        assert metrics["prompt_token_result_count"] == 1
        assert metrics["completion_token_result_count"] == 1
        assert metrics["average_generation_duration_ms"] == 30

        # Simulate a generation failure after retrieval: its retrieval still counts.
        with connect() as connection:
            connection.execute(
                "DELETE FROM benchmark_generation_result WHERE example_run_id = ?",
                (completed["examples"][1]["id"],),
            )
            connection.execute(
                "UPDATE benchmark_example_run SET status = 'failed' WHERE id = ?",
                (completed["examples"][1]["id"],),
            )
            connection.execute(
                "UPDATE benchmark_run SET status = 'failed', completed_examples = 1 WHERE id = ?",
                (completed["id"],),
            )
        partial = list_benchmark_runs()[0]
        assert partial["failed_examples"] == 1
        assert partial["completed_examples"] == 1
        assert partial["metrics"]["retrieval_result_count"] == 2
        assert partial["metrics"]["generation_result_count"] == 1
        assert partial["metrics"]["average_generation_duration_ms"] == 20

    database_directory.cleanup()


@pytest.mark.parametrize("run_status", ["pending", "running", "failed"])
def test_history_summary_preserves_lifecycle_and_error(
    tmp_path: Path, run_status: str
) -> None:
    """Use isolated inputs and a lifecycle value to verify compact API summaries."""
    with patch("backend.db.connection.DATABASE_PATH", tmp_path / "history.sqlite3"):
        assert asyncio.run(read_benchmark_runs()) == []
        _seed_ready_inputs()
        _, _, config = resolve_benchmark_configuration(
            "prepared-index-1", "dataset-1", _experiment_configuration()
        )
        run = create_pending_benchmark_run("prepared-index-1", "dataset-1", config)
        # Simulate worker state without making any provider requests.
        with connect() as connection:
            connection.execute(
                "UPDATE benchmark_run SET status = ? WHERE id = ?",
                (run_status, run["id"]),
            )
            connection.execute(
                "UPDATE benchmark_example_run SET status = ? WHERE id = ?",
                (run_status, run["examples"][0]["id"]),
            )
            # Failed parents retain safe errors independently of child progress counts.
            if run_status == "failed":
                connection.execute(
                    """UPDATE benchmark_run SET error_code = 'generation_request_timeout',
                       error_details_json = ? WHERE id = ?""",
                    (
                        json.dumps({"message": "Timed out", "stage": "generation"}),
                        run["id"],
                    ),
                )
        summary = asyncio.run(read_benchmark_runs())[0]
        assert summary.configuration == config
        assert summary.pending_examples == (2 if run_status == "pending" else 1)
        assert summary.running_examples == int(run_status == "running")
        assert summary.failed_examples == int(run_status == "failed")
        assert summary.metrics.average_retrieval_duration_ms is None
        assert summary.metrics.prompt_tokens is None
        # Serialize the public error only for failed runs.
        if run_status == "failed":
            assert summary.error is not None
            assert summary.error.code == "generation_request_timeout"
        else:
            assert summary.error is None


def test_history_api_returns_structured_persistence_error() -> None:
    """A database failure returns a safe HTTP error rather than raw SQLite details."""
    with (
        patch(
            "backend.api.routers.runs.list_benchmark_runs",
            side_effect=sqlite3.OperationalError,
        ),
        pytest.raises(HTTPException) as captured,
    ):
        asyncio.run(read_benchmark_runs())
    assert captured.value.status_code == 500
    assert captured.value.detail["code"] == "persistence_error"


def test_benchmark_rejects_dataset_from_another_corpus() -> None:
    """Verify corpus lineage prevents irrelevant relevance labels from launching."""
    database_directory = TemporaryDirectory()
    database_path = Path(database_directory.name) / "benchmark.sqlite3"

    # Add a second corpus and move the dataset there to create an incompatible pair.
    with patch("backend.db.connection.DATABASE_PATH", database_path):
        _seed_ready_inputs()

        with connect() as connection:
            connection.execute(
                "INSERT INTO corpus VALUES (?, ?, ?, ?, ?)",
                (
                    "corpus-2",
                    "Other",
                    None,
                    "2026-09-01T00:00:00Z",
                    "2026-09-01T00:00:00Z",
                ),
            )
            connection.execute(
                "UPDATE evaluation_dataset SET corpus_id = 'corpus-2' WHERE id = 'dataset-1'"
            )

        try:
            resolve_benchmark_configuration(
                "prepared-index-1",
                "dataset-1",
                _experiment_configuration(),
            )
        except BenchmarkInputMismatchError:
            pass
        else:
            raise AssertionError("Mismatched corpus lineage should be rejected.")

        # A rejected launch creates no durable user-visible run.
        with connect() as connection:
            count = connection.execute("SELECT COUNT(*) FROM benchmark_run").fetchone()[
                0
            ]

        assert count == 0

    database_directory.cleanup()


def test_runs_api_enqueues_saved_dataset_benchmark() -> None:
    """Verify the public runs handler accepts only query-time benchmark settings."""
    database_directory = TemporaryDirectory()
    database_path = Path(database_directory.name) / "benchmark.sqlite3"

    # The handler resolves preparation from the index instead of trusting the client.
    with patch("backend.db.connection.DATABASE_PATH", database_path):
        _seed_ready_inputs()
        request = BenchmarkRunCreateRequest(
            prepared_index_id="prepared-index-1",
            dataset_id="dataset-1",
            configuration=_experiment_configuration(),
        )
        response = asyncio.run(create_benchmark_run(request))

        assert isinstance(response, BenchmarkRunResponse)
        assert response.status == "pending"
        assert response.total_examples == 2
        assert response.configuration.embedding.model == "nomic-embed-text"

    database_directory.cleanup()


def test_runs_api_contract_rejects_legacy_single_question_payload() -> None:
    """Verify the runs request contract no longer accepts ad-hoc questions.

    Returns:
        None. Validation must reject the removed corpus/question request shape.
    """
    legacy_payload = {
        "corpus_id": "corpus-1",
        "question": "What is the refund policy?",
        "configuration": _experiment_configuration().model_dump(mode="json"),
    }

    # Required benchmark identities are absent from the removed request format.
    with pytest.raises(ValidationError):
        BenchmarkRunCreateRequest.model_validate(legacy_payload)

    # A question is also forbidden when otherwise-valid benchmark IDs are present.
    mixed_payload = {
        "prepared_index_id": "prepared-index-1",
        "dataset_id": "dataset-1",
        "configuration": _experiment_configuration().model_dump(mode="json"),
        "question": "This field belonged to the removed workflow.",
    }

    with pytest.raises(ValidationError):
        BenchmarkRunCreateRequest.model_validate(mixed_payload)
