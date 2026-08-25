"""Coordinate pipeline stages while keeping stage implementations independent."""

import logging
import sqlite3
from collections.abc import Callable
from time import perf_counter
from typing import Any

from backend.db.repositories.runs import (
    ChunkSetNotReadyError,
    InvalidRunStateError,
    complete_run,
    create_pending_run,
    fail_run,
    mark_run_running,
)
from backend.ingestion.chunk_sets import (
    ChunkingCorpusNotFoundError,
    ChunkSetBuildResult,
    EmptyChunkingCorpusError,
    MissingParseArtifactError,
    build_or_reuse_chunk_set,
)
from backend.ingestion.chunkers.models import ChunkingTokenizer
from backend.ingestion.chunkers.tokenizer import TokenizerAssetError
from backend.pipeline.configs import ChunkingConfig, PipelineConfig

# Use the module name to identify overall pipeline lifecycle records.
logger = logging.getLogger(__name__)

# A chunk-set builder receives immutable inputs and returns a ready reusable artifact.
ChunkSetBuilder = Callable[
    [str, ChunkingConfig, ChunkingTokenizer | None],
    ChunkSetBuildResult,
]


class PipelineRunExecutionError(RuntimeError):
    """Expose one safe failed-run result to the API transport layer."""

    def __init__(
        self,
        run_id: str,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Store the persisted run identity and structured public failure.

        Args:
            run_id: Stable identifier of the failed pipeline run.
            code: Machine-readable failure category.
            message: Safe user-readable failure explanation.
            details: Optional additional safe structured fields.

        Returns:
            None. The initialized exception carries the transport-safe failure.
        """
        # Initialize RuntimeError for conventional logging and exception chaining.
        super().__init__(message)
        self.run_id = run_id
        self.code = code
        self.message = message
        self.details = details or {}


class PipelineExecutor:
    """Run pipeline stages in order and persist the overall run lifecycle.

    The executor currently coordinates chunking only. Future embedding, retrieval,
    generation, and evaluation services will be injected here as independent stages;
    their provider logic must remain outside this coordinator.
    """

    def __init__(
        self,
        chunk_set_builder: ChunkSetBuilder = build_or_reuse_chunk_set,
        tokenizer: ChunkingTokenizer | None = None,
    ) -> None:
        """Configure the independently testable chunking-stage dependency.

        Args:
            chunk_set_builder: Service that builds or reuses one ready chunk set.
            tokenizer: Optional tokenizer override used by deterministic tests.

        Returns:
            None. Dependencies are retained without mutable execution state.
        """
        self._chunk_set_builder = chunk_set_builder
        self._tokenizer = tokenizer

    def execute(
        self,
        corpus_id: str,
        question: str,
        configuration: PipelineConfig,
    ) -> dict[str, Any]:
        """Execute every currently implemented stage for one immutable run.

        Args:
            corpus_id: Stable identifier of the selected immutable corpus.
            question: Normalized non-empty user question.
            configuration: Fully resolved and compatibility-validated pipeline config.

        Returns:
            Completed persisted run with a compact chunking artifact summary.

        Raises:
            PipelineRunExecutionError: If chunking or lifecycle persistence fails.
            CorpusNotFoundError: Indirectly if pending-run creation finds no corpus.
            sqlite3.Error: If the pending run cannot be created or started.
        """
        # Persist the immutable request first so execution failures remain auditable.
        pending_run = create_pending_run(corpus_id, question, configuration)
        run_id = pending_run["id"]
        started_counter = perf_counter()
        logger.info("pipeline_run_created run_id=%s corpus_id=%s", run_id, corpus_id)

        mark_run_running(run_id)
        logger.info("pipeline_run_started run_id=%s corpus_id=%s", run_id, corpus_id)

        try:
            # Chunking is the first and currently only executable pipeline stage.
            chunking_result = self._chunk_set_builder(
                corpus_id,
                configuration.chunking,
                self._tokenizer,
            )
            duration_ms = _elapsed_milliseconds(started_counter)

            # A run completes only after it references a fully persisted ready artifact.
            completed_run = complete_run(
                run_id,
                chunking_result.artifact["id"],
                chunking_result.reused,
                duration_ms,
            )
            logger.info(
                "pipeline_run_completed run_id=%s corpus_id=%s chunk_set_id=%s "
                "chunk_set_reused=%s duration_ms=%d",
                run_id,
                corpus_id,
                chunking_result.artifact["id"],
                chunking_result.reused,
                duration_ms,
            )
            return completed_run
        except Exception as error:
            # Convert stage-specific errors into one safe, persisted run failure shape.
            execution_error = _map_execution_error(run_id, error)
            duration_ms = _elapsed_milliseconds(started_counter)
            logger.exception(
                "pipeline_run_stage_failed run_id=%s corpus_id=%s stage=chunking "
                "error_code=%s duration_ms=%d",
                run_id,
                corpus_id,
                execution_error.code,
                duration_ms,
            )
            error_details = {
                "stage": "chunking",
                "message": execution_error.message,
                **execution_error.details,
            }

            try:
                # Preserve the failure against the already-created running run.
                fail_run(
                    run_id,
                    execution_error.code,
                    error_details,
                    duration_ms,
                )
            except (sqlite3.Error, InvalidRunStateError) as persistence_error:
                logger.exception(
                    "pipeline_run_failure_persistence_failed run_id=%s corpus_id=%s",
                    run_id,
                    corpus_id,
                )

                # If failure recording itself fails, expose a persistence error without
                # leaking either database details or the original stage exception.
                raise PipelineRunExecutionError(
                    run_id,
                    "persistence_error",
                    "The failed pipeline run could not be recorded.",
                ) from persistence_error

            raise execution_error from error


def _elapsed_milliseconds(started_counter: float) -> int:
    """Calculate a non-negative elapsed duration from a monotonic start value.

    Args:
        started_counter: Value previously returned by ``perf_counter``.

    Returns:
        Rounded non-negative elapsed milliseconds.
    """
    # Clamp defensively even though a monotonic clock should never move backward.
    return max(0, round((perf_counter() - started_counter) * 1000))


def _map_execution_error(
    run_id: str,
    error: Exception,
) -> PipelineRunExecutionError:
    """Translate an internal chunking failure into a stable public category.

    Args:
        run_id: Stable identifier of the run that encountered the failure.
        error: Internal exception raised by chunking or lifecycle persistence.

    Returns:
        Safe structured pipeline execution error suitable for persistence and HTTP.
    """
    # A known corpus with no documents cannot produce a meaningful chunk artifact.
    if isinstance(error, EmptyChunkingCorpusError):
        return PipelineRunExecutionError(
            run_id,
            "empty_corpus",
            "The selected corpus does not contain any documents to chunk.",
        )

    # Report all documents missing their canonical parse in one repairable response.
    if isinstance(error, MissingParseArtifactError):
        return PipelineRunExecutionError(
            run_id,
            "missing_parse_artifact",
            "Canonical parsing is incomplete for the selected corpus.",
            {"document_ids": error.document_ids},
        )

    # A corpus removed or otherwise unavailable between lifecycle steps is not usable.
    if isinstance(error, ChunkingCorpusNotFoundError):
        return PipelineRunExecutionError(
            run_id,
            "corpus_not_found",
            "The selected corpus does not exist.",
        )

    # Missing or modified pinned assets are operational backend failures.
    if isinstance(error, TokenizerAssetError):
        return PipelineRunExecutionError(
            run_id,
            "chunking_tokenizer_unavailable",
            "The configured chunking tokenizer is unavailable.",
        )

    # Database and invalid lifecycle/artifact states share the persistence boundary.
    if isinstance(
        error,
        (sqlite3.Error, InvalidRunStateError, ChunkSetNotReadyError),
    ):
        return PipelineRunExecutionError(
            run_id,
            "persistence_error",
            "The pipeline execution state could not be saved.",
        )

    # Hide unexpected implementation details behind one stable chunking error.
    return PipelineRunExecutionError(
        run_id,
        "chunking_failed",
        "The selected corpus could not be chunked.",
    )


def get_pipeline_executor() -> PipelineExecutor:
    """Provide the stateless production pipeline executor to FastAPI.

    Args:
        None.

    Returns:
        Executor configured with the pinned production chunking tokenizer.
    """
    # A fresh coordinator carries no run state and remains safe for concurrent requests.
    return PipelineExecutor()
