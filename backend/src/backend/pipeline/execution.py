"""Coordinate persisted pipeline stages while keeping implementations independent."""

import logging
import sqlite3
from collections.abc import Callable
from time import perf_counter
from typing import Any

from backend.db.repositories.retrieval_results import (
    InvalidRetrievalResultError,
    RetrievalArtifactMismatchError,
)
from backend.db.repositories.runs import (
    ChunkSetNotReadyError,
    InvalidRunStateError,
    RunNotFoundError,
    VectorIndexArtifactMismatchError,
    VectorIndexNotReadyError,
    complete_run_with_retrieval,
    fail_run,
    get_run_execution_input,
    record_chunking_result,
    record_embedding_result,
)
from backend.embedding.models import (
    EmbeddingAuthenticationError,
    EmbeddingInputTooLargeError,
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingProviderUnavailableError,
    EmbeddingRateLimitError,
    EmbeddingRequestRejectedError,
    EmbeddingRequestTimeoutError,
    InvalidEmbeddingResponseError,
    VectorStore,
    VectorStoreError,
)
from backend.embedding.service import (
    EmptyChunkSetError,
    VectorIndexBuildResult,
    build_or_reuse_vector_index,
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
from backend.pipeline.configs import ChunkingConfig, EmbeddingConfig, RetrievalConfig
from backend.retrieval.chunk_hydration import ChunkHydrationError
from backend.retrieval.models import HydratedVectorSearchHit
from backend.retrieval.service import (
    InvalidRetrievalArtifactError,
    retrieve_chunks,
)
from backend.retrieval.vector_search import (
    IncompatibleVectorIndexError,
    InvalidVectorSearchRequestError,
)

# Use the module name to identify overall pipeline lifecycle records.
logger = logging.getLogger(__name__)

# A chunk builder receives immutable inputs and returns a reusable ready artifact.
ChunkSetBuilder = Callable[
    [str, ChunkingConfig, ChunkingTokenizer | None],
    ChunkSetBuildResult,
]

# A vector-index builder consumes a ready chunk artifact and provider/store adapters.
VectorIndexBuilder = Callable[
    [
        dict[str, Any],
        EmbeddingConfig,
        EmbeddingProvider | None,
        VectorStore | None,
    ],
    VectorIndexBuildResult,
]

# A retriever embeds one question, searches one exact index, and hydrates its hits.
ChunkRetriever = Callable[
    [
        str,
        RetrievalConfig,
        EmbeddingConfig,
        dict[str, Any],
        EmbeddingProvider | None,
        VectorStore | None,
    ],
    tuple[HydratedVectorSearchHit, ...],
]


class PipelineRunExecutionError(RuntimeError):
    """Expose one safe failed-run result to worker and transport boundaries."""

    def __init__(
        self,
        run_id: str,
        stage: str,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Store persisted run identity and structured public failure.

        Args:
            run_id: Stable identifier of the failed pipeline run.
            stage: Pipeline stage that encountered the failure.
            code: Machine-readable failure category.
            message: Safe user-readable failure explanation.
            details: Optional additional safe structured fields.

        Returns:
            None. The exception carries a transport-safe failure.
        """
        # Initialize RuntimeError for conventional logging and exception chaining.
        super().__init__(message)
        self.run_id = run_id
        self.stage = stage
        self.code = code
        self.message = message
        self.details = details or {}


class PipelineExecutor:
    """Execute claimed runs through chunking, embedding, and retrieval."""

    def __init__(
        self,
        chunk_set_builder: ChunkSetBuilder = build_or_reuse_chunk_set,
        vector_index_builder: VectorIndexBuilder = build_or_reuse_vector_index,
        chunk_retriever: ChunkRetriever = retrieve_chunks,
        tokenizer: ChunkingTokenizer | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        """Configure independently testable stage dependencies.

        Args:
            chunk_set_builder: Service that builds or reuses one chunk set.
            vector_index_builder: Service that builds or reuses one vector index.
            chunk_retriever: Service that searches and hydrates ranked chunks.
            tokenizer: Optional chunking tokenizer override for deterministic tests.
            embedding_provider: Optional embedding adapter override for tests.
            vector_store: Optional vector-store adapter override for tests.

        Returns:
            None. Dependencies are retained without mutable run state.
        """
        self._chunk_set_builder = chunk_set_builder
        self._vector_index_builder = vector_index_builder
        self._chunk_retriever = chunk_retriever
        self._tokenizer = tokenizer
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def execute(self, run_id: str) -> dict[str, Any]:
        """Execute one already-claimed persisted run through available stages.

        Args:
            run_id: Stable identifier of a running persisted pipeline job.

        Returns:
            Completed run linked to ready artifacts and its retrieval result.

        Raises:
            PipelineRunExecutionError: If a stage or lifecycle operation fails.
        """
        execution_input = get_run_execution_input(run_id)
        corpus_id = execution_input["corpus_id"]
        configuration = execution_input["configuration"]
        run_started_counter = perf_counter()
        current_stage = "chunking"
        logger.info("pipeline_run_started run_id=%s corpus_id=%s", run_id, corpus_id)

        try:
            # Resolve chunking independently and record its run-specific latency.
            chunking_started_counter = perf_counter()
            chunking_result = self._chunk_set_builder(
                corpus_id,
                configuration.chunking,
                self._tokenizer,
            )
            chunking_duration_ms = _elapsed_milliseconds(chunking_started_counter)
            record_chunking_result(
                run_id,
                chunking_result.artifact["id"],
                chunking_result.reused,
                chunking_duration_ms,
            )
            logger.info(
                "pipeline_run_stage_completed run_id=%s corpus_id=%s "
                "stage=chunking artifact_id=%s reused=%s duration_ms=%d",
                run_id,
                corpus_id,
                chunking_result.artifact["id"],
                chunking_result.reused,
                chunking_duration_ms,
            )

            # Embed only the exact ready chunk artifact attached to this run.
            current_stage = "embedding"
            embedding_started_counter = perf_counter()
            embedding_result = self._vector_index_builder(
                chunking_result.artifact,
                configuration.embedding,
                self._embedding_provider,
                self._vector_store,
            )
            embedding_duration_ms = _elapsed_milliseconds(embedding_started_counter)
            record_embedding_result(
                run_id,
                embedding_result.artifact["id"],
                embedding_result.reused,
                embedding_duration_ms,
            )
            logger.info(
                "pipeline_run_stage_completed run_id=%s corpus_id=%s "
                "stage=embedding artifact_id=%s reused=%s duration_ms=%d",
                run_id,
                corpus_id,
                embedding_result.artifact["id"],
                embedding_result.reused,
                embedding_duration_ms,
            )

            # Retrieve and hydrate query-specific chunks from the attached exact index.
            current_stage = "retrieval"
            retrieval_started_counter = perf_counter()
            retrieval_hits = self._chunk_retriever(
                execution_input["question"],
                configuration.retrieval,
                configuration.embedding,
                embedding_result.artifact,
                self._embedding_provider,
                self._vector_store,
            )
            retrieval_duration_ms = _elapsed_milliseconds(retrieval_started_counter)
            total_duration_ms = _elapsed_milliseconds(run_started_counter)
            completed_run = complete_run_with_retrieval(
                run_id,
                embedding_result.artifact["id"],
                configuration.retrieval.top_k,
                configuration.embedding.distance_metric,
                retrieval_hits,
                retrieval_duration_ms,
                total_duration_ms,
            )
            logger.info(
                "pipeline_run_stage_completed run_id=%s corpus_id=%s "
                "stage=retrieval vector_index_id=%s hit_count=%d "
                "distance_metric=%s duration_ms=%d",
                run_id,
                corpus_id,
                embedding_result.artifact["id"],
                len(retrieval_hits),
                configuration.embedding.distance_metric.value,
                retrieval_duration_ms,
            )
            logger.info(
                "pipeline_run_completed run_id=%s corpus_id=%s "
                "chunk_set_id=%s vector_index_id=%s duration_ms=%d",
                run_id,
                corpus_id,
                chunking_result.artifact["id"],
                embedding_result.artifact["id"],
                total_duration_ms,
            )
            return completed_run
        except Exception as error:
            # Convert stage-specific errors into one safe persisted failure shape.
            execution_error = _map_execution_error(run_id, current_stage, error)
            duration_ms = _elapsed_milliseconds(run_started_counter)
            logger.exception(
                "pipeline_run_stage_failed run_id=%s corpus_id=%s stage=%s "
                "error_code=%s duration_ms=%d",
                run_id,
                corpus_id,
                current_stage,
                execution_error.code,
                duration_ms,
            )
            error_details = {
                "stage": current_stage,
                "message": execution_error.message,
                **execution_error.details,
            }

            try:
                # Preserve the failure and any already-attached upstream artifact.
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
                raise PipelineRunExecutionError(
                    run_id,
                    current_stage,
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
    stage: str,
    error: Exception,
) -> PipelineRunExecutionError:
    """Translate an internal stage failure into a stable public category.

    Args:
        run_id: Stable identifier of the run that encountered the failure.
        stage: Pipeline stage active when the exception was raised.
        error: Internal exception raised by stage or lifecycle code.

    Returns:
        Safe structured pipeline execution error suitable for persistence.
    """
    # Map chunking-domain failures only while the chunking stage is active.
    if stage == "chunking":
        return _map_chunking_error(run_id, error)

    # Embedding and retrieval reuse adapters but need stage-specific public errors.
    if stage == "embedding":
        return _map_embedding_error(run_id, error)

    return _map_retrieval_error(run_id, error)


def _map_chunking_error(
    run_id: str,
    error: Exception,
) -> PipelineRunExecutionError:
    """Translate a chunking failure into a safe pipeline error.

    Args:
        run_id: Stable identifier of the failed run.
        error: Exception raised during chunk-set resolution.

    Returns:
        Safe chunking-stage execution error.
    """
    # A known corpus with no documents cannot produce a chunk artifact.
    if isinstance(error, EmptyChunkingCorpusError):
        return PipelineRunExecutionError(
            run_id,
            "chunking",
            "empty_corpus",
            "The selected corpus does not contain any documents to chunk.",
        )

    # Report every source document missing its canonical parse together.
    if isinstance(error, MissingParseArtifactError):
        return PipelineRunExecutionError(
            run_id,
            "chunking",
            "missing_parse_artifact",
            "Canonical parsing is incomplete for the selected corpus.",
            {"document_ids": error.document_ids},
        )

    # A corpus removed between enqueue and execution is no longer usable.
    if isinstance(error, ChunkingCorpusNotFoundError):
        return PipelineRunExecutionError(
            run_id,
            "chunking",
            "corpus_not_found",
            "The selected corpus does not exist.",
        )

    # Missing or modified pinned tokenizer assets are operational failures.
    if isinstance(error, TokenizerAssetError):
        return PipelineRunExecutionError(
            run_id,
            "chunking",
            "chunking_tokenizer_unavailable",
            "The configured chunking tokenizer is unavailable.",
        )

    # Database and invalid artifact states share the persistence boundary.
    if isinstance(
        error,
        (
            sqlite3.Error,
            InvalidRunStateError,
            ChunkSetNotReadyError,
            RunNotFoundError,
        ),
    ):
        return PipelineRunExecutionError(
            run_id,
            "chunking",
            "persistence_error",
            "The pipeline execution state could not be saved.",
        )

    # Hide unexpected chunker implementation details.
    return PipelineRunExecutionError(
        run_id,
        "chunking",
        "chunking_failed",
        "The selected corpus could not be chunked.",
    )


def _map_embedding_error(
    run_id: str,
    error: Exception,
) -> PipelineRunExecutionError:
    """Translate an embedding or indexing failure into a safe pipeline error.

    Args:
        run_id: Stable identifier of the failed run.
        error: Exception raised while building the vector index.

    Returns:
        Safe embedding-stage execution error.
    """
    # Keep common remote-provider failure modes distinguishable for retry UX.
    provider_errors: tuple[tuple[type[Exception], str, str], ...] = (
        (
            EmbeddingProviderUnavailableError,
            "embedding_provider_unavailable",
            "The embedding provider could not complete the request.",
        ),
        (
            EmbeddingRequestTimeoutError,
            "embedding_request_timeout",
            "The embedding provider request timed out.",
        ),
        (
            EmbeddingAuthenticationError,
            "embedding_authentication_failed",
            "The embedding provider rejected backend authentication.",
        ),
        (
            EmbeddingRateLimitError,
            "embedding_rate_limited",
            "The embedding provider rate limit was reached.",
        ),
        (
            EmbeddingInputTooLargeError,
            "embedding_input_too_large",
            "A chunk exceeds the selected embedding model's input limit.",
        ),
        (
            EmbeddingRequestRejectedError,
            "embedding_request_rejected",
            "The embedding provider rejected the model or request.",
        ),
        (
            InvalidEmbeddingResponseError,
            "invalid_embedding_response",
            "The embedding provider returned an invalid vector response.",
        ),
    )

    # Return the first matching specific provider category.
    for error_type, code, message in provider_errors:
        if isinstance(error, error_type):
            return PipelineRunExecutionError(
                run_id,
                "embedding",
                code,
                message,
            )

    # A ready but empty upstream artifact cannot produce a searchable index.
    if isinstance(error, EmptyChunkSetError):
        return PipelineRunExecutionError(
            run_id,
            "embedding",
            "empty_chunk_set",
            "The chunk artifact does not contain any text to embed.",
        )

    # Chroma failures remain separate from remote embedding-provider failures.
    if isinstance(error, VectorStoreError):
        return PipelineRunExecutionError(
            run_id,
            "embedding",
            "vector_store_unavailable",
            "The vector index could not be stored.",
        )

    # Relational state failures must not leak SQLite or constraint details.
    if isinstance(
        error,
        (
            sqlite3.Error,
            InvalidRunStateError,
            VectorIndexArtifactMismatchError,
            VectorIndexNotReadyError,
            RunNotFoundError,
        ),
    ):
        return PipelineRunExecutionError(
            run_id,
            "embedding",
            "vector_index_persistence_failed",
            "The vector-index execution state could not be saved.",
        )

    # Hide every other provider or index implementation detail.
    if isinstance(error, EmbeddingProviderError):
        return PipelineRunExecutionError(
            run_id,
            "embedding",
            "embedding_failed",
            "The selected chunks could not be embedded.",
        )

    return PipelineRunExecutionError(
        run_id,
        "embedding",
        "embedding_failed",
        "The selected chunks could not be embedded.",
    )


def _map_retrieval_error(
    run_id: str,
    error: Exception,
) -> PipelineRunExecutionError:
    """Translate query embedding, search, hydration, or persistence failures.

    Args:
        run_id: Stable identifier of the failed run.
        error: Exception raised while resolving and saving retrieved chunks.

    Returns:
        Safe retrieval-stage execution error.
    """
    # Keep common remote query-embedding failures distinguishable for retry UX.
    provider_errors: tuple[tuple[type[Exception], str, str], ...] = (
        (
            EmbeddingProviderUnavailableError,
            "retrieval_provider_unavailable",
            "The embedding provider could not embed the retrieval question.",
        ),
        (
            EmbeddingRequestTimeoutError,
            "retrieval_request_timeout",
            "The retrieval embedding request timed out.",
        ),
        (
            EmbeddingAuthenticationError,
            "retrieval_authentication_failed",
            "The embedding provider rejected backend authentication.",
        ),
        (
            EmbeddingRateLimitError,
            "retrieval_rate_limited",
            "The embedding provider rate limit was reached during retrieval.",
        ),
        (
            EmbeddingInputTooLargeError,
            "retrieval_query_too_large",
            "The question exceeds the embedding model's input limit.",
        ),
        (
            EmbeddingRequestRejectedError,
            "retrieval_request_rejected",
            "The embedding provider rejected the retrieval request.",
        ),
        (
            InvalidEmbeddingResponseError,
            "invalid_retrieval_embedding_response",
            "The embedding provider returned an invalid query vector.",
        ),
    )

    # Return the first matching specific query-provider category.
    for error_type, code, message in provider_errors:
        if isinstance(error, error_type):
            return PipelineRunExecutionError(
                run_id,
                "retrieval",
                code,
                message,
            )

    # Invalid requests should be impossible after API validation but remain explicit.
    if isinstance(error, InvalidVectorSearchRequestError):
        return PipelineRunExecutionError(
            run_id,
            "retrieval",
            "invalid_retrieval_request",
            "The saved question or retrieval limit is invalid.",
        )

    # Compatibility failures prevent querying a stale or unrelated index space.
    if isinstance(
        error,
        (InvalidRetrievalArtifactError, IncompatibleVectorIndexError),
    ):
        return PipelineRunExecutionError(
            run_id,
            "retrieval",
            "incompatible_vector_index",
            "The vector index is incompatible with this retrieval request.",
        )

    # Hydration must resolve every vector hit back to exact application-owned chunks.
    if isinstance(error, ChunkHydrationError):
        return PipelineRunExecutionError(
            run_id,
            "retrieval",
            "retrieval_chunk_hydration_failed",
            "The retrieved chunks could not be loaded safely.",
        )

    # Chroma query failures remain distinct from query-embedding failures.
    if isinstance(error, VectorStoreError):
        return PipelineRunExecutionError(
            run_id,
            "retrieval",
            "retrieval_vector_store_unavailable",
            "The vector index could not be searched.",
        )

    # Relational failures must not leak constraints or internal artifact identifiers.
    if isinstance(
        error,
        (
            sqlite3.Error,
            InvalidRunStateError,
            InvalidRetrievalResultError,
            RetrievalArtifactMismatchError,
            RunNotFoundError,
        ),
    ):
        return PipelineRunExecutionError(
            run_id,
            "retrieval",
            "retrieval_persistence_failed",
            "The retrieval result could not be saved.",
        )

    # Hide all other provider and implementation details from persisted errors.
    if isinstance(error, EmbeddingProviderError):
        return PipelineRunExecutionError(
            run_id,
            "retrieval",
            "retrieval_embedding_failed",
            "The retrieval question could not be embedded.",
        )

    return PipelineRunExecutionError(
        run_id,
        "retrieval",
        "retrieval_failed",
        "The retrieval stage could not be completed.",
    )


def get_pipeline_executor() -> PipelineExecutor:
    """Provide the stateless production pipeline executor.

    Args:
        None.

    Returns:
        Executor configured with production stage services and lazy adapters.
    """
    # A fresh coordinator carries no run state and remains safe across worker jobs.
    return PipelineExecutor()
