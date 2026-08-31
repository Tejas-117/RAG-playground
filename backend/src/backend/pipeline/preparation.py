"""Execute named index preparation through chunking and embedding only."""

import logging
import sqlite3
from collections.abc import Callable
from time import perf_counter
from typing import Any

from backend.db.repositories.prepared_indexes import (
    InvalidPreparedIndexStateError,
    PreparedIndexNotFoundError,
    complete_prepared_index,
    fail_prepared_index,
    get_prepared_index_execution_input,
    record_prepared_chunking_result,
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
from backend.pipeline.configs import ChunkingConfig, EmbeddingConfig

# Use the module name to distinguish named preparation logs from full runs.
logger = logging.getLogger(__name__)

# Chunking produces one reusable artifact from a corpus and resolved configuration.
PreparationChunkSetBuilder = Callable[
    [str, ChunkingConfig, ChunkingTokenizer | None],
    ChunkSetBuildResult,
]

# Embedding produces one reusable vector artifact from the exact ready chunk set.
PreparationVectorIndexBuilder = Callable[
    [
        dict[str, Any],
        EmbeddingConfig,
        EmbeddingProvider | None,
        VectorStore | None,
    ],
    VectorIndexBuildResult,
]


class PreparedIndexExecutionError(RuntimeError):
    """Expose one safe failed preparation result to the worker boundary."""

    def __init__(
        self,
        prepared_index_id: str,
        stage: str,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Store the failed request identity and public structured failure.

        Args:
            prepared_index_id: Stable identifier of the failed named index.
            stage: Preparation stage that encountered the failure.
            code: Machine-readable public failure category.
            message: Safe user-readable failure explanation.
            details: Optional additional safe structured fields.

        Returns:
            None. The exception carries a transport-safe failure.
        """
        # Initialize RuntimeError so worker logs preserve conventional exception text.
        super().__init__(message)
        self.prepared_index_id = prepared_index_id
        self.stage = stage
        self.code = code
        self.message = message
        self.details = details or {}


class PreparedIndexExecutor:
    """Build or reuse chunk and vector artifacts for one named index request."""

    def __init__(
        self,
        chunk_set_builder: PreparationChunkSetBuilder = build_or_reuse_chunk_set,
        vector_index_builder: PreparationVectorIndexBuilder = (
            build_or_reuse_vector_index
        ),
        tokenizer: ChunkingTokenizer | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        """Configure independently testable preparation-stage dependencies.

        Args:
            chunk_set_builder: Service that builds or reuses one chunk set.
            vector_index_builder: Service that builds or reuses one vector index.
            tokenizer: Optional deterministic tokenizer override for tests.
            embedding_provider: Optional embedding adapter override for tests.
            vector_store: Optional vector-store adapter override for tests.

        Returns:
            None. Dependencies are retained without mutable job state.
        """
        self._chunk_set_builder = chunk_set_builder
        self._vector_index_builder = vector_index_builder
        self._tokenizer = tokenizer
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def execute(self, prepared_index_id: str) -> dict[str, Any]:
        """Execute one claimed prepared index through chunking and embedding.

        Args:
            prepared_index_id: Stable identifier of a running preparation request.

        Returns:
            Ready named index linked to exact reusable technical artifacts.

        Raises:
            PreparedIndexExecutionError: If either stage or persistence fails.
        """
        execution_input = get_prepared_index_execution_input(prepared_index_id)
        corpus_id = execution_input["corpus_id"]
        configuration = execution_input["configuration"]
        preparation_started_counter = perf_counter()
        current_stage = "chunking"
        logger.info(
            "prepared_index_started prepared_index_id=%s corpus_id=%s",
            prepared_index_id,
            corpus_id,
        )

        try:
            # Resolve the exact chunk artifact and preserve request-specific latency.
            chunking_started_counter = perf_counter()
            chunking_result = self._chunk_set_builder(
                corpus_id,
                configuration.chunking,
                self._tokenizer,
            )
            chunking_duration_ms = _elapsed_milliseconds(chunking_started_counter)
            record_prepared_chunking_result(
                prepared_index_id,
                chunking_result.artifact["id"],
                chunking_result.reused,
                chunking_duration_ms,
            )
            logger.info(
                "prepared_index_stage_completed prepared_index_id=%s "
                "corpus_id=%s stage=chunking artifact_id=%s reused=%s "
                "duration_ms=%d",
                prepared_index_id,
                corpus_id,
                chunking_result.artifact["id"],
                chunking_result.reused,
                chunking_duration_ms,
            )

            # Build or reuse the vector space derived from that exact chunk artifact.
            current_stage = "embedding"
            embedding_started_counter = perf_counter()
            embedding_result = self._vector_index_builder(
                chunking_result.artifact,
                configuration.embedding,
                self._embedding_provider,
                self._vector_store,
            )
            embedding_duration_ms = _elapsed_milliseconds(embedding_started_counter)
            total_duration_ms = _elapsed_milliseconds(preparation_started_counter)
            prepared_index = complete_prepared_index(
                prepared_index_id,
                embedding_result.artifact["id"],
                embedding_result.reused,
                embedding_duration_ms,
                total_duration_ms,
            )
            logger.info(
                "prepared_index_completed prepared_index_id=%s corpus_id=%s "
                "chunk_set_id=%s vector_index_id=%s vector_index_reused=%s "
                "duration_ms=%d",
                prepared_index_id,
                corpus_id,
                chunking_result.artifact["id"],
                embedding_result.artifact["id"],
                embedding_result.reused,
                total_duration_ms,
            )
            return prepared_index
        except Exception as error:
            # Translate implementation errors into safe persisted preparation failures.
            execution_error = _map_preparation_error(
                prepared_index_id,
                current_stage,
                error,
            )
            duration_ms = _elapsed_milliseconds(preparation_started_counter)
            logger.exception(
                "prepared_index_stage_failed prepared_index_id=%s corpus_id=%s "
                "stage=%s error_code=%s duration_ms=%d",
                prepared_index_id,
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
                # Retain an attached chunk artifact when embedding fails.
                fail_prepared_index(
                    prepared_index_id,
                    execution_error.code,
                    error_details,
                    duration_ms,
                )
            except (sqlite3.Error, InvalidPreparedIndexStateError) as persistence_error:
                logger.exception(
                    "prepared_index_failure_persistence_failed "
                    "prepared_index_id=%s corpus_id=%s",
                    prepared_index_id,
                    corpus_id,
                )
                raise PreparedIndexExecutionError(
                    prepared_index_id,
                    current_stage,
                    "persistence_error",
                    "The failed index preparation could not be recorded.",
                ) from persistence_error

            raise execution_error from error


def get_prepared_index_executor() -> PreparedIndexExecutor:
    """Construct the production prepared-index executor.

    Args:
        None.

    Returns:
        Executor that resolves production adapters lazily inside stage services.
    """
    # A factory keeps worker construction testable and avoids global provider state.
    return PreparedIndexExecutor()


def _elapsed_milliseconds(started_counter: float) -> int:
    """Calculate elapsed whole milliseconds from a monotonic start value.

    Args:
        started_counter: Value previously returned by ``perf_counter``.

    Returns:
        Rounded non-negative elapsed milliseconds.
    """
    # Clamp defensively even though the monotonic clock should not move backward.
    return max(0, round((perf_counter() - started_counter) * 1000))


def _map_preparation_error(
    prepared_index_id: str,
    stage: str,
    error: Exception,
) -> PreparedIndexExecutionError:
    """Translate one internal stage exception into a stable public failure.

    Args:
        prepared_index_id: Stable identifier of the failed named index.
        stage: Active preparation stage when the exception occurred.
        error: Internal exception raised by a service or repository.

    Returns:
        Safe structured execution error for persistence and worker logging.
    """
    # Chunking has corpus, parse, tokenizer, and artifact-persistence failure modes.
    if stage == "chunking":
        if isinstance(error, EmptyChunkingCorpusError):
            return PreparedIndexExecutionError(
                prepared_index_id,
                stage,
                "empty_corpus",
                "The selected corpus does not contain any documents to chunk.",
            )

        if isinstance(error, MissingParseArtifactError):
            return PreparedIndexExecutionError(
                prepared_index_id,
                stage,
                "missing_parse_artifact",
                "Canonical parsing is incomplete for the selected corpus.",
                {"document_ids": error.document_ids},
            )

        if isinstance(error, ChunkingCorpusNotFoundError):
            return PreparedIndexExecutionError(
                prepared_index_id,
                stage,
                "corpus_not_found",
                "The selected corpus does not exist.",
            )

        if isinstance(error, TokenizerAssetError):
            return PreparedIndexExecutionError(
                prepared_index_id,
                stage,
                "chunking_tokenizer_unavailable",
                "The configured chunking tokenizer is unavailable.",
            )

        if isinstance(
            error,
            (sqlite3.Error, InvalidPreparedIndexStateError, PreparedIndexNotFoundError),
        ):
            return PreparedIndexExecutionError(
                prepared_index_id,
                stage,
                "persistence_error",
                "The index preparation state could not be saved.",
            )

        return PreparedIndexExecutionError(
            prepared_index_id,
            stage,
            "chunking_failed",
            "The selected corpus could not be chunked.",
        )

    # Preserve actionable remote-provider categories without exposing raw responses.
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

    # Return the first specific provider error before broader base classes.
    for error_type, code, message in provider_errors:
        if isinstance(error, error_type):
            return PreparedIndexExecutionError(
                prepared_index_id,
                stage,
                code,
                message,
            )

    # Separate unusable upstream data and external vector-store failures.
    if isinstance(error, EmptyChunkSetError):
        return PreparedIndexExecutionError(
            prepared_index_id,
            stage,
            "empty_chunk_set",
            "The chunk artifact does not contain any text to embed.",
        )

    if isinstance(error, VectorStoreError):
        return PreparedIndexExecutionError(
            prepared_index_id,
            stage,
            "vector_store_unavailable",
            "The vector index could not be stored.",
        )

    if isinstance(
        error,
        (sqlite3.Error, InvalidPreparedIndexStateError, PreparedIndexNotFoundError),
    ):
        return PreparedIndexExecutionError(
            prepared_index_id,
            stage,
            "vector_index_persistence_failed",
            "The vector-index preparation state could not be saved.",
        )

    # Hide all remaining provider and implementation details.
    if isinstance(error, EmbeddingProviderError):
        return PreparedIndexExecutionError(
            prepared_index_id,
            stage,
            "embedding_failed",
            "The selected chunks could not be embedded.",
        )

    return PreparedIndexExecutionError(
        prepared_index_id,
        stage,
        "embedding_failed",
        "The selected chunks could not be embedded.",
    )
