"""Shared query-time contracts and safe error mapping for benchmark execution."""

import sqlite3
from collections.abc import Callable
from time import perf_counter
from typing import Any

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
from backend.generation.models import (
    GenerationAuthenticationError,
    GenerationInputTooLargeError,
    GenerationProvider,
    GenerationProviderError,
    GenerationProviderUnavailableError,
    GenerationRateLimitError,
    GenerationRequestRejectedError,
    GenerationRequestTimeoutError,
    GenerationServiceResult,
    InvalidGenerationResponseError,
)
from backend.ingestion.chunkers.models import ChunkingTokenizer
from backend.ingestion.chunkers.tokenizer import TokenizerAssetError
from backend.pipeline.configs import EmbeddingConfig, GenerationConfig, RetrievalConfig
from backend.retrieval.chunk_hydration import ChunkHydrationError
from backend.retrieval.models import HydratedVectorSearchHit
from backend.retrieval.service import InvalidRetrievalArtifactError
from backend.retrieval.vector_search import (
    IncompatibleVectorIndexError,
    InvalidVectorSearchRequestError,
)

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

# An answer generator consumes the question, exact ranked context, and run settings.
AnswerGenerator = Callable[
    [
        str,
        GenerationConfig,
        tuple[HydratedVectorSearchHit, ...],
        GenerationProvider | None,
        ChunkingTokenizer | None,
    ],
    GenerationServiceResult,
]


class QueryExecutionError(RuntimeError):
    """Expose one safe benchmark failure to worker and transport boundaries."""

    def __init__(
        self,
        run_id: str,
        stage: str,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Store persisted run identity and a structured public failure.

        Args:
            run_id: Stable identifier of the failed benchmark run.
            stage: Query-time stage that encountered the failure.
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


def elapsed_milliseconds(started_counter: float) -> int:
    """Calculate elapsed milliseconds from a monotonic start value.

    Args:
        started_counter: Value previously returned by ``perf_counter``.

    Returns:
        Rounded, non-negative elapsed milliseconds.
    """
    # Clamp defensively even though a monotonic clock should not move backward.
    return max(0, round((perf_counter() - started_counter) * 1000))


def map_query_execution_error(
    run_id: str,
    stage: str,
    error: Exception,
) -> QueryExecutionError:
    """Translate an internal query-time failure into a stable public category.

    Args:
        run_id: Stable benchmark identifier associated with the failure.
        stage: Retrieval or generation stage active when the failure occurred.
        error: Internal exception raised by a service, provider, or repository.

    Returns:
        Safe structured error suitable for benchmark persistence.
    """
    # Retrieval and generation expose different actionable failure categories.
    if stage == "retrieval":
        return _map_retrieval_error(run_id, error)

    return _map_generation_error(run_id, error)


def _map_retrieval_error(run_id: str, error: Exception) -> QueryExecutionError:
    """Translate query embedding, search, hydration, or persistence failures.

    Args:
        run_id: Stable identifier of the failed benchmark.
        error: Exception raised while retrieving or saving ranked chunks.

    Returns:
        Safe retrieval-stage execution error.
    """
    # Preserve common provider failures so clients can distinguish corrective action.
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

    # Return the first matching provider category without exposing adapter details.
    for error_type, code, message in provider_errors:
        if isinstance(error, error_type):
            return QueryExecutionError(run_id, "retrieval", code, message)

    # Invalid requests indicate corrupt or incompatible persisted configuration.
    if isinstance(error, InvalidVectorSearchRequestError):
        return QueryExecutionError(
            run_id,
            "retrieval",
            "invalid_retrieval_request",
            "The saved question or retrieval limit is invalid.",
        )

    # Compatibility failures prevent querying a stale or unrelated vector space.
    if isinstance(error, (InvalidRetrievalArtifactError, IncompatibleVectorIndexError)):
        return QueryExecutionError(
            run_id,
            "retrieval",
            "incompatible_vector_index",
            "The vector index is incompatible with this retrieval request.",
        )

    # Every vector hit must resolve back to exact application-owned chunk data.
    if isinstance(error, ChunkHydrationError):
        return QueryExecutionError(
            run_id,
            "retrieval",
            "retrieval_chunk_hydration_failed",
            "The retrieved chunks could not be loaded safely.",
        )

    # Vector-store failures remain distinct from query-embedding failures.
    if isinstance(error, VectorStoreError):
        return QueryExecutionError(
            run_id,
            "retrieval",
            "retrieval_vector_store_unavailable",
            "The vector index could not be searched.",
        )

    # Relational failures must not leak SQLite constraints or internal identifiers.
    if isinstance(error, sqlite3.Error):
        return QueryExecutionError(
            run_id,
            "retrieval",
            "retrieval_persistence_failed",
            "The retrieval result could not be saved.",
        )

    # Hide all other embedding-provider and implementation details.
    if isinstance(error, EmbeddingProviderError):
        return QueryExecutionError(
            run_id,
            "retrieval",
            "retrieval_embedding_failed",
            "The retrieval question could not be embedded.",
        )

    return QueryExecutionError(
        run_id,
        "retrieval",
        "retrieval_failed",
        "The retrieval stage could not be completed.",
    )


def _map_generation_error(run_id: str, error: Exception) -> QueryExecutionError:
    """Translate prompt, provider, or generation-persistence failures safely.

    Args:
        run_id: Stable identifier of the failed benchmark.
        error: Exception raised while constructing or generating the answer.

    Returns:
        Safe generation-stage execution error suitable for persistence.
    """
    # Preserve provider categories needed for useful client-side error messages.
    provider_errors: tuple[tuple[type[Exception], str, str], ...] = (
        (
            GenerationProviderUnavailableError,
            "generation_provider_unavailable",
            "The generation provider could not complete the request.",
        ),
        (
            GenerationRequestTimeoutError,
            "generation_request_timeout",
            "The generation provider request timed out.",
        ),
        (
            GenerationAuthenticationError,
            "generation_authentication_failed",
            "The generation provider credentials are missing or were rejected.",
        ),
        (
            GenerationRateLimitError,
            "generation_rate_limited",
            "The generation provider rate limit was reached.",
        ),
        (
            GenerationInputTooLargeError,
            "generation_input_too_large",
            "The question and retrieved context exceed the model input limit.",
        ),
        (
            GenerationRequestRejectedError,
            "generation_request_rejected",
            "The generation provider rejected the model or request.",
        ),
        (
            InvalidGenerationResponseError,
            "invalid_generation_response",
            "The generation provider returned an invalid answer response.",
        ),
    )

    # Return the first matching provider category without exposing SDK details.
    for error_type, code, message in provider_errors:
        if isinstance(error, error_type):
            return QueryExecutionError(run_id, "generation", code, message)

    # Prompt budgeting uses the same pinned tokenizer identity as chunking.
    if isinstance(error, TokenizerAssetError):
        return QueryExecutionError(
            run_id,
            "generation",
            "generation_tokenizer_unavailable",
            "The configured generation budgeting tokenizer is unavailable.",
        )

    # Relational failures preserve retrieval while rejecting partial answer rows.
    if isinstance(error, sqlite3.Error):
        return QueryExecutionError(
            run_id,
            "generation",
            "generation_persistence_failed",
            "The generated answer could not be saved.",
        )

    # A catalog miss indicates a provider/model not executable at runtime.
    if isinstance(error, LookupError):
        return QueryExecutionError(
            run_id,
            "generation",
            "generation_request_rejected",
            "The selected generation model is not registered.",
        )

    # Hide any other adapter implementation details from persisted failures.
    if isinstance(error, GenerationProviderError):
        return QueryExecutionError(
            run_id,
            "generation",
            "generation_failed",
            "The answer could not be generated.",
        )

    return QueryExecutionError(
        run_id,
        "generation",
        "generation_failed",
        "The generation stage could not be completed.",
    )
