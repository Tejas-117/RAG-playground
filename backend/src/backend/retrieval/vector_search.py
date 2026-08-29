"""Embed one question and search an exact compatible vector index."""

from typing import Any

from backend.embedding.models import (
    EmbeddingInputPurpose,
    EmbeddingProvider,
    InvalidEmbeddingResponseError,
    VectorSearchHit,
    VectorStore,
)
from backend.embedding.providers import get_embedding_provider
from backend.embedding.vector_store import get_vector_store
from backend.pipeline.configs import EmbeddingConfig


class InvalidVectorSearchRequestError(ValueError):
    """Report a question or result limit that cannot form a vector search."""


class IncompatibleVectorIndexError(ValueError):
    """Report a ready index that does not match the requested embedding space."""


def search_vector_index(
    question: str,
    top_k: int,
    config: EmbeddingConfig,
    vector_index: dict[str, Any],
    provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
) -> tuple[VectorSearchHit, ...]:
    """Embed one question and search its exact compatible vector collection.

    Args:
        question: User question requiring nearest indexed chunks.
        top_k: Maximum number of nearest chunks requested.
        config: Embedding-space configuration captured by the run.
        vector_index: Ready reusable index artifact selected by the pipeline.
        provider: Optional embedding adapter override for deterministic tests.
        vector_store: Optional vector-store override for deterministic tests.

    Returns:
        Ranked chunk identifiers with unmodified vector-store distances.

    Raises:
        InvalidVectorSearchRequestError: If the question or limit is invalid.
        IncompatibleVectorIndexError: If the index and query space do not match.
        EmbeddingProviderError: If query embedding fails.
        VectorStoreError: If vector search fails or returns malformed results.
    """
    normalized_question = question.strip()

    # A blank query cannot produce a meaningful retrieval vector.
    if not normalized_question:
        raise InvalidVectorSearchRequestError(
            "The vector search question must not be blank."
        )

    # Reject booleans explicitly because Python treats them as integers.
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise InvalidVectorSearchRequestError(
            "The vector search limit must be positive."
        )

    resolved_provider = provider or get_embedding_provider(config.provider)
    resolved_store = vector_store or get_vector_store()
    _validate_index_compatibility(
        vector_index,
        config,
        resolved_provider,
        resolved_store,
    )

    # Query-purpose embedding applies model-specific search-query preparation.
    query_batch = resolved_provider.embed(
        config.model,
        [normalized_question],
        EmbeddingInputPurpose.QUERY,
    )

    # Even injected providers must preserve the one-question/one-vector contract.
    if len(query_batch.vectors) != 1:
        raise InvalidEmbeddingResponseError(
            "The embedding provider returned an unexpected query vector count."
        )

    # Query and document vectors must have the same width to share an index space.
    if query_batch.dimensions != vector_index["dimensions"]:
        raise IncompatibleVectorIndexError(
            "The query embedding dimensions do not match the vector index."
        )

    _validate_query_provenance(
        vector_index,
        query_batch.provider_model,
        query_batch.provider_revision,
    )

    # The persisted vector count safely bounds the nearest-neighbor request.
    effective_top_k = min(top_k, vector_index["vector_count"])
    return resolved_store.query(
        vector_index["collection_name"],
        list(query_batch.vectors[0]),
        effective_top_k,
    )


def _validate_index_compatibility(
    vector_index: dict[str, Any],
    config: EmbeddingConfig,
    provider: EmbeddingProvider,
    vector_store: VectorStore,
) -> None:
    """Confirm a vector-index artifact matches every current search dependency.

    Args:
        vector_index: Ready artifact whose collection will be searched.
        config: Requested embedding provider, model, and metric.
        provider: Resolved query embedding adapter.
        vector_store: Resolved collection adapter.

    Returns:
        None when the index is safe to query.

    Raises:
        IncompatibleVectorIndexError: If required identity or shape differs.
    """
    required_fields = {
        "status",
        "provider",
        "model",
        "distance_metric",
        "input_policy_version",
        "indexer_name",
        "indexer_version",
        "collection_name",
        "dimensions",
        "vector_count",
    }

    # Reject incomplete dictionaries before reading compatibility fields directly.
    if not required_fields.issubset(vector_index):
        raise IncompatibleVectorIndexError("The vector index artifact is incomplete.")

    # Retrieval may consume only a completely materialized reusable index.
    if vector_index["status"] != "ready":
        raise IncompatibleVectorIndexError("The vector index is not ready.")

    expected_identity = (
        config.provider,
        config.model,
        config.distance_metric.value,
        provider.input_policy_version(config.model),
        vector_store.identifier,
        vector_store.version,
    )
    actual_identity = (
        vector_index["provider"],
        vector_index["model"],
        vector_index["distance_metric"],
        vector_index["input_policy_version"],
        vector_index["indexer_name"],
        vector_index["indexer_version"],
    )

    # Identity mismatches indicate a different embedding or index implementation.
    if actual_identity != expected_identity:
        raise IncompatibleVectorIndexError(
            "The vector index is incompatible with the search configuration."
        )

    dimensions = vector_index["dimensions"]
    vector_count = vector_index["vector_count"]
    collection_name = vector_index["collection_name"]

    # Persisted dimensions and counts must describe a non-empty searchable index.
    if (
        isinstance(dimensions, bool)
        or not isinstance(dimensions, int)
        or dimensions <= 0
        or isinstance(vector_count, bool)
        or not isinstance(vector_count, int)
        or vector_count <= 0
    ):
        raise IncompatibleVectorIndexError(
            "The vector index has invalid dimensions or vector count."
        )

    # An empty or non-string collection reference cannot identify a Chroma index.
    if not isinstance(collection_name, str) or not collection_name.strip():
        raise IncompatibleVectorIndexError(
            "The vector index has an invalid collection reference."
        )


def _validate_query_provenance(
    vector_index: dict[str, Any],
    query_provider_model: str | None,
    query_provider_revision: str | None,
) -> None:
    """Compare optional provider model provenance for documents and query.

    Args:
        vector_index: Ready index containing document-vector provenance.
        query_provider_model: Model identity reported for the query vector.
        query_provider_revision: Immutable model revision reported for the query.

    Returns:
        None when either value is unavailable or both values match.

    Raises:
        IncompatibleVectorIndexError: If both model identities exist and differ.
    """
    index_provider_model = vector_index.get("provider_model")
    index_provider_revision = vector_index.get("provider_revision")

    # Compare provider model identity only when both calls expose it.
    if (
        index_provider_model is not None
        and query_provider_model is not None
        and query_provider_model != index_provider_model
    ):
        raise IncompatibleVectorIndexError(
            "The query embedding model does not match the vector index."
        )

    # Compare immutable revisions independently when both calls expose them.
    if (
        index_provider_revision is not None
        and query_provider_revision is not None
        and query_provider_revision != index_provider_revision
    ):
        raise IncompatibleVectorIndexError(
            "The query embedding revision does not match the vector index."
        )
