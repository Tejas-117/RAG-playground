"""Shared immutable contracts for embedding providers and vector stores."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class EmbeddingInputPurpose(str, Enum):
    """Identify how a model should interpret an embedding input."""

    DOCUMENT = "document"
    QUERY = "query"


@dataclass(frozen=True)
class EmbeddingBatch:
    """Return validated vectors and provider-reported model provenance.

    Attributes:
        vectors: Ordered vectors corresponding exactly to the submitted texts.
        dimensions: Shared positive vector width.
        provider_model: Provider-reported model identifier when available.
        provider_revision: Provider-reported immutable revision when available.
    """

    vectors: tuple[tuple[float, ...], ...]
    dimensions: int
    provider_model: str | None = None
    provider_revision: str | None = None


class EmbeddingProviderError(RuntimeError):
    """Base class for safe provider-neutral embedding failures."""


class EmbeddingProviderUnavailableError(EmbeddingProviderError):
    """Report that the configured provider cannot currently be reached."""


class EmbeddingRequestTimeoutError(EmbeddingProviderError):
    """Report that the provider did not finish within the request timeout."""


class EmbeddingAuthenticationError(EmbeddingProviderError):
    """Report that the provider rejected backend authentication."""


class EmbeddingRateLimitError(EmbeddingProviderError):
    """Report that the provider temporarily rejected request volume."""


class EmbeddingRequestRejectedError(EmbeddingProviderError):
    """Report a provider-rejected model or request without guessing its cause."""


class EmbeddingInputTooLargeError(EmbeddingRequestRejectedError):
    """Report that a provider refused an input exceeding its model limit."""


class InvalidEmbeddingResponseError(EmbeddingProviderError):
    """Report a provider response that cannot represent the requested batch."""


class VectorStoreError(RuntimeError):
    """Hide vector-store-specific exceptions behind one stable boundary."""


class InvalidVectorSearchResponseError(VectorStoreError):
    """Report a vector-search response that cannot represent ranked hits."""


@dataclass(frozen=True)
class VectorSearchHit:
    """Represent one vector-store match without provider-specific response data.

    Attributes:
        chunk_id: Stable application chunk identifier stored with the vector.
        raw_distance: Unmodified distance returned by the configured index space.
    """

    chunk_id: str
    raw_distance: float


class EmbeddingProvider(Protocol):
    """Define the behavior required from every embedding provider adapter."""

    identifier: str
    version: str

    def input_policy_version(self, model: str) -> str:
        """Return the versioned text-transformation policy for one model.

        Args:
            model: Provider model identifier selected by the run.

        Returns:
            Stable policy identity included in vector-index fingerprints.
        """
        ...

    def embed(
        self,
        model: str,
        texts: list[str],
        purpose: EmbeddingInputPurpose,
    ) -> EmbeddingBatch:
        """Embed ordered texts for a declared document or query purpose.

        Args:
            model: Provider model identifier selected by the run.
            texts: Non-empty ordered text inputs.
            purpose: Semantic input role used by model-specific policies.

        Returns:
            Validated vectors in the same order as the submitted texts.
        """
        ...


class VectorStore(Protocol):
    """Define provider-neutral vector-index storage and search operations."""

    identifier: str
    version: str

    def create_collection(self, name: str, distance_metric: str) -> None:
        """Create an empty collection using the requested distance space.

        Args:
            name: Unique provider-safe collection name.
            distance_metric: Application-level distance metric identifier.

        Returns:
            None. A new empty collection is persisted by the adapter.
        """
        ...

    def add(
        self,
        collection_name: str,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict[str, str | int | float | bool]],
    ) -> None:
        """Add one aligned batch of vector records to a collection.

        Args:
            collection_name: Existing collection receiving the records.
            ids: Stable chunk identifiers.
            vectors: Dense vectors aligned with the identifiers.
            metadata: Scalar provenance aligned with the identifiers.

        Returns:
            None. The batch is persisted by the adapter.
        """
        ...

    def count(self, collection_name: str) -> int:
        """Return the number of records stored in one collection.

        Args:
            collection_name: Existing collection to inspect.

        Returns:
            Non-negative stored record count.
        """
        ...

    def query(
        self,
        collection_name: str,
        vector: list[float],
        top_k: int,
    ) -> tuple[VectorSearchHit, ...]:
        """Search one collection with an explicit query vector.

        Args:
            collection_name: Existing collection containing indexed chunk vectors.
            vector: Explicit provider-generated query vector.
            top_k: Maximum number of nearest chunks to return.

        Returns:
            Ranked immutable hits carrying stable chunk IDs and raw distances.
        """
        ...

    def delete_collection(self, name: str) -> None:
        """Delete one exact collection during rollback or race cleanup.

        Args:
            name: Exact collection name owned by the current build attempt.

        Returns:
            None. Missing collections may be treated as already deleted.
        """
        ...
