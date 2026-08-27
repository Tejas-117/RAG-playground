"""HTTP adapters for backend-configured embedding providers."""

import math
import os
from typing import Any

import httpx

from backend.embedding.models import (
    EmbeddingAuthenticationError,
    EmbeddingBatch,
    EmbeddingInputPurpose,
    EmbeddingInputTooLargeError,
    EmbeddingProvider,
    EmbeddingProviderUnavailableError,
    EmbeddingRateLimitError,
    EmbeddingRequestRejectedError,
    EmbeddingRequestTimeoutError,
    InvalidEmbeddingResponseError,
)

# The local Ollama HTTP service uses this origin unless deployment overrides it.
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# Bound a single provider batch without constraining the complete background run.
OLLAMA_REQUEST_TIMEOUT_SECONDS = 120.0

# Prefix policy changes alter vectors and therefore require a versioned identity.
_RAW_INPUT_POLICY_VERSION = "raw-v1"
_NOMIC_INPUT_POLICY_VERSION = "nomic-retrieval-prefix-v1"


class OllamaHttpEmbeddingProvider:
    """Generate embeddings through Ollama's HTTP API without managing Ollama."""

    identifier = "ollama-http"
    version = "1"

    def __init__(
        self,
        client: httpx.Client | None = None,
        base_url: str | None = None,
    ) -> None:
        """Configure an injectable synchronous Ollama HTTP client.

        Args:
            client: Optional preconfigured client used by deterministic tests.
            base_url: Optional provider origin overriding ``OLLAMA_BASE_URL``.

        Returns:
            None. The adapter retains no model installation or process state.
        """
        resolved_base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL
        ).rstrip("/")

        # Own a bounded client only when dependency injection did not provide one.
        self._client = client or httpx.Client(
            base_url=resolved_base_url,
            timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS,
        )

    def input_policy_version(self, model: str) -> str:
        """Return the text policy used for a selected Ollama model.

        Args:
            model: Ollama model identifier selected by the run.

        Returns:
            Stable raw or Nomic retrieval-prefix policy version.
        """
        # Nomic Embed Text requires asymmetric retrieval task prefixes.
        if model.casefold().startswith("nomic-embed-text"):
            return _NOMIC_INPUT_POLICY_VERSION

        return _RAW_INPUT_POLICY_VERSION

    def embed(
        self,
        model: str,
        texts: list[str],
        purpose: EmbeddingInputPurpose,
    ) -> EmbeddingBatch:
        """Request one ordered embedding batch from Ollama over HTTP.

        Args:
            model: Ollama model identifier submitted to the API.
            texts: Non-empty ordered input strings.
            purpose: Document or query role used by model-specific prefixes.

        Returns:
            Validated vectors aligned with the submitted input order.

        Raises:
            EmbeddingProviderError: If transport, HTTP, or response validation fails.
        """
        # Empty batches indicate an embedding-service programming error.
        if not texts:
            raise InvalidEmbeddingResponseError(
                "The embedding request did not contain any texts."
            )

        prepared_texts = [self._prepare_text(model, text, purpose) for text in texts]

        try:
            # Disable provider truncation so no indexed vector silently loses content.
            response = self._client.post(
                "/api/embed",
                json={"model": model, "input": prepared_texts, "truncate": False},
            )
            response.raise_for_status()
        except httpx.TimeoutException as error:
            raise EmbeddingRequestTimeoutError(
                "The embedding provider request timed out."
            ) from error
        except httpx.ConnectError as error:
            raise EmbeddingProviderUnavailableError(
                "The embedding provider could not be reached."
            ) from error
        except httpx.HTTPStatusError as error:
            self._raise_http_error(error.response)

        try:
            payload = response.json()
        except ValueError as error:
            raise InvalidEmbeddingResponseError(
                "The embedding provider returned invalid JSON."
            ) from error

        return self._validate_response(payload, len(texts))

    def _prepare_text(
        self,
        model: str,
        text: str,
        purpose: EmbeddingInputPurpose,
    ) -> str:
        """Apply the model-specific retrieval prefix to one input.

        Args:
            model: Selected Ollama model identifier.
            text: Exact chunk or query text.
            purpose: Semantic document or query role.

        Returns:
            Provider-ready text without mutating the persisted canonical chunk.
        """
        # Only the Nomic family requires asymmetric task prefixes.
        if not model.casefold().startswith("nomic-embed-text"):
            return text

        # Keep query and document vectors in their intended Nomic retrieval roles.
        prefix = (
            "search_document: "
            if purpose is EmbeddingInputPurpose.DOCUMENT
            else "search_query: "
        )
        return f"{prefix}{text}"

    def _raise_http_error(self, response: httpx.Response) -> None:
        """Translate an Ollama HTTP response into a safe provider-neutral error.

        Args:
            response: Non-successful Ollama HTTP response.

        Returns:
            Never returns because a mapped exception is always raised.

        Raises:
            EmbeddingProviderError: Stable category based on status and safe text.
        """
        status_code = response.status_code

        # Authentication failures require backend provider configuration changes.
        if status_code in {401, 403}:
            raise EmbeddingAuthenticationError(
                "The embedding provider rejected backend authentication."
            )

        # Rate limiting is temporary and should remain distinguishable from rejection.
        if status_code == 429:
            raise EmbeddingRateLimitError(
                "The embedding provider rate limit was reached."
            )

        # Detect provider wording for an input rejected because it cannot fit.
        response_text = response.text.casefold()
        if status_code in {400, 413, 422} and any(
            marker in response_text
            for marker in ("context length", "too long", "input length")
        ):
            raise EmbeddingInputTooLargeError(
                "A chunk exceeds the selected embedding model's input limit."
            )

        # Provider server errors describe availability, not user configuration.
        if status_code >= 500:
            raise EmbeddingProviderUnavailableError(
                "The embedding provider could not complete the request."
            )

        # Do not guess whether a rejected model is missing, unauthorized, or disabled.
        raise EmbeddingRequestRejectedError(
            "The embedding provider rejected the model or request."
        )

    def _validate_response(
        self,
        payload: Any,
        expected_count: int,
    ) -> EmbeddingBatch:
        """Validate one untrusted Ollama embedding response.

        Args:
            payload: JSON-decoded response value.
            expected_count: Number of texts submitted in the request.

        Returns:
            Immutable validated embedding batch.

        Raises:
            InvalidEmbeddingResponseError: If vectors are absent or inconsistent.
        """
        # Require the documented object and embedding-array response shape.
        if not isinstance(payload, dict) or not isinstance(
            payload.get("embeddings"), list
        ):
            raise InvalidEmbeddingResponseError(
                "The embedding provider response did not contain vectors."
            )

        raw_vectors = payload["embeddings"]

        # Positional alignment requires exactly one vector for every submitted text.
        if len(raw_vectors) != expected_count:
            raise InvalidEmbeddingResponseError(
                "The embedding provider returned an unexpected vector count."
            )

        vectors: list[tuple[float, ...]] = []
        dimensions: int | None = None

        # Validate every scalar before any vector is handed to the vector store.
        for raw_vector in raw_vectors:
            if not isinstance(raw_vector, list) or not raw_vector:
                raise InvalidEmbeddingResponseError(
                    "The embedding provider returned an empty vector."
                )

            vector: list[float] = []

            # Reject booleans and non-finite numeric values as invalid coordinates.
            for value in raw_vector:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise InvalidEmbeddingResponseError(
                        "The embedding provider returned a non-numeric vector value."
                    )

                coordinate = float(value)
                if not math.isfinite(coordinate):
                    raise InvalidEmbeddingResponseError(
                        "The embedding provider returned a non-finite vector value."
                    )
                vector.append(coordinate)

            # Every vector in one model space must have the same width.
            if dimensions is None:
                dimensions = len(vector)
            elif len(vector) != dimensions:
                raise InvalidEmbeddingResponseError(
                    "The embedding provider returned inconsistent vector dimensions."
                )

            vectors.append(tuple(vector))

        provider_model = payload.get("model")

        # Optional provenance must still be a meaningful string when present.
        if provider_model is not None and not isinstance(provider_model, str):
            raise InvalidEmbeddingResponseError(
                "The embedding provider returned an invalid model identifier."
            )

        return EmbeddingBatch(
            vectors=tuple(vectors),
            dimensions=dimensions or 0,
            provider_model=provider_model,
            provider_revision=None,
        )


def get_embedding_provider(provider: str) -> EmbeddingProvider:
    """Resolve one backend-registered embedding provider adapter.

    Args:
        provider: Provider identifier from the validated pipeline configuration.

    Returns:
        Stateless provider adapter implementing the shared embedding protocol.

    Raises:
        EmbeddingRequestRejectedError: If no adapter is registered.
    """
    # Keep provider selection centralized so future adapters do not alter the service.
    if provider == "ollama":
        return OllamaHttpEmbeddingProvider()

    raise EmbeddingRequestRejectedError(
        "The selected embedding provider is not registered."
    )
