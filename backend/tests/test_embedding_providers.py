"""Offline tests for embedding HTTP and vector-store adapter boundaries."""

import json

import httpx
import pytest

from backend.embedding.models import (
    EmbeddingInputPurpose,
    EmbeddingProviderUnavailableError,
    EmbeddingRequestRejectedError,
    InvalidEmbeddingResponseError,
)
from backend.embedding.providers import OllamaHttpEmbeddingProvider
from backend.embedding.vector_store import ChromaVectorStore


def test_ollama_adapter_posts_prefixed_batch_without_truncation() -> None:
    """Verify the adapter sends one explicit HTTP request with Nomic policy.

    Args:
        None.

    Returns:
        None. Assertions verify request shape and validated response provenance.
    """

    def handle_request(request: httpx.Request) -> httpx.Response:
        """Validate the outbound request and return deterministic vectors.

        Args:
            request: HTTPX request created by the adapter.

        Returns:
            Successful fake Ollama response.
        """
        payload = json.loads(request.content)

        # The backend must use Ollama's API without provider-side silent truncation.
        assert request.url.path == "/api/embed"
        assert payload == {
            "model": "nomic-embed-text",
            "input": ["search_document: alpha", "search_document: beta"],
            "truncate": False,
        }
        return httpx.Response(
            200,
            json={
                "model": "nomic-embed-text:latest",
                "embeddings": [[1, 2, 3], [4, 5, 6]],
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handle_request),
        base_url="http://ollama.test",
    )
    provider = OllamaHttpEmbeddingProvider(client=client)
    batch = provider.embed(
        "nomic-embed-text",
        ["alpha", "beta"],
        EmbeddingInputPurpose.DOCUMENT,
    )

    # Response vectors remain aligned and carry provider-reported model identity.
    assert batch.vectors == ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    assert batch.dimensions == 3
    assert batch.provider_model == "nomic-embed-text:latest"


def test_ollama_adapter_maps_connection_failure() -> None:
    """Verify an unreachable HTTP service becomes a provider-neutral failure.

    Args:
        None.

    Returns:
        None. Assertions verify graceful transport error mapping.
    """

    def handle_request(request: httpx.Request) -> httpx.Response:
        """Raise the same connection error produced by an unavailable service.

        Args:
            request: HTTP request that cannot reach its destination.

        Returns:
            Never returns because the transport is unavailable.

        Raises:
            httpx.ConnectError: Always.
        """
        # Attach the request so HTTPX preserves normal transport error context.
        raise httpx.ConnectError("connection refused", request=request)

    client = httpx.Client(
        transport=httpx.MockTransport(handle_request),
        base_url="http://ollama.test",
    )
    provider = OllamaHttpEmbeddingProvider(client=client)

    # Provider-neutral callers do not need to understand HTTPX exceptions.
    with pytest.raises(EmbeddingProviderUnavailableError):
        provider.embed("model", ["alpha"], EmbeddingInputPurpose.DOCUMENT)


def test_ollama_adapter_maps_rejected_model_without_guessing_install_state() -> None:
    """Verify provider rejection does not trigger model management behavior.

    Args:
        None.

    Returns:
        None. Assertions verify a stable rejected-request category.
    """

    def handle_request(request: httpx.Request) -> httpx.Response:
        """Return a model rejection from the fake provider service.

        Args:
            request: HTTP request containing the unknown provider model.

        Returns:
            Provider-owned 404 response.
        """
        # The response intentionally does not imply whether installation is possible.
        return httpx.Response(404, json={"error": "model not found"})

    client = httpx.Client(
        transport=httpx.MockTransport(handle_request),
        base_url="http://ollama.test",
    )
    provider = OllamaHttpEmbeddingProvider(client=client)

    # The backend reports rejection and never invokes an Ollama CLI or pull endpoint.
    with pytest.raises(EmbeddingRequestRejectedError):
        provider.embed("missing", ["alpha"], EmbeddingInputPurpose.DOCUMENT)


def test_ollama_adapter_rejects_inconsistent_vector_dimensions() -> None:
    """Verify malformed provider vectors never reach the vector store.

    Args:
        None.

    Returns:
        None. Assertions verify strict batch response validation.
    """

    def handle_request(request: httpx.Request) -> httpx.Response:
        """Return vectors that do not share one embedding width.

        Args:
            request: Valid adapter request ignored by the fake response.

        Returns:
            Invalid but successful-looking provider response.
        """
        # Mixed dimensions would make one vector collection incompatible.
        return httpx.Response(200, json={"embeddings": [[1, 2], [3, 4, 5]]})

    client = httpx.Client(
        transport=httpx.MockTransport(handle_request),
        base_url="http://ollama.test",
    )
    provider = OllamaHttpEmbeddingProvider(client=client)

    # Validate the entire batch before the indexing service can consume it.
    with pytest.raises(InvalidEmbeddingResponseError):
        provider.embed("model", ["alpha", "beta"], EmbeddingInputPurpose.DOCUMENT)


def test_chroma_reset_deletes_only_managed_collections() -> None:
    """Verify dev cleanup cannot delete unrelated Chroma collections.

    Args:
        None.

    Returns:
        None. Assertions verify exact application-prefix ownership filtering.
    """

    class FakeChromaClient:
        """Expose only collection listing and deletion used by reset."""

        def __init__(self) -> None:
            """Create mixed managed and unrelated collection names.

            Args:
                None.

            Returns:
                None. Collection state is retained for assertions.
            """
            self.collections = ["rag_idx_first", "unrelated", "rag_idx_second"]
            self.deleted: list[str] = []

        def list_collections(self) -> list[str]:
            """Return every fake collection name.

            Args:
                None.

            Returns:
                Copy of stored collection names.
            """
            # A copy prevents deletion records from changing iteration.
            return list(self.collections)

        def delete_collection(self, name: str) -> None:
            """Record one exact deleted collection.

            Args:
                name: Collection selected by ownership filtering.

            Returns:
                None. The deleted name is appended for assertions.
            """
            self.deleted.append(name)

    client = FakeChromaClient()
    store = ChromaVectorStore(client=client)
    deleted_count = store.delete_managed_collections()

    # Only application-generated collection prefixes qualify for destructive cleanup.
    assert deleted_count == 2
    assert client.deleted == ["rag_idx_first", "rag_idx_second"]
