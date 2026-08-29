"""Offline tests for embedding HTTP and vector-store adapter boundaries."""

import json

import httpx
import pytest

from backend.embedding.models import (
    EmbeddingInputPurpose,
    EmbeddingProviderUnavailableError,
    EmbeddingRequestRejectedError,
    InvalidEmbeddingResponseError,
    InvalidVectorSearchResponseError,
    VectorStoreError,
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


def test_ollama_adapter_uses_nomic_query_prefix() -> None:
    """Verify query embeddings use Nomic's asymmetric retrieval prefix.

    Args:
        None.

    Returns:
        None. Assertions verify query-purpose request preparation.
    """

    def handle_request(request: httpx.Request) -> httpx.Response:
        """Validate the query-prefixed request and return one vector.

        Args:
            request: HTTPX request created by the adapter.

        Returns:
            Successful deterministic Ollama response.
        """
        payload = json.loads(request.content)

        # Query text must occupy the complementary Nomic retrieval vector role.
        assert payload["input"] == ["search_query: refund policy"]
        return httpx.Response(200, json={"embeddings": [[1, 2, 3]]})

    client = httpx.Client(
        transport=httpx.MockTransport(handle_request),
        base_url="http://ollama.test",
    )
    provider = OllamaHttpEmbeddingProvider(client=client)

    provider.embed(
        "nomic-embed-text",
        ["refund policy"],
        EmbeddingInputPurpose.QUERY,
    )


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


def test_chroma_query_returns_deterministic_raw_distance_hits() -> None:
    """Verify explicit queries are clamped, validated, and deterministically ranked.

    Args:
        None.

    Returns:
        None. Assertions verify Chroma request shape and provider-neutral hits.
    """

    class FakeCollection:
        """Expose the count and query operations required by vector search."""

        def __init__(self) -> None:
            """Create query-call storage for later assertions.

            Args:
                None.

            Returns:
                None. No query has been recorded initially.
            """
            self.query_arguments: dict[str, object] | None = None

        def count(self) -> int:
            """Return the number of vectors available to the fake query.

            Args:
                None.

            Returns:
                Three available vector records.
            """
            return 3

        def query(self, **arguments: object) -> dict[str, object]:
            """Record query arguments and return tied out-of-order distances.

            Args:
                arguments: Keyword arguments passed to Chroma's query operation.

            Returns:
                One nested list of IDs aligned with raw distances.
            """
            self.query_arguments = arguments
            return {
                "ids": [["chunk-b", "chunk-c", "chunk-a"]],
                "distances": [[0.25, 0.75, 0.25]],
            }

    class FakeChromaClient:
        """Return one injected collection for exact-name lookup."""

        def __init__(self, collection: FakeCollection) -> None:
            """Retain the collection used by the search adapter.

            Args:
                collection: Fake collection returned by lookup.

            Returns:
                None. The collection is retained for later calls.
            """
            self.collection = collection

        def get_collection(
            self,
            name: str,
            embedding_function: object,
        ) -> FakeCollection:
            """Validate lookup arguments and return the fake collection.

            Args:
                name: Exact collection name selected for retrieval.
                embedding_function: Must remain disabled for explicit vectors.

            Returns:
                Injected fake collection.
            """
            # Retrieval must target the exact persisted collection without embedding.
            assert name == "rag_idx_test"
            assert embedding_function is None
            return self.collection

    collection = FakeCollection()
    store = ChromaVectorStore(client=FakeChromaClient(collection))
    hits = store.query("rag_idx_test", [1.0, 2.0, 3.0], 10)

    # The collection count clamps top_k while equal distances sort by chunk ID.
    assert collection.query_arguments == {
        "query_embeddings": [[1.0, 2.0, 3.0]],
        "n_results": 3,
        "include": ["distances"],
    }
    assert [(hit.chunk_id, hit.raw_distance) for hit in hits] == [
        ("chunk-a", 0.25),
        ("chunk-b", 0.25),
        ("chunk-c", 0.75),
    ]


@pytest.mark.parametrize(
    "query_result",
    [
        None,
        {"ids": [["chunk-a"]], "distances": []},
        {"ids": [["chunk-a"]], "distances": [[0.1, 0.2]]},
        {"ids": [["chunk-a", "chunk-a"]], "distances": [[0.1, 0.2]]},
        {"ids": [[""]], "distances": [[0.1]]},
        {"ids": [["chunk-a"]], "distances": [[True]]},
        {"ids": [["chunk-a"]], "distances": [[float("nan")]]},
    ],
)
def test_chroma_query_rejects_malformed_results(query_result: object) -> None:
    """Verify malformed Chroma results never cross the vector-store boundary.

    Args:
        query_result: Invalid response returned by the fake collection.

    Returns:
        None. Assertions verify safe response validation.
    """

    class FakeCollection:
        """Return one injected malformed query response."""

        def count(self) -> int:
            """Return the expected number of malformed result entries.

            Args:
                None.

            Returns:
                Two only for the duplicate-ID case, otherwise one.
            """
            # Match expected length so each fixture reaches its intended validation.
            if isinstance(query_result, dict):
                raw_ids = query_result.get("ids")
                if (
                    isinstance(raw_ids, list)
                    and raw_ids
                    and isinstance(raw_ids[0], list)
                ):
                    return max(len(raw_ids[0]), 1)

            return 1

        def query(self, **arguments: object) -> object:
            """Return the malformed response without transformation.

            Args:
                arguments: Chroma query arguments unused by this fake.

            Returns:
                Injected malformed query response.
            """
            return query_result

    class FakeChromaClient:
        """Expose the malformed fake collection through normal lookup."""

        def get_collection(
            self,
            name: str,
            embedding_function: object,
        ) -> FakeCollection:
            """Return a new malformed-response collection.

            Args:
                name: Collection name unused by the fake.
                embedding_function: Explicit-vector marker unused by the fake.

            Returns:
                Collection returning the injected invalid response.
            """
            return FakeCollection()

    store = ChromaVectorStore(client=FakeChromaClient())

    # Response-shape failures remain distinct from Chroma transport failures.
    with pytest.raises(InvalidVectorSearchResponseError):
        store.query("rag_idx_test", [1.0], 2)


def test_chroma_query_returns_no_hits_for_empty_collection() -> None:
    """Verify an empty collection is handled without an invalid Chroma query.

    Args:
        None.

    Returns:
        None. Assertions verify the adapter returns an empty immutable result.
    """

    class FakeCollection:
        """Represent an empty collection that must never receive a query."""

        def count(self) -> int:
            """Report that the collection contains no vectors.

            Args:
                None.

            Returns:
                Zero stored vector records.
            """
            return 0

        def query(self, **arguments: object) -> object:
            """Reject any attempted nearest-neighbor query.

            Args:
                arguments: Unexpected query arguments.

            Returns:
                Never returns because empty collections must bypass queries.

            Raises:
                AssertionError: Always, if the adapter incorrectly calls query.
            """
            raise AssertionError("query must not be called for an empty collection")

    class FakeChromaClient:
        """Expose one empty fake collection."""

        def get_collection(
            self,
            name: str,
            embedding_function: object,
        ) -> FakeCollection:
            """Return the empty collection.

            Args:
                name: Collection name unused by the fake.
                embedding_function: Explicit-vector marker unused by the fake.

            Returns:
                Empty fake collection.
            """
            return FakeCollection()

    store = ChromaVectorStore(client=FakeChromaClient())

    assert store.query("rag_idx_empty", [1.0], 10) == ()


def test_chroma_query_maps_collection_failure() -> None:
    """Verify Chroma query failures become provider-neutral store failures.

    Args:
        None.

    Returns:
        None. Assertions verify implementation exceptions remain hidden.
    """

    class FakeChromaClient:
        """Fail collection lookup before nearest-neighbor search begins."""

        def get_collection(
            self,
            name: str,
            embedding_function: object,
        ) -> object:
            """Raise an implementation-specific collection failure.

            Args:
                name: Collection name that cannot be opened.
                embedding_function: Explicit-vector marker unused by the fake.

            Returns:
                Never returns because collection lookup fails.

            Raises:
                RuntimeError: Always.
            """
            raise RuntimeError("simulated Chroma failure")

    store = ChromaVectorStore(client=FakeChromaClient())

    # Callers receive only the stable vector-store boundary exception.
    with pytest.raises(VectorStoreError) as captured_error:
        store.query("rag_idx_missing", [1.0], 1)

    assert type(captured_error.value).__name__ == "VectorStoreError"


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
