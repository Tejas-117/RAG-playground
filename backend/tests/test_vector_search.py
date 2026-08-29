"""Offline tests for query embedding and compatible vector-index search."""

from typing import Any

import pytest

from backend.embedding.models import (
    EmbeddingBatch,
    EmbeddingInputPurpose,
    InvalidEmbeddingResponseError,
    VectorSearchHit,
)
from backend.pipeline.configs import EmbeddingConfig
from backend.retrieval.vector_search import (
    IncompatibleVectorIndexError,
    InvalidVectorSearchRequestError,
    search_vector_index,
)


class FakeQueryEmbeddingProvider:
    """Return one configurable query batch and record the semantic input role."""

    identifier = "test-provider-adapter"
    version = "1"

    def __init__(self, batch: EmbeddingBatch) -> None:
        """Configure the deterministic query embedding response.

        Args:
            batch: Embedding response returned for every test query.

        Returns:
            None. Calls begin empty for later assertions.
        """
        self.batch = batch
        self.calls: list[tuple[str, list[str], EmbeddingInputPurpose]] = []

    def input_policy_version(self, model: str) -> str:
        """Return the policy identity stored by the compatible test index.

        Args:
            model: Model identifier unused by the deterministic fake.

        Returns:
            Stable test policy version.
        """
        return "test-query-policy-v1"

    def embed(
        self,
        model: str,
        texts: list[str],
        purpose: EmbeddingInputPurpose,
    ) -> EmbeddingBatch:
        """Record one query embedding call and return the configured batch.

        Args:
            model: Provider model selected for retrieval.
            texts: Normalized question submitted as one-element input.
            purpose: Query role required by asymmetric embedding models.

        Returns:
            Configured deterministic embedding batch.
        """
        self.calls.append((model, texts, purpose))
        return self.batch


class FakeQueryVectorStore:
    """Record one search call and return configurable provider-neutral hits."""

    identifier = "test-vector-store"
    version = "1"

    def __init__(self, hits: tuple[VectorSearchHit, ...] = ()) -> None:
        """Configure deterministic nearest-neighbor hits.

        Args:
            hits: Immutable results returned by the fake query boundary.

        Returns:
            None. Query calls begin empty for later assertions.
        """
        self.hits = hits
        self.calls: list[tuple[str, list[float], int]] = []

    def query(
        self,
        collection_name: str,
        vector: list[float],
        top_k: int,
    ) -> tuple[VectorSearchHit, ...]:
        """Record the exact collection search and return configured hits.

        Args:
            collection_name: Ready collection selected by the index artifact.
            vector: Explicit query vector returned by the embedding provider.
            top_k: Result limit already clamped to the persisted vector count.

        Returns:
            Configured deterministic hits.
        """
        self.calls.append((collection_name, vector, top_k))
        return self.hits


def _embedding_config() -> EmbeddingConfig:
    """Create the embedding-space configuration used by search tests.

    Args:
        None.

    Returns:
        Validated Ollama/Nomic cosine configuration.
    """
    return EmbeddingConfig(
        provider="ollama",
        model="nomic-embed-text",
        distance_metric="cosine",
    )


def _ready_vector_index(**overrides: Any) -> dict[str, Any]:
    """Create a complete compatible ready index with optional field overrides.

    Args:
        overrides: Artifact fields replacing compatible defaults.

    Returns:
        Ready vector-index dictionary accepted by the search service.
    """
    vector_index = {
        "status": "ready",
        "provider": "ollama",
        "model": "nomic-embed-text",
        "provider_model": "nomic-embed-text:latest",
        "provider_revision": "revision-1",
        "distance_metric": "cosine",
        "input_policy_version": "test-query-policy-v1",
        "indexer_name": "test-vector-store",
        "indexer_version": "1",
        "collection_name": "rag_idx_test",
        "dimensions": 3,
        "vector_count": 2,
    }
    vector_index.update(overrides)
    return vector_index


def test_search_vector_index_embeds_query_and_clamps_limit() -> None:
    """Verify one normalized question searches its exact compatible collection.

    Args:
        None.

    Returns:
        None. Assertions verify query role, explicit vector, limit, and hits.
    """
    provider = FakeQueryEmbeddingProvider(
        EmbeddingBatch(
            vectors=((1.0, 2.0, 3.0),),
            dimensions=3,
            provider_model="nomic-embed-text:latest",
            provider_revision="revision-1",
        )
    )
    expected_hits = (VectorSearchHit("chunk-1", 0.125),)
    vector_store = FakeQueryVectorStore(expected_hits)

    hits = search_vector_index(
        "  What is the refund policy?  ",
        10,
        _embedding_config(),
        _ready_vector_index(),
        provider,
        vector_store,
    )

    # The standalone service owns query preparation and index-bound search selection.
    assert provider.calls == [
        (
            "nomic-embed-text",
            ["What is the refund policy?"],
            EmbeddingInputPurpose.QUERY,
        )
    ]
    assert vector_store.calls == [("rag_idx_test", [1.0, 2.0, 3.0], 2)]
    assert hits == expected_hits


@pytest.mark.parametrize(
    ("question", "top_k"),
    [
        ("   ", 1),
        ("valid", 0),
        ("valid", True),
    ],
)
def test_search_vector_index_rejects_invalid_requests(
    question: str,
    top_k: int,
) -> None:
    """Verify invalid questions and limits fail before provider execution.

    Args:
        question: Blank or valid query text selected by the fixture.
        top_k: Invalid or otherwise paired result limit.

    Returns:
        None. Assertions verify request-boundary validation.
    """
    provider = FakeQueryEmbeddingProvider(
        EmbeddingBatch(vectors=((1.0,),), dimensions=1)
    )
    vector_store = FakeQueryVectorStore()

    with pytest.raises(InvalidVectorSearchRequestError):
        search_vector_index(
            question,
            top_k,
            _embedding_config(),
            _ready_vector_index(),
            provider,
            vector_store,
        )

    # Invalid requests must not consume provider or vector-store resources.
    assert provider.calls == []
    assert vector_store.calls == []


@pytest.mark.parametrize(
    ("overrides", "store_identifier"),
    [
        ({"status": "running"}, "test-vector-store"),
        ({"model": "all-minilm"}, "test-vector-store"),
        ({"distance_metric": "euclidean"}, "test-vector-store"),
        ({"input_policy_version": "old-policy"}, "test-vector-store"),
        ({}, "other-vector-store"),
        ({"dimensions": 0}, "test-vector-store"),
        ({"vector_count": 0}, "test-vector-store"),
        ({"collection_name": ""}, "test-vector-store"),
    ],
)
def test_search_vector_index_rejects_incompatible_artifact(
    overrides: dict[str, Any],
    store_identifier: str,
) -> None:
    """Verify incompatible index identity and shape fail before query embedding.

    Args:
        overrides: Index fields changed from compatible defaults.
        store_identifier: Adapter identity used for compatibility comparison.

    Returns:
        None. Assertions verify no provider or search operation begins.
    """
    provider = FakeQueryEmbeddingProvider(
        EmbeddingBatch(vectors=((1.0, 2.0, 3.0),), dimensions=3)
    )
    vector_store = FakeQueryVectorStore()
    vector_store.identifier = store_identifier

    with pytest.raises(IncompatibleVectorIndexError):
        search_vector_index(
            "question",
            1,
            _embedding_config(),
            _ready_vector_index(**overrides),
            provider,
            vector_store,
        )

    assert provider.calls == []
    assert vector_store.calls == []


@pytest.mark.parametrize(
    "batch",
    [
        EmbeddingBatch(
            vectors=((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
            dimensions=3,
        ),
        EmbeddingBatch(vectors=((1.0, 2.0),), dimensions=2),
        EmbeddingBatch(
            vectors=((1.0, 2.0, 3.0),),
            dimensions=3,
            provider_model="different-model",
        ),
        EmbeddingBatch(
            vectors=((1.0, 2.0, 3.0),),
            dimensions=3,
            provider_model="nomic-embed-text:latest",
            provider_revision="different-revision",
        ),
    ],
)
def test_search_vector_index_rejects_incompatible_query_batch(
    batch: EmbeddingBatch,
) -> None:
    """Verify vector count, width, model, and revision cannot change at query time.

    Args:
        batch: Query embedding response violating one compatibility invariant.

    Returns:
        None. Assertions verify the incompatible vector never reaches search.
    """
    provider = FakeQueryEmbeddingProvider(batch)
    vector_store = FakeQueryVectorStore()

    # Wrong counts are malformed responses; other changes are index incompatibility.
    expected_error = (
        InvalidEmbeddingResponseError
        if len(batch.vectors) != 1
        else IncompatibleVectorIndexError
    )
    with pytest.raises(expected_error):
        search_vector_index(
            "question",
            1,
            _embedding_config(),
            _ready_vector_index(),
            provider,
            vector_store,
        )

    assert vector_store.calls == []
