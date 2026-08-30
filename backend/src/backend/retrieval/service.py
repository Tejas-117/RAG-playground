"""Coordinate query embedding, vector search, and chunk hydration."""

from backend.embedding.models import EmbeddingProvider, VectorStore
from backend.pipeline.configs import EmbeddingConfig, RetrievalConfig
from backend.retrieval.chunk_hydration import (
    ChunkLoader,
    hydrate_vector_search_hits,
)
from backend.retrieval.models import HydratedVectorSearchHit
from backend.retrieval.vector_search import search_vector_index


class InvalidRetrievalArtifactError(ValueError):
    """Report a vector index that lacks required retrieval provenance."""


def retrieve_chunks(
    question: str,
    retrieval_config: RetrievalConfig,
    embedding_config: EmbeddingConfig,
    vector_index: dict[str, object],
    provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
    chunk_loader: ChunkLoader | None = None,
) -> tuple[HydratedVectorSearchHit, ...]:
    """Search one compatible vector index and hydrate its ranked chunk hits.

    Args:
        question: User question used to create the retrieval query vector.
        retrieval_config: Validated nearest-neighbor result configuration.
        embedding_config: Embedding space required by the selected index.
        vector_index: Ready reusable vector-index artifact selected for the run.
        provider: Optional embedding adapter override for deterministic tests.
        vector_store: Optional vector-store override for deterministic tests.
        chunk_loader: Optional scoped SQLite chunk loader override for tests.

    Returns:
        Ranked, hydrated chunks with raw distances and source provenance.

    Raises:
        InvalidRetrievalArtifactError: If chunk-set provenance is unavailable.
        InvalidVectorSearchRequestError: If the question or limit is invalid.
        IncompatibleVectorIndexError: If the query and index spaces differ.
        ChunkHydrationError: If any search hit is missing or belongs elsewhere.
    """
    chunk_set_id = vector_index.get("chunk_set_id")

    # Hydration must be scoped to the exact chunk set that produced the index.
    if not isinstance(chunk_set_id, str) or not chunk_set_id.strip():
        raise InvalidRetrievalArtifactError(
            "The vector index is missing its chunk-set identifier."
        )

    # Search returns only stable IDs and distances, keeping vector storage lightweight.
    vector_hits = search_vector_index(
        question,
        retrieval_config.top_k,
        embedding_config,
        vector_index,
        provider,
        vector_store,
    )

    # Resolve full text and provenance from immutable application-owned chunk rows.
    if chunk_loader is None:
        return hydrate_vector_search_hits(vector_hits, chunk_set_id)

    return hydrate_vector_search_hits(vector_hits, chunk_set_id, chunk_loader)
