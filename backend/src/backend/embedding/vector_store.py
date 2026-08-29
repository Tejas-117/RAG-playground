"""Persistent Chroma adapter for explicit provider-generated embeddings."""

import math
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from backend.embedding.models import (
    InvalidVectorSearchResponseError,
    VectorSearchHit,
    VectorStoreError,
)

# Keep generated Chroma data beside other ignored backend runtime artifacts.
DEFAULT_CHROMA_DATA_PATH = Path(__file__).resolve().parents[3] / "chroma_data"

# Translate public experiment metrics into Chroma's index-space identifiers.
_CHROMA_DISTANCE_SPACES = {
    "cosine": "cosine",
    "dot_product": "ip",
    "euclidean": "l2",
}

# Only collections created by this application may be removed during a dev reset.
MANAGED_COLLECTION_PREFIX = "rag_idx_"


class ChromaVectorStore:
    """Store explicit dense vectors in an embedded persistent Chroma database."""

    identifier = "chroma-persistent"
    version = "1"

    def __init__(
        self, client: Any | None = None, data_path: Path | None = None
    ) -> None:
        """Configure an injectable Chroma client and local persistence path.

        Args:
            client: Optional fake or Chroma client used by tests.
            data_path: Optional storage directory overriding ``CHROMA_DATA_PATH``.

        Returns:
            None. Persistent data is created lazily by Chroma.
        """
        configured_path = os.getenv("CHROMA_DATA_PATH")
        resolved_path = data_path or (
            Path(configured_path) if configured_path else DEFAULT_CHROMA_DATA_PATH
        )

        # Disable product telemetry for this local single-user data store.
        self._client = client or chromadb.PersistentClient(
            path=str(resolved_path),
            settings=Settings(anonymized_telemetry=False),
        )

    def create_collection(self, name: str, distance_metric: str) -> None:
        """Create an empty collection with no automatic embedding function.

        Args:
            name: Unique collection name owned by one build attempt.
            distance_metric: Application-level distance metric identifier.

        Returns:
            None. Chroma persists the empty collection.

        Raises:
            VectorStoreError: If the metric or Chroma operation is invalid.
        """
        space = _CHROMA_DISTANCE_SPACES.get(distance_metric)

        # Reject unknown metrics before creating a collection with the wrong default.
        if space is None:
            raise VectorStoreError("The vector-store distance metric is unsupported.")

        try:
            self._client.create_collection(
                name=name,
                embedding_function=None,
                configuration={"hnsw": {"space": space}},
            )
        except Exception as error:
            raise VectorStoreError(
                "The vector collection could not be created."
            ) from error

    def add(
        self,
        collection_name: str,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict[str, str | int | float | bool]],
    ) -> None:
        """Add one aligned explicit-vector batch to Chroma.

        Args:
            collection_name: Existing collection receiving the batch.
            ids: Stable chunk identifiers.
            vectors: Dense provider-generated vectors.
            metadata: Scalar chunk provenance aligned with each identifier.

        Returns:
            None. Chroma persists the batch.

        Raises:
            VectorStoreError: If Chroma rejects or cannot persist the batch.
        """
        try:
            collection = self._client.get_collection(
                name=collection_name,
                embedding_function=None,
            )
            collection.add(ids=ids, embeddings=vectors, metadatas=metadata)
        except Exception as error:
            raise VectorStoreError("The vector batch could not be stored.") from error

    def count(self, collection_name: str) -> int:
        """Return the number of records persisted in one Chroma collection.

        Args:
            collection_name: Existing collection to inspect.

        Returns:
            Non-negative vector record count.

        Raises:
            VectorStoreError: If the collection cannot be read.
        """
        try:
            collection = self._client.get_collection(
                name=collection_name,
                embedding_function=None,
            )
            return int(collection.count())
        except Exception as error:
            raise VectorStoreError(
                "The vector collection could not be verified."
            ) from error

    def query(
        self,
        collection_name: str,
        vector: list[float],
        top_k: int,
    ) -> tuple[VectorSearchHit, ...]:
        """Search one Chroma collection with an explicit query vector.

        Args:
            collection_name: Existing collection containing indexed chunk vectors.
            vector: Explicit provider-generated query vector.
            top_k: Maximum number of nearest chunks to return.

        Returns:
            Ranked immutable hits with raw Chroma distances.

        Raises:
            VectorStoreError: If the request or Chroma operation is invalid.
            InvalidVectorSearchResponseError: If Chroma returns malformed results.
        """
        # Reject invalid limits at the provider-neutral adapter boundary.
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise VectorStoreError("The vector search limit must be positive.")

        # Require one non-empty finite vector before calling the external store.
        if not vector or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in vector
        ):
            raise VectorStoreError("The vector search query is invalid.")

        try:
            # The collection owns its configured metric and has no embedding function.
            collection = self._client.get_collection(
                name=collection_name,
                embedding_function=None,
            )
            collection_count = int(collection.count())

            # An empty collection has no valid n_results value and no possible hits.
            if collection_count == 0:
                return ()

            # Avoid requesting more neighbors than the collection can contain.
            effective_top_k = min(top_k, collection_count)
            result = collection.query(
                query_embeddings=[vector],
                n_results=effective_top_k,
                include=["distances"],
            )
        except Exception as error:
            raise VectorStoreError(
                "The vector collection could not be searched."
            ) from error

        return _validate_query_result(result, effective_top_k)

    def delete_collection(self, name: str) -> None:
        """Delete one exact Chroma collection during cleanup.

        Args:
            name: Exact collection name owned by a failed or losing attempt.

        Returns:
            None. Missing collections are treated as already cleaned up.
        """
        try:
            self._client.delete_collection(name=name)
        except Exception:  # noqa: BLE001 - cleanup must tolerate adapter failures.
            # Rollback cleanup is best-effort and must not hide the primary failure.
            return

    def delete_managed_collections(self) -> int:
        """Delete every application-owned collection during a local dev reset.

        Args:
            None.

        Returns:
            Number of managed Chroma collections deleted.

        Raises:
            VectorStoreError: If Chroma collections cannot be listed or deleted.
        """
        deleted_collections = 0

        try:
            # Chroma versions may return collection objects or collection names.
            for collection in self._client.list_collections():
                collection_name = (
                    collection if isinstance(collection, str) else collection.name
                )

                # Preserve any collection not created by the RAG Playground indexer.
                if not collection_name.startswith(MANAGED_COLLECTION_PREFIX):
                    continue

                self._client.delete_collection(name=collection_name)
                deleted_collections += 1
        except Exception as error:
            # Reset is an explicit operation, so cleanup failures must be visible.
            raise VectorStoreError(
                "The managed vector collections could not be cleared."
            ) from error

        return deleted_collections


def get_vector_store() -> ChromaVectorStore:
    """Create the production embedded Chroma adapter.

    Args:
        None.

    Returns:
        Persistent vector-store adapter configured from the environment.
    """
    # The adapter holds only Chroma client resources, never mutable pipeline run state.
    return ChromaVectorStore()


def _validate_query_result(
    result: Any,
    expected_count: int,
) -> tuple[VectorSearchHit, ...]:
    """Validate and normalize one untrusted single-vector Chroma response.

    Args:
        result: Chroma query response containing nested IDs and distances.
        expected_count: Exact number of nearest-neighbor hits requested.

    Returns:
        Hits ordered by ascending raw distance and stable chunk ID.

    Raises:
        InvalidVectorSearchResponseError: If the response cannot represent hits.
    """
    # A single query vector must produce one nested ID list and one distance list.
    if not isinstance(result, dict):
        raise InvalidVectorSearchResponseError(
            "The vector store returned an invalid search response."
        )

    raw_ids = result.get("ids")
    raw_distances = result.get("distances")
    if (
        not isinstance(raw_ids, list)
        or len(raw_ids) != 1
        or not isinstance(raw_ids[0], list)
        or not isinstance(raw_distances, list)
        or len(raw_distances) != 1
        or not isinstance(raw_distances[0], list)
    ):
        raise InvalidVectorSearchResponseError(
            "The vector store returned an invalid search response."
        )

    ids = raw_ids[0]
    distances = raw_distances[0]

    # Exact positional alignment is required before constructing ranked hits.
    if len(ids) != expected_count or len(distances) != expected_count:
        raise InvalidVectorSearchResponseError(
            "The vector store returned an unexpected number of search results."
        )

    hits: list[VectorSearchHit] = []
    seen_ids: set[str] = set()

    # Validate every identifier and distance before returning any partial result.
    for chunk_id, raw_distance in zip(ids, distances, strict=True):
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise InvalidVectorSearchResponseError(
                "The vector store returned an invalid chunk identifier."
            )

        # One chunk may appear at most once in a nearest-neighbor result.
        if chunk_id in seen_ids:
            raise InvalidVectorSearchResponseError(
                "The vector store returned duplicate chunk identifiers."
            )

        if (
            isinstance(raw_distance, bool)
            or not isinstance(raw_distance, (int, float))
            or not math.isfinite(float(raw_distance))
        ):
            raise InvalidVectorSearchResponseError(
                "The vector store returned an invalid distance."
            )

        seen_ids.add(chunk_id)
        hits.append(
            VectorSearchHit(
                chunk_id=chunk_id,
                raw_distance=float(raw_distance),
            )
        )

    # Preserve nearest-first ordering while making equal-distance ranks deterministic.
    return tuple(sorted(hits, key=lambda hit: (hit.raw_distance, hit.chunk_id)))
