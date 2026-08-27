"""Build or reuse complete vector indexes from persisted chunk artifacts."""

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4, uuid5

from backend.db.repositories.vector_indexes import (
    get_ready_vector_index,
    save_ready_vector_index,
)
from backend.embedding.models import (
    EmbeddingBatch,
    EmbeddingInputPurpose,
    EmbeddingProvider,
    InvalidEmbeddingResponseError,
    VectorStore,
    VectorStoreError,
)
from backend.embedding.providers import get_embedding_provider
from backend.embedding.vector_store import get_vector_store
from backend.pipeline.configs import EmbeddingConfig

# Use the module name to distinguish embedding/index lifecycle records.
logger = logging.getLogger(__name__)

# Bound provider and Chroma writes without exposing batch tuning in the MVP UI.
EMBEDDING_BATCH_SIZE = 32

# UUID5 keeps vector-index IDs stable for identical compatibility fingerprints.
_VECTOR_INDEX_NAMESPACE = UUID("b85d8ea2-d464-4af0-92e3-2bc2aca0b4e7")


@dataclass(frozen=True)
class VectorIndexBuildResult:
    """Return one ready vector index and how the request obtained it.

    Attributes:
        artifact: Complete persisted vector-index artifact.
        reused: Whether a compatible ready artifact already existed.
    """

    artifact: dict[str, Any]
    reused: bool


class EmptyChunkSetError(ValueError):
    """Report that a ready chunk artifact contains no embeddable records."""


def build_or_reuse_vector_index(
    chunk_set: dict[str, Any],
    config: EmbeddingConfig,
    provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
) -> VectorIndexBuildResult:
    """Build or reuse a complete vector index for one ready chunk set.

    Args:
        chunk_set: Ready chunk artifact with ordered chunk dictionaries.
        config: Validated provider, model, and distance configuration.
        provider: Optional provider adapter used by deterministic tests.
        vector_store: Optional vector-store adapter used by deterministic tests.

    Returns:
        Ready persisted vector index and whether it was reused.

    Raises:
        EmptyChunkSetError: If the upstream artifact contains no chunks.
        EmbeddingProviderError: If provider execution or validation fails.
        VectorStoreError: If the external index cannot be materialized.
        sqlite3.Error: If ready artifact persistence fails.
    """
    chunks = chunk_set["chunks"]

    # A reusable vector index must contain at least one searchable chunk.
    if not chunks:
        raise EmptyChunkSetError(chunk_set["id"])

    resolved_provider = provider or get_embedding_provider(config.provider)
    resolved_store = vector_store or get_vector_store()
    input_policy_version = resolved_provider.input_policy_version(config.model)
    started_counter = perf_counter()
    started_at = _utc_timestamp()
    first_chunks = chunks[:EMBEDDING_BATCH_SIZE]
    logger.info(
        "vector_index_requested chunk_set_id=%s provider=%s model=%s "
        "distance_metric=%s chunk_count=%d",
        chunk_set["id"],
        config.provider,
        config.model,
        config.distance_metric.value,
        len(chunks),
    )

    # A real provider response supplies the dimensions needed by compatibility checks.
    first_batch = resolved_provider.embed(
        config.model,
        [chunk["text"] for chunk in first_chunks],
        EmbeddingInputPurpose.DOCUMENT,
    )
    fingerprint = _build_fingerprint(
        chunk_set,
        config,
        first_batch,
        input_policy_version,
        resolved_provider,
        resolved_store,
    )
    existing = get_ready_vector_index(fingerprint)

    # Reuse only after a live provider response confirms the current vector dimensions.
    if existing is not None:
        logger.info(
            "vector_index_reused vector_index_id=%s chunk_set_id=%s "
            "vector_count=%d dimensions=%d",
            existing["id"],
            chunk_set["id"],
            existing["vector_count"],
            existing["dimensions"],
        )
        return VectorIndexBuildResult(artifact=existing, reused=True)

    collection_name = f"rag_idx_{uuid4().hex}"
    collection_created = False

    try:
        # Each attempt owns a unique collection, making rollback and races safe.
        resolved_store.create_collection(
            collection_name,
            config.distance_metric.value,
        )
        collection_created = True
        _store_batch(
            resolved_store,
            collection_name,
            chunk_set,
            first_chunks,
            first_batch,
        )
        logger.info(
            "vector_index_build_started chunk_set_id=%s collection_name=%s "
            "dimensions=%d batch_size=%d",
            chunk_set["id"],
            collection_name,
            first_batch.dimensions,
            EMBEDDING_BATCH_SIZE,
        )

        # Embed and store remaining chunks without retaining every vector in memory.
        for batch_start in range(
            EMBEDDING_BATCH_SIZE, len(chunks), EMBEDDING_BATCH_SIZE
        ):
            batch_chunks = chunks[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
            batch = resolved_provider.embed(
                config.model,
                [chunk["text"] for chunk in batch_chunks],
                EmbeddingInputPurpose.DOCUMENT,
            )
            _validate_batch_compatibility(first_batch, batch)
            _store_batch(
                resolved_store,
                collection_name,
                chunk_set,
                batch_chunks,
                batch,
            )

        vector_count = resolved_store.count(collection_name)

        # Count verification prevents a partial external collection becoming reusable.
        if vector_count != len(chunks):
            raise VectorStoreError(
                "The vector collection count does not match the chunk set."
            )

        completed_at = _utc_timestamp()
        duration_ms = _elapsed_milliseconds(started_counter)
        vector_index_id = str(
            uuid5(_VECTOR_INDEX_NAMESPACE, f"vector-index:{fingerprint}")
        )
        artifact = {
            "id": vector_index_id,
            "chunk_set_id": chunk_set["id"],
            "fingerprint": fingerprint,
            "embedding_config_json": _canonical_json(config.model_dump(mode="json")),
            "provider": config.provider,
            "model": config.model,
            "provider_model": first_batch.provider_model,
            "provider_revision": first_batch.provider_revision,
            "dimensions": first_batch.dimensions,
            "distance_metric": config.distance_metric.value,
            "input_policy_version": input_policy_version,
            "indexer_name": resolved_store.identifier,
            "indexer_version": resolved_store.version,
            "collection_name": collection_name,
            "vector_count": vector_count,
            "created_at": started_at,
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_ms": duration_ms,
        }

        try:
            # SQLite exposes the external collection only after full verification.
            save_ready_vector_index(artifact)
        except sqlite3.IntegrityError:
            # A concurrent identical build may have committed the fingerprint first.
            concurrent = get_ready_vector_index(fingerprint)

            # Delete only this attempt's unique collection before reusing the winner.
            if concurrent is not None:
                resolved_store.delete_collection(collection_name)
                collection_created = False
                logger.info(
                    "vector_index_reused_after_race vector_index_id=%s chunk_set_id=%s",
                    concurrent["id"],
                    chunk_set["id"],
                )
                return VectorIndexBuildResult(artifact=concurrent, reused=True)

            raise

        persisted = get_ready_vector_index(fingerprint)

        # Read back through the repository so new and reused results share one shape.
        if persisted is None:
            raise RuntimeError(
                "The ready vector index could not be read after persistence."
            )

        collection_created = False
        logger.info(
            "vector_index_completed vector_index_id=%s chunk_set_id=%s "
            "vector_count=%d dimensions=%d duration_ms=%d",
            persisted["id"],
            chunk_set["id"],
            vector_count,
            first_batch.dimensions,
            duration_ms,
        )
        return VectorIndexBuildResult(artifact=persisted, reused=False)
    except Exception:
        # Roll back only the collection owned by this failed build attempt.
        if collection_created:
            resolved_store.delete_collection(collection_name)

        logger.exception(
            "vector_index_build_failed chunk_set_id=%s collection_name=%s",
            chunk_set["id"],
            collection_name,
        )
        raise


def _store_batch(
    vector_store: VectorStore,
    collection_name: str,
    chunk_set: dict[str, Any],
    chunks: list[dict[str, Any]],
    batch: EmbeddingBatch,
) -> None:
    """Persist one aligned embedding batch with bounded scalar provenance.

    Args:
        vector_store: Adapter receiving explicit vectors.
        collection_name: Collection owned by the current build attempt.
        chunk_set: Parent artifact providing corpus and chunk-set IDs.
        chunks: Ordered source chunks aligned with the embedding batch.
        batch: Validated provider vectors for those chunks.

    Returns:
        None. The adapter persists the aligned records.
    """
    metadata = [_build_vector_metadata(chunk_set, chunk) for chunk in chunks]
    vector_store.add(
        collection_name,
        [chunk["id"] for chunk in chunks],
        [list(vector) for vector in batch.vectors],
        metadata,
    )


def _build_vector_metadata(
    chunk_set: dict[str, Any],
    chunk: dict[str, Any],
) -> dict[str, str | int | float | bool]:
    """Select flat scalar provenance suitable for Chroma filters.

    Args:
        chunk_set: Parent chunk artifact containing corpus identity.
        chunk: Persisted chunk and its source metadata.

    Returns:
        Flat metadata without nulls, nested values, or duplicated chunk text.
    """
    source_document = chunk["source_metadata"].get("document") or {}
    metadata: dict[str, str | int | float | bool] = {
        "corpus_id": chunk_set["corpus_id"],
        "chunk_set_id": chunk_set["id"],
        "source_document_id": chunk["source_document_id"],
        "chunk_ordinal": chunk["ordinal"],
    }
    original_filename = source_document.get("original_filename")

    # Chroma metadata does not accept null values, so add optional fields selectively.
    if isinstance(original_filename, str):
        metadata["original_filename"] = original_filename

    if chunk["page_start"] is not None:
        metadata["page_start"] = chunk["page_start"]

    if chunk["page_end"] is not None:
        metadata["page_end"] = chunk["page_end"]

    return metadata


def _validate_batch_compatibility(
    first_batch: EmbeddingBatch,
    batch: EmbeddingBatch,
) -> None:
    """Ensure every provider batch belongs to one compatible vector space.

    Args:
        first_batch: First response defining dimensions and provider provenance.
        batch: Later response that must match the first response.

    Returns:
        None when the batch belongs to the same vector space.

    Raises:
        InvalidEmbeddingResponseError: If dimensions or provenance changes.
    """
    # Chroma cannot mix vector widths in one collection.
    if batch.dimensions != first_batch.dimensions:
        raise InvalidEmbeddingResponseError(
            "The embedding provider changed vector dimensions between batches."
        )

    # A provider-reported model change would make one collection non-reproducible.
    if (
        first_batch.provider_model is not None
        and batch.provider_model is not None
        and batch.provider_model != first_batch.provider_model
    ):
        raise InvalidEmbeddingResponseError(
            "The embedding provider changed models between batches."
        )

    # Compare revisions only when both responses expose immutable revision metadata.
    if (
        first_batch.provider_revision is not None
        and batch.provider_revision is not None
        and batch.provider_revision != first_batch.provider_revision
    ):
        raise InvalidEmbeddingResponseError(
            "The embedding provider changed model revisions between batches."
        )


def _build_fingerprint(
    chunk_set: dict[str, Any],
    config: EmbeddingConfig,
    first_batch: EmbeddingBatch,
    input_policy_version: str,
    provider: EmbeddingProvider,
    vector_store: VectorStore,
) -> str:
    """Hash every known immutable input defining the vector space.

    Args:
        chunk_set: Exact upstream chunk artifact.
        config: Provider, model, and distance configuration.
        first_batch: Live response defining dimensions and provider provenance.
        input_policy_version: Versioned provider text transformation.
        provider: Adapter implementation identity.
        vector_store: Index implementation identity.

    Returns:
        Lowercase SHA-256 compatibility fingerprint.
    """
    payload = {
        "chunk_set": {
            "id": chunk_set["id"],
            "fingerprint": chunk_set["fingerprint"],
        },
        "embedding_config": config.model_dump(mode="json"),
        "provider_model": first_batch.provider_model,
        "provider_revision": first_batch.provider_revision,
        "dimensions": first_batch.dimensions,
        "input_policy_version": input_policy_version,
        "provider_adapter": {
            "identifier": provider.identifier,
            "version": provider.version,
        },
        "vector_store": {
            "identifier": vector_store.identifier,
            "version": vector_store.version,
        },
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _utc_timestamp() -> str:
    """Create an ISO-8601 UTC timestamp for artifact persistence.

    Args:
        None.

    Returns:
        UTC timestamp ending in ``Z``.
    """
    # Keep artifact timestamp format aligned with other persistence services.
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _elapsed_milliseconds(started_counter: float) -> int:
    """Calculate non-negative elapsed milliseconds from a monotonic counter.

    Args:
        started_counter: Value previously returned by ``perf_counter``.

    Returns:
        Rounded non-negative elapsed duration.
    """
    # A monotonic clock should not regress, but clamp defensively for persistence.
    return max(0, round((perf_counter() - started_counter) * 1000))


def _canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible fingerprint or configuration value.

    Args:
        value: JSON-compatible value requiring stable serialization.

    Returns:
        Compact JSON with deterministic key ordering.
    """
    # Canonical serialization makes hashes stable across processes.
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
