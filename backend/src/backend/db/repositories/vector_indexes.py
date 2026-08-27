"""SQLite boundaries for reusable ready vector-index artifacts."""

import json
import sqlite3
from typing import Any

from backend.db.connection import connect


def get_ready_vector_index(fingerprint: str) -> dict[str, Any] | None:
    """Load a ready vector index by its complete compatibility fingerprint.

    Args:
        fingerprint: SHA-256 identity of upstream and embedding-space inputs.

    Returns:
        Materialized ready artifact, or ``None`` when no compatible index exists.
    """
    # Only ready artifacts may be reused by pipeline runs.
    with connect() as connection:
        row = connection.execute(
            """
            SELECT * FROM vector_index
            WHERE fingerprint = ? AND status = 'ready'
            """,
            (fingerprint,),
        ).fetchone()

    # Avoid materialization work for an unknown compatibility identity.
    if row is None:
        return None

    return _materialize_vector_index(row)


def save_ready_vector_index(vector_index: dict[str, Any]) -> None:
    """Persist one fully built vector-index artifact.

    Args:
        vector_index: Complete ready artifact matching the SQLite schema.

    Returns:
        None. A successful transaction exposes the artifact for reuse.

    Raises:
        sqlite3.Error: If the artifact violates persistence constraints.
    """
    # Persist only after the external Chroma collection has been fully verified.
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO vector_index (
                id, chunk_set_id, fingerprint, embedding_config_json,
                provider, model, provider_model, provider_revision,
                dimensions, distance_metric, input_policy_version,
                indexer_name, indexer_version, collection_name, status,
                vector_count, created_at, started_at, completed_at, duration_ms
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready',
                ?, ?, ?, ?, ?
            )
            """,
            (
                vector_index["id"],
                vector_index["chunk_set_id"],
                vector_index["fingerprint"],
                vector_index["embedding_config_json"],
                vector_index["provider"],
                vector_index["model"],
                vector_index["provider_model"],
                vector_index["provider_revision"],
                vector_index["dimensions"],
                vector_index["distance_metric"],
                vector_index["input_policy_version"],
                vector_index["indexer_name"],
                vector_index["indexer_version"],
                vector_index["collection_name"],
                vector_index["vector_count"],
                vector_index["created_at"],
                vector_index["started_at"],
                vector_index["completed_at"],
                vector_index["duration_ms"],
            ),
        )


def load_vector_index(vector_index_id: str) -> dict[str, Any] | None:
    """Load one ready vector index by its stable application identifier.

    Args:
        vector_index_id: Stable index artifact identifier.

    Returns:
        Materialized ready index or ``None`` when it does not exist.
    """
    # Retrieval will use this boundary to resolve the Chroma collection reference.
    with connect() as connection:
        row = connection.execute(
            "SELECT * FROM vector_index WHERE id = ? AND status = 'ready'",
            (vector_index_id,),
        ).fetchone()

    return _materialize_vector_index(row) if row is not None else None


def _materialize_vector_index(row: sqlite3.Row) -> dict[str, Any]:
    """Convert one SQLite vector-index row into a JSON-friendly dictionary.

    Args:
        row: Ready vector-index row returned by SQLite.

    Returns:
        Deserialized reusable vector-index artifact.
    """
    # Keep JSON serialization private to the persistence adapter.
    return {
        "id": row["id"],
        "chunk_set_id": row["chunk_set_id"],
        "fingerprint": row["fingerprint"],
        "configuration": json.loads(row["embedding_config_json"]),
        "provider": row["provider"],
        "model": row["model"],
        "provider_model": row["provider_model"],
        "provider_revision": row["provider_revision"],
        "dimensions": row["dimensions"],
        "distance_metric": row["distance_metric"],
        "input_policy_version": row["input_policy_version"],
        "indexer_name": row["indexer_name"],
        "indexer_version": row["indexer_version"],
        "collection_name": row["collection_name"],
        "status": row["status"],
        "vector_count": row["vector_count"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
        "duration_ms": row["duration_ms"],
    }
