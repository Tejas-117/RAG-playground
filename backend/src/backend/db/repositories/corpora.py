"""SQLite persistence helpers for corpora and uploaded documents."""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from backend.db.connection import connect


def _utc_timestamp() -> str:
    """Create the UTC timestamp format used by the SQLite schema.

    Args:
        None.

    Returns:
        An ISO-8601 UTC timestamp ending in ``Z``.
    """
    # Use timezone-aware datetime values so persisted timestamps are unambiguous.
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def create_upload_batch(
    corpus_name: str,
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Persist one corpus and its uploaded document metadata transactionally.

    Args:
        corpus_name: User-facing name for the corpus created by this upload batch.
        documents: Validated document metadata, including storage paths and hashes.

    Returns:
        A response-ready corpus dictionary containing persisted document details.
    """
    # Generate stable identifiers and one timestamp for the corpus creation event.
    corpus_id = str(uuid4())
    timestamp = _utc_timestamp()
    persisted_documents: list[dict[str, Any]] = []

    # Open a transaction so the corpus and all document rows are committed together.
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO corpus (id, name, description, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (corpus_id, corpus_name, None, timestamp, timestamp),
        )

        # Insert each uploaded document under the corpus created for this request.
        for document in documents:
            document_id = str(uuid4())
            connection.execute(
                """
                INSERT INTO document (
                    id, corpus_id, original_filename, storage_path, mime_type,
                    size_bytes, content_sha256, uploaded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    corpus_id,
                    document["original_filename"],
                    document["storage_path"],
                    document["mime_type"],
                    document["size_bytes"],
                    document["content_sha256"],
                    timestamp,
                ),
            )
            persisted_documents.append(
                {"id": document_id, **document, "uploaded_at": timestamp}
            )

    # Return the same hierarchy used by the corpus listing endpoint.
    return {
        "id": corpus_id,
        "name": corpus_name,
        "description": None,
        "created_at": timestamp,
        "updated_at": timestamp,
        "documents": persisted_documents,
    }


def list_corpora() -> list[dict[str, Any]]:
    """Read all persisted corpora and their documents in upload order.

    Args:
        None.

    Returns:
        Corpus dictionaries containing nested document metadata.
    """
    # Open a short-lived read connection so the route does not hold shared state.
    with connect() as connection:
        corpus_rows = connection.execute(
            "SELECT id, name, description, created_at, updated_at FROM corpus "
            "ORDER BY created_at DESC, id DESC"
        ).fetchall()
        document_rows = connection.execute(
            """
            SELECT id, corpus_id, original_filename, storage_path, mime_type,
                   size_bytes, content_sha256, uploaded_at
            FROM document ORDER BY uploaded_at ASC, id ASC
            """
        ).fetchall()

    # Group documents by corpus ID before constructing the response hierarchy.
    documents_by_corpus: dict[str, list[dict[str, Any]]] = {}
    for row in document_rows:
        documents_by_corpus.setdefault(row["corpus_id"], []).append(
            {
                "id": row["id"],
                "original_filename": row["original_filename"],
                "storage_path": row["storage_path"],
                "mime_type": row["mime_type"],
                "size_bytes": row["size_bytes"],
                "content_sha256": row["content_sha256"],
                "uploaded_at": row["uploaded_at"],
            }
        )

    # Add each corpus's child documents while preserving newest-first corpus order.
    return [
        {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "documents": documents_by_corpus.get(row["id"], []),
        }
        for row in corpus_rows
    ]
