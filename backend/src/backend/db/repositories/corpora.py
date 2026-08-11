"""SQLite persistence helpers for corpora and uploaded documents."""

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from backend.db.connection import connect
from backend.ingestion.canonicalization import CanonicalDocument


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
        documents: Validated document metadata, hashes, and canonical parse artifacts.

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
            parse_summary = _insert_parse_artifact(
                connection,
                document_id,
                document["parse"],
                document["parse_duration_ms"],
                timestamp,
            )
            response_document = {
                key: value
                for key, value in document.items()
                if key not in {"parse", "parse_duration_ms"}
            }
            persisted_documents.append(
                {
                    "id": document_id,
                    **response_document,
                    "uploaded_at": timestamp,
                    "parse": parse_summary,
                }
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
            SELECT document.id, document.corpus_id, document.original_filename,
                   document.storage_path, document.mime_type,
                   document.size_bytes, document.content_sha256,
                   document.uploaded_at,
                   document_parse.id AS parse_id,
                   document_parse.parser_name,
                   document_parse.parser_version,
                   document_parse.warnings_json,
                   document_parse.page_count,
                   document_parse.block_count,
                   document_parse.utf8_size_bytes,
                   document_parse.character_count,
                   document_parse.duration_ms,
                   document_parse.created_at AS parsed_at
            FROM document
            LEFT JOIN document_parse ON document_parse.document_id = document.id
            ORDER BY uploaded_at ASC, document.rowid ASC
            """
        ).fetchall()

    # Group documents by corpus ID before constructing the response hierarchy.
    documents_by_corpus: dict[str, list[dict[str, Any]]] = {}
    for row in document_rows:
        document = {
            "id": row["id"],
            "original_filename": row["original_filename"],
            "storage_path": row["storage_path"],
            "mime_type": row["mime_type"],
            "size_bytes": row["size_bytes"],
            "content_sha256": row["content_sha256"],
            "uploaded_at": row["uploaded_at"],
            "parse": _parse_summary_from_row(row),
        }
        documents_by_corpus.setdefault(row["corpus_id"], []).append(document)

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


def _insert_parse_artifact(
    connection: Any,
    document_id: str,
    parsed: CanonicalDocument,
    duration_ms: int,
    timestamp: str,
) -> dict[str, Any]:
    """Insert one canonical parse and all of its offset records.

    Args:
        connection: Active SQLite transaction owning the upload batch.
        document_id: Stable identifier of the newly inserted source document.
        parsed: Validated canonical document to persist.
        duration_ms: Wall-clock parsing and canonicalization duration.
        timestamp: Shared UTC creation timestamp for this upload batch.

    Returns:
        A response-safe summary of the inserted parse artifact.
    """
    parse_id = str(uuid4())
    connection.execute(
        """
        INSERT INTO document_parse (
            id, document_id, normalized_text, utf8_size_bytes,
            character_count, parser_name, parser_version,
            document_metadata_json, warnings_json, page_count, block_count,
            duration_ms, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            parse_id,
            document_id,
            parsed.normalized_text,
            parsed.utf8_size_bytes,
            len(parsed.normalized_text),
            parsed.parser_name,
            parsed.parser_version,
            _serialize_json(parsed.metadata),
            _serialize_json(parsed.warnings),
            len(parsed.pages),
            parsed.block_count,
            duration_ms,
            timestamp,
        ),
    )

    # Insert pages first so nested blocks can reference their stable page IDs.
    for page in parsed.pages:
        page_id = str(uuid4())
        connection.execute(
            """
            INSERT INTO parsed_page (
                id, parse_id, page_number, character_start_offset,
                character_end_offset, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                page_id,
                parse_id,
                page.page_number,
                page.character_start_offset,
                page.character_end_offset,
                _serialize_json(page.metadata),
            ),
        )

        # Store only offsets and provenance; block text remains in normalized_text.
        for block in page.blocks:
            connection.execute(
                """
                INSERT INTO parsed_block (
                    id, parse_id, page_id, ordinal, source_block_index,
                    character_start_offset, character_end_offset,
                    bounding_box_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    parse_id,
                    page_id,
                    block.ordinal,
                    block.source_block_index,
                    block.character_start_offset,
                    block.character_end_offset,
                    _serialize_json(block.bounding_box)
                    if block.bounding_box is not None
                    else None,
                    _serialize_json(block.metadata),
                ),
            )

    return {
        "id": parse_id,
        "parser_name": parsed.parser_name,
        "parser_version": parsed.parser_version,
        "warnings": parsed.warnings,
        "page_count": len(parsed.pages),
        "block_count": parsed.block_count,
        "utf8_size_bytes": parsed.utf8_size_bytes,
        "character_count": len(parsed.normalized_text),
        "duration_ms": duration_ms,
        "created_at": timestamp,
    }


def _serialize_json(value: Any) -> str:
    """Serialize parser metadata deterministically for SQLite storage.

    Args:
        value: JSON-compatible parser metadata, warnings, or coordinates.

    Returns:
        A compact deterministic JSON string.
    """
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _parse_summary_from_row(row: Any) -> dict[str, Any] | None:
    """Convert joined SQLite parse columns into an API summary.

    Args:
        row: SQLite row returned by the corpus document listing query.

    Returns:
        A parse summary, or None for legacy documents without a parse artifact.
    """
    # Keep old databases readable while newly uploaded documents always have parses.
    if row["parse_id"] is None:
        return None

    return {
        "id": row["parse_id"],
        "parser_name": row["parser_name"],
        "parser_version": row["parser_version"],
        "warnings": json.loads(row["warnings_json"]),
        "page_count": row["page_count"],
        "block_count": row["block_count"],
        "utf8_size_bytes": row["utf8_size_bytes"],
        "character_count": row["character_count"],
        "duration_ms": row["duration_ms"],
        "created_at": row["parsed_at"],
    }
