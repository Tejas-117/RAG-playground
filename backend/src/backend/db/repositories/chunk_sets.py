"""SQLite read and write boundaries for reusable chunk-set artifacts."""

import json
import sqlite3
from typing import Any

from backend.db.connection import connect


def load_corpus_chunking_inputs(corpus_id: str) -> dict[str, Any] | None:
    """Load an immutable corpus and all canonical parse inputs in stable order.

    Args:
        corpus_id: Stable corpus identifier selected for chunking.

    Returns:
        Corpus documents and parse provenance, or None for an unknown corpus.
    """
    # Read all fingerprint and chunking inputs from one consistent connection.
    with connect() as connection:
        corpus_row = connection.execute(
            "SELECT id FROM corpus WHERE id = ?",
            (corpus_id,),
        ).fetchone()

        # Distinguish an unknown corpus from a known corpus with no documents.
        if corpus_row is None:
            return None

        document_rows = connection.execute(
            """
            SELECT document.id, document.original_filename, document.mime_type,
                   document.content_sha256, document.uploaded_at,
                   document_parse.id AS parse_id,
                   document_parse.normalized_text,
                   document_parse.parser_name,
                   document_parse.parser_version,
                   document_parse.document_metadata_json
            FROM document
            LEFT JOIN document_parse
                ON document_parse.document_id = document.id
            WHERE document.corpus_id = ?
            ORDER BY document.uploaded_at, document.id
            """,
            (corpus_id,),
        ).fetchall()
        documents: list[dict[str, Any]] = []

        # Attach each parse's ordered page and block offsets to its source document.
        for row in document_rows:
            pages: list[dict[str, Any]] = []
            blocks: list[dict[str, Any]] = []

            # Legacy documents may not have a parse, which the service reports explicitly.
            if row["parse_id"] is not None:
                page_rows = connection.execute(
                    """
                    SELECT page_number, character_start_offset,
                           character_end_offset
                    FROM parsed_page
                    WHERE parse_id = ?
                    ORDER BY page_number
                    """,
                    (row["parse_id"],),
                ).fetchall()
                block_rows = connection.execute(
                    """
                    SELECT ordinal, character_start_offset, character_end_offset
                    FROM parsed_block
                    WHERE parse_id = ?
                    ORDER BY ordinal
                    """,
                    (row["parse_id"],),
                ).fetchall()
                pages = [dict(page_row) for page_row in page_rows]
                blocks = [dict(block_row) for block_row in block_rows]

            documents.append(
                {
                    "id": row["id"],
                    "original_filename": row["original_filename"],
                    "mime_type": row["mime_type"],
                    "content_sha256": row["content_sha256"],
                    "parse_id": row["parse_id"],
                    "normalized_text": row["normalized_text"],
                    "parser_name": row["parser_name"],
                    "parser_version": row["parser_version"],
                    "parse_metadata": json.loads(row["document_metadata_json"])
                    if row["document_metadata_json"] is not None
                    else None,
                    "pages": pages,
                    "blocks": blocks,
                }
            )

    return {"id": corpus_id, "documents": documents}


def get_ready_chunk_set(fingerprint: str) -> dict[str, Any] | None:
    """Load a ready chunk set and its chunks by immutable fingerprint.

    Args:
        fingerprint: SHA-256 compatibility identity of the chunking inputs.

    Returns:
        Materialized artifact dictionary, or None when no ready match exists.
    """
    # Restrict reuse to complete artifacts so partial lifecycle rows are never consumed.
    with connect() as connection:
        chunk_set_row = connection.execute(
            """
            SELECT id, corpus_id, fingerprint, chunking_config_json,
                   chunker_name, chunker_version, status, chunk_count,
                   created_at, started_at, completed_at, duration_ms
            FROM chunk_set
            WHERE fingerprint = ? AND status = 'ready'
            """,
            (fingerprint,),
        ).fetchone()

        # Return absence before issuing the child query for an unknown artifact.
        if chunk_set_row is None:
            return None

        chunk_rows = connection.execute(
            """
            SELECT chunk.id, chunk.source_document_id, chunk.ordinal, chunk.text,
                   chunk.character_start_offset, chunk.character_end_offset,
                   chunk.token_start_offset, chunk.token_end_offset,
                   chunk.page_start, chunk.page_end,
                   chunk.section_path_json, chunk.source_metadata_json
            FROM chunk
            JOIN document ON document.id = chunk.source_document_id
            WHERE chunk.chunk_set_id = ?
            ORDER BY document.uploaded_at, document.id, chunk.ordinal
            """,
            (chunk_set_row["id"],),
        ).fetchall()

    return _materialize_chunk_set(chunk_set_row, chunk_rows)


def save_ready_chunk_set(
    chunk_set: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> None:
    """Atomically persist one complete ready chunk set and all child chunks.

    Args:
        chunk_set: Parent artifact fields matching the SQLite schema.
        chunks: Ordered chunk rows with complete provenance.

    Returns:
        None. Successful context-manager exit commits the transaction.

    Raises:
        sqlite3.Error: If any parent or child row cannot be saved.
    """
    # One transaction prevents failed writes from exposing partial chunk artifacts.
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO chunk_set (
                id, corpus_id, fingerprint, chunking_config_json,
                chunker_name, chunker_version, status, chunk_count,
                created_at, started_at, completed_at, duration_ms,
                error_code, error_details_json
            ) VALUES (?, ?, ?, ?, ?, ?, 'ready', ?, ?, ?, ?, ?, NULL, NULL)
            """,
            (
                chunk_set["id"],
                chunk_set["corpus_id"],
                chunk_set["fingerprint"],
                chunk_set["chunking_config_json"],
                chunk_set["chunker_name"],
                chunk_set["chunker_version"],
                len(chunks),
                chunk_set["created_at"],
                chunk_set["started_at"],
                chunk_set["completed_at"],
                chunk_set["duration_ms"],
            ),
        )

        # Insert children individually to preserve a narrow rollback-test boundary.
        for chunk in chunks:
            _insert_chunk(connection, chunk_set["id"], chunk)


def _insert_chunk(
    connection: sqlite3.Connection,
    chunk_set_id: str,
    chunk: dict[str, Any],
) -> None:
    """Insert one chunk inside its parent's active transaction.

    Args:
        connection: SQLite transaction that already owns the parent chunk set.
        chunk_set_id: Stable parent artifact identifier.
        chunk: Persistable chunk fields and serialized metadata.

    Returns:
        None. The caller controls transaction commit or rollback.
    """
    # Keep this operation small so any failure rolls back the entire artifact.
    connection.execute(
        """
        INSERT INTO chunk (
            id, chunk_set_id, source_document_id, ordinal, text,
            character_start_offset, character_end_offset,
            token_start_offset, token_end_offset, page_start, page_end,
            section_path_json, source_metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
        """,
        (
            chunk["id"],
            chunk_set_id,
            chunk["source_document_id"],
            chunk["ordinal"],
            chunk["text"],
            chunk["character_start_offset"],
            chunk["character_end_offset"],
            chunk["token_start_offset"],
            chunk["token_end_offset"],
            chunk["page_start"],
            chunk["page_end"],
            chunk["source_metadata_json"],
        ),
    )


def _materialize_chunk_set(
    chunk_set_row: sqlite3.Row,
    chunk_rows: list[sqlite3.Row],
) -> dict[str, Any]:
    """Convert SQLite rows into a typed-JSON-friendly artifact dictionary.

    Args:
        chunk_set_row: Ready parent row returned by SQLite.
        chunk_rows: Ordered child rows belonging to the parent.

    Returns:
        Deserialized parent fields and chunk provenance.
    """
    chunks: list[dict[str, Any]] = []

    # Decode persisted JSON fields without exposing storage serialization details.
    for row in chunk_rows:
        chunks.append(
            {
                "id": row["id"],
                "source_document_id": row["source_document_id"],
                "ordinal": row["ordinal"],
                "text": row["text"],
                "character_start_offset": row["character_start_offset"],
                "character_end_offset": row["character_end_offset"],
                "token_start_offset": row["token_start_offset"],
                "token_end_offset": row["token_end_offset"],
                "page_start": row["page_start"],
                "page_end": row["page_end"],
                "section_path": json.loads(row["section_path_json"])
                if row["section_path_json"] is not None
                else None,
                "source_metadata": json.loads(row["source_metadata_json"]),
            }
        )

    return {
        "id": chunk_set_row["id"],
        "corpus_id": chunk_set_row["corpus_id"],
        "fingerprint": chunk_set_row["fingerprint"],
        "configuration": json.loads(chunk_set_row["chunking_config_json"]),
        "chunker_name": chunk_set_row["chunker_name"],
        "chunker_version": chunk_set_row["chunker_version"],
        "status": chunk_set_row["status"],
        "chunk_count": chunk_set_row["chunk_count"],
        "created_at": chunk_set_row["created_at"],
        "started_at": chunk_set_row["started_at"],
        "completed_at": chunk_set_row["completed_at"],
        "duration_ms": chunk_set_row["duration_ms"],
        "chunks": chunks,
    }
