"""Read boundary for persisted canonical parse artifacts."""

import json
from typing import Any

from backend.db.connection import connect


def get_parsed_document(document_id: str) -> dict[str, Any] | None:
    """Load canonical text and ordered provenance for one source document.

    Args:
        document_id: Stable source-document identifier whose parse should be read.

    Returns:
        The canonical parse with pages and blocks, or None when it does not exist.
    """
    # Read the single text row before loading its smaller ordered offset records.
    with connect() as connection:
        parse_row = connection.execute(
            """
            SELECT id, normalized_text, parser_name, parser_version,
                   document_metadata_json, warnings_json
            FROM document_parse WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()

        # Return an explicit absence result for unknown or legacy documents.
        if parse_row is None:
            return None

        page_rows = connection.execute(
            """
            SELECT id, page_number, character_start_offset,
                   character_end_offset, metadata_json
            FROM parsed_page WHERE parse_id = ? ORDER BY page_number
            """,
            (parse_row["id"],),
        ).fetchall()
        block_rows = connection.execute(
            """
            SELECT page_id, ordinal, source_block_index,
                   character_start_offset, character_end_offset,
                   bounding_box_json, metadata_json
            FROM parsed_block WHERE parse_id = ? ORDER BY ordinal
            """,
            (parse_row["id"],),
        ).fetchall()

    blocks_by_page: dict[str, list[dict[str, Any]]] = {}

    # Group blocks under pages while preserving their canonical ordinal order.
    for row in block_rows:
        blocks_by_page.setdefault(row["page_id"], []).append(
            {
                "ordinal": row["ordinal"],
                "source_block_index": row["source_block_index"],
                "character_start_offset": row["character_start_offset"],
                "character_end_offset": row["character_end_offset"],
                "bounding_box": json.loads(row["bounding_box_json"])
                if row["bounding_box_json"] is not None
                else None,
                "metadata": json.loads(row["metadata_json"]),
            }
        )

    pages = []

    # Materialize ordered pages with their nested source-aware blocks.
    for row in page_rows:
        pages.append(
            {
                "page_number": row["page_number"],
                "character_start_offset": row["character_start_offset"],
                "character_end_offset": row["character_end_offset"],
                "metadata": json.loads(row["metadata_json"]),
                "blocks": blocks_by_page.get(row["id"], []),
            }
        )

    return {
        "id": parse_row["id"],
        "normalized_text": parse_row["normalized_text"],
        "parser_name": parse_row["parser_name"],
        "parser_version": parse_row["parser_version"],
        "metadata": json.loads(parse_row["document_metadata_json"]),
        "warnings": json.loads(parse_row["warnings_json"]),
        "pages": pages,
    }
