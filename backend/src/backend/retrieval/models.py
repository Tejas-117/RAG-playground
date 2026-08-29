"""Immutable provider-neutral contracts for retrieved application chunks."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HydratedVectorSearchHit:
    """Combine one ranked vector hit with its persisted chunk provenance.

    Attributes:
        rank: One-based nearest-neighbor position returned by vector search.
        chunk_id: Stable application chunk identifier.
        raw_distance: Unmodified distance from the configured vector index.
        source_document_id: Stable document that produced the chunk.
        ordinal: Zero-based chunk position within its source document.
        text: Exact text supplied to the document embedding model.
        character_start_offset: Inclusive character offset in canonical text.
        character_end_offset: Exclusive character offset in canonical text.
        token_start_offset: Inclusive token offset in canonical text.
        token_end_offset: Exclusive token offset in canonical text.
        page_start: First physical page intersected by the chunk when available.
        page_end: Last physical page intersected by the chunk when available.
        section_path: Optional logical heading hierarchy when available.
        source_metadata: Parser, document, and source-block provenance.
    """

    rank: int
    chunk_id: str
    raw_distance: float
    source_document_id: str
    ordinal: int
    text: str
    character_start_offset: int | None
    character_end_offset: int | None
    token_start_offset: int | None
    token_end_offset: int | None
    page_start: int | None
    page_end: int | None
    section_path: list[str] | None
    source_metadata: dict[str, Any]
