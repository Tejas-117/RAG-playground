"""Hydrate ranked vector hits from immutable SQLite chunk records."""

from collections.abc import Callable
from typing import Any

from backend.db.repositories.chunk_sets import load_chunks_by_ids
from backend.embedding.models import VectorSearchHit
from backend.retrieval.models import HydratedVectorSearchHit

# A loader resolves selected IDs only within one exact reusable chunk artifact.
ChunkLoader = Callable[
    [str, tuple[str, ...]],
    dict[str, dict[str, Any]],
]


class ChunkHydrationError(RuntimeError):
    """Report vector hits that cannot map safely to the expected chunk set."""


def hydrate_vector_search_hits(
    hits: tuple[VectorSearchHit, ...],
    chunk_set_id: str,
    chunk_loader: ChunkLoader = load_chunks_by_ids,
) -> tuple[HydratedVectorSearchHit, ...]:
    """Attach persisted text and source provenance to ranked vector hits.

    Args:
        hits: Ranked lightweight matches returned by the vector-store adapter.
        chunk_set_id: Exact chunk artifact used to build the searched index.
        chunk_loader: Injectable batched persistence boundary for selected chunks.

    Returns:
        Hydrated immutable hits in the vector store's original ranking order.

    Raises:
        ChunkHydrationError: If IDs are duplicated, missing, or from another set.
    """
    # An empty nearest-neighbor result needs no artifact identity or persistence work.
    if not hits:
        return ()

    normalized_chunk_set_id = chunk_set_id.strip()

    # A blank artifact identity cannot safely scope the persistence lookup.
    if not normalized_chunk_set_id:
        raise ChunkHydrationError("The expected chunk-set identifier is blank.")

    chunk_ids = tuple(hit.chunk_id for hit in hits)

    # Duplicate IDs would make ranking and later result persistence ambiguous.
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ChunkHydrationError("Vector search returned duplicate chunk identifiers.")

    chunks_by_id = chunk_loader(normalized_chunk_set_id, chunk_ids)
    missing_chunk_ids = tuple(
        chunk_id for chunk_id in chunk_ids if chunk_id not in chunks_by_id
    )

    # The scoped loader treats absent and foreign-set chunks as incompatible hits.
    if missing_chunk_ids:
        raise ChunkHydrationError(
            "One or more vector hits do not belong to the expected chunk set."
        )

    hydrated_hits: list[HydratedVectorSearchHit] = []

    # Rebuild from vector-hit order because SQL does not preserve IN-clause order.
    for rank, hit in enumerate(hits, start=1):
        chunk = chunks_by_id[hit.chunk_id]
        hydrated_hits.append(
            HydratedVectorSearchHit(
                rank=rank,
                chunk_id=hit.chunk_id,
                raw_distance=hit.raw_distance,
                source_document_id=chunk["source_document_id"],
                ordinal=chunk["ordinal"],
                text=chunk["text"],
                character_start_offset=chunk["character_start_offset"],
                character_end_offset=chunk["character_end_offset"],
                token_start_offset=chunk["token_start_offset"],
                token_end_offset=chunk["token_end_offset"],
                page_start=chunk["page_start"],
                page_end=chunk["page_end"],
                section_path=chunk["section_path"],
                source_metadata=chunk["source_metadata"],
            )
        )

    return tuple(hydrated_hits)
