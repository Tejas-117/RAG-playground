"""Orchestrate deterministic chunk-set construction, reuse, and persistence."""

import hashlib
import json
import logging
import sqlite3
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import pairwise
from time import perf_counter
from typing import Any
from uuid import UUID, uuid5

from backend.db.repositories.chunk_sets import (
    get_ready_chunk_set,
    load_corpus_chunking_inputs,
    save_ready_chunk_set,
)
from backend.ingestion.chunkers.models import ChunkingTokenizer, ChunkSpan
from backend.ingestion.chunkers.strategies import MAX_CHUNK_CHARACTERS, get_chunker
from backend.ingestion.chunkers.tokenizer import get_chunking_tokenizer
from backend.pipeline.configs import ChunkingConfig

# Use the module name to identify chunk-set and per-document stage records.
logger = logging.getLogger(__name__)

# UUID5 combines this permanent application namespace with stable artifact inputs
# to create deterministic chunk-set and chunk IDs. It must remain hardcoded and
# unchanged so identical inputs keep the same IDs across processes and installations.
_ARTIFACT_NAMESPACE = UUID("cfce009a-7f87-4a7c-86aa-76f2f613e0b4")


@dataclass(frozen=True)
class ChunkSetBuildResult:
    """Return one ready chunk artifact and how this request obtained it.

    Attributes:
        artifact: Complete persisted chunk-set artifact and ordered chunks.
        reused: Whether an already-ready artifact satisfied this request.
    """

    artifact: dict[str, Any]
    reused: bool


@dataclass(frozen=True)
class _SourceRangeIndex:
    """Index ordered parse ranges for repeated chunk-provenance lookups.

    Attributes:
        records: Source page or block dictionaries in their persisted order.
        end_offsets: Exclusive character ends matching the records by position.
        supports_binary_search: Whether the ranges are ordered and non-overlapping.
    """

    records: tuple[dict[str, Any], ...]
    end_offsets: tuple[int, ...]
    supports_binary_search: bool


class ChunkingCorpusNotFoundError(LookupError):
    """Report that chunking references a corpus that does not exist."""


class EmptyChunkingCorpusError(ValueError):
    """Report that a known corpus has no source documents to chunk."""


class MissingParseArtifactError(ValueError):
    """Report source documents that do not have canonical parse artifacts."""

    def __init__(self, document_ids: list[str]) -> None:
        """Store the stable identifiers of every unparsed document.

        Args:
            document_ids: Ordered document identifiers missing parse artifacts.

        Returns:
            None. The exception exposes the identifiers for structured handling.
        """
        self.document_ids = document_ids
        super().__init__(
            "Canonical parse artifacts are missing for: " + ", ".join(document_ids)
        )


def build_or_reuse_chunk_set(
    corpus_id: str,
    config: ChunkingConfig,
    tokenizer: ChunkingTokenizer | None = None,
) -> ChunkSetBuildResult:
    """Build or reuse the complete chunk artifact for one immutable corpus.

    Args:
        corpus_id: Stable corpus identifier whose documents should be chunked.
        config: Validated chunking configuration with resolved defaults.
        tokenizer: Optional injected tokenizer for deterministic unit tests.

    Returns:
        Ready chunk-set artifact together with whether it was reused.

    Raises:
        ChunkingCorpusNotFoundError: If the corpus does not exist.
        EmptyChunkingCorpusError: If the corpus contains no documents.
        MissingParseArtifactError: If any document lacks canonical text.
    """
    logger.info(
        "chunk_set_requested corpus_id=%s strategy=%s chunk_size_tokens=%d "
        "chunk_overlap_tokens=%d",
        corpus_id,
        config.strategy.value,
        config.chunk_size_tokens,
        config.chunk_overlap_tokens or 0,
    )
    corpus = load_corpus_chunking_inputs(corpus_id)

    # Fail with a domain-specific error before tokenizer or persistence work begins.
    if corpus is None:
        logger.warning(
            "chunk_set_rejected corpus_id=%s error_code=corpus_not_found",
            corpus_id,
        )
        raise ChunkingCorpusNotFoundError(corpus_id)

    documents = corpus["documents"]

    # A reusable artifact cannot represent an input set with no documents.
    if not documents:
        logger.warning(
            "chunk_set_rejected corpus_id=%s error_code=empty_corpus",
            corpus_id,
        )
        raise EmptyChunkingCorpusError(corpus_id)

    missing_parse_ids = [
        document["id"] for document in documents if document["parse_id"] is None
    ]

    # Report every missing parse together so the caller can repair the corpus once.
    if missing_parse_ids:
        logger.warning(
            "chunk_set_rejected corpus_id=%s error_code=missing_parse_artifact "
            "missing_document_count=%d",
            corpus_id,
            len(missing_parse_ids),
        )
        raise MissingParseArtifactError(missing_parse_ids)

    resolved_tokenizer = tokenizer or get_chunking_tokenizer()
    chunker = get_chunker(config.strategy)
    fingerprint = _build_fingerprint(
        corpus_id,
        documents,
        config,
        chunker.name,
        chunker.version,
        resolved_tokenizer,
    )

    existing = get_ready_chunk_set(fingerprint)

    # Reuse an identical complete artifact without rerunning any chunking strategy.
    if existing is not None:
        logger.info(
            "chunk_set_reused corpus_id=%s chunk_set_id=%s chunk_count=%d",
            corpus_id,
            existing["id"],
            existing["chunk_count"],
        )
        return ChunkSetBuildResult(artifact=existing, reused=True)

    started_counter = perf_counter()
    started_at = _utc_timestamp()
    chunk_set_id = str(uuid5(_ARTIFACT_NAMESPACE, f"chunk-set:{fingerprint}"))
    chunks: list[dict[str, Any]] = []
    logger.info(
        "chunk_set_build_started corpus_id=%s chunk_set_id=%s "
        "document_count=%d strategy=%s",
        corpus_id,
        chunk_set_id,
        len(documents),
        chunker.name,
    )

    # Chunk each document independently so overlap never crosses a source boundary.
    for document in documents:
        document_started_counter = perf_counter()
        logger.info(
            "document_chunking_started corpus_id=%s chunk_set_id=%s document_id=%s "
            "character_count=%d page_count=%d block_count=%d",
            corpus_id,
            chunk_set_id,
            document["id"],
            len(document["normalized_text"]),
            len(document["pages"]),
            len(document["blocks"]),
        )
        spans = chunker.chunk(document["normalized_text"], config, resolved_tokenizer)

        # Build each provenance index once per document. Every generated chunk reuses
        # these indexes instead of scanning all pages and blocks from the beginning.
        page_index = _build_source_range_index(document["pages"])
        block_index = _build_source_range_index(document["blocks"])

        # Reset ordinals for each document while keeping globally stable chunk IDs.
        for ordinal, span in enumerate(spans):
            chunks.append(
                _build_persistable_chunk(
                    chunk_set_id,
                    document,
                    ordinal,
                    span,
                    page_index,
                    block_index,
                )
            )

        document_duration_ms = max(
            0,
            round((perf_counter() - document_started_counter) * 1000),
        )
        logger.info(
            "document_chunking_completed corpus_id=%s chunk_set_id=%s "
            "document_id=%s chunk_count=%d duration_ms=%d",
            corpus_id,
            chunk_set_id,
            document["id"],
            len(spans),
            document_duration_ms,
        )

    completed_at = _utc_timestamp()
    duration_ms = max(0, round((perf_counter() - started_counter) * 1000))
    configuration_json = _canonical_json(config.model_dump(mode="json"))
    chunk_set = {
        "id": chunk_set_id,
        "corpus_id": corpus_id,
        "fingerprint": fingerprint,
        "chunking_config_json": configuration_json,
        "chunker_name": chunker.name,
        "chunker_version": chunker.version,
        "created_at": started_at,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_ms": duration_ms,
    }
    try:
        # Persist the parent and all children atomically for the normal build path.
        save_ready_chunk_set(chunk_set, chunks)
    except sqlite3.IntegrityError:
        # Concurrent identical requests can both miss the initial lookup. If another
        # request committed this exact fingerprint first, use its complete artifact.
        concurrently_persisted = get_ready_chunk_set(fingerprint)

        # Only convert a uniqueness race into reuse when the expected artifact exists.
        if concurrently_persisted is not None:
            logger.info(
                "chunk_set_reused_after_race corpus_id=%s chunk_set_id=%s "
                "chunk_count=%d",
                corpus_id,
                concurrently_persisted["id"],
                concurrently_persisted["chunk_count"],
            )
            return ChunkSetBuildResult(
                artifact=concurrently_persisted,
                reused=True,
            )

        # Preserve unrelated integrity failures such as invalid child provenance.
        logger.exception(
            "chunk_set_persistence_failed corpus_id=%s chunk_set_id=%s",
            corpus_id,
            chunk_set_id,
        )
        raise

    # Read through the repository boundary so new and reused results share one shape.
    persisted = get_ready_chunk_set(fingerprint)
    if persisted is None:
        logger.error(
            "chunk_set_readback_failed corpus_id=%s chunk_set_id=%s",
            corpus_id,
            chunk_set_id,
        )
        raise RuntimeError("The ready chunk set could not be read after persistence.")

    logger.info(
        "chunk_set_completed corpus_id=%s chunk_set_id=%s chunk_count=%d "
        "duration_ms=%d",
        corpus_id,
        chunk_set_id,
        persisted["chunk_count"],
        duration_ms,
    )
    return ChunkSetBuildResult(artifact=persisted, reused=False)


def _build_fingerprint(
    corpus_id: str,
    documents: list[dict[str, Any]],
    config: ChunkingConfig,
    chunker_name: str,
    chunker_version: str,
    tokenizer: ChunkingTokenizer,
) -> str:
    """Hash every immutable input that can change chunk boundaries.

    Args:
        corpus_id: Stable owning corpus identifier.
        documents: Ordered source and canonical parse identities.
        config: Fully resolved chunking configuration.
        chunker_name: Selected strategy implementation name.
        chunker_version: Boundary-algorithm implementation version.
        tokenizer: Tokenizer identity and immutable asset digest.

    Returns:
        Lowercase SHA-256 hexadecimal compatibility fingerprint.
    """
    payload = {
        "corpus_id": corpus_id,
        "documents": [
            {
                "document_id": document["id"],
                "parse_id": document["parse_id"],
            }
            for document in documents
        ],
        "chunking_config": config.model_dump(mode="json"),
        "chunker": {"name": chunker_name, "version": chunker_version},
        "tokenizer": {
            "identifier": tokenizer.identifier,
            "revision": tokenizer.revision,
            "asset_sha256": tokenizer.asset_sha256,
            "special_tokens_policy": tokenizer.special_tokens_policy,
        },
        "max_chunk_characters": MAX_CHUNK_CHARACTERS,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _build_persistable_chunk(
    chunk_set_id: str,
    document: dict[str, Any],
    ordinal: int,
    span: ChunkSpan,
    page_index: _SourceRangeIndex,
    block_index: _SourceRangeIndex,
) -> dict[str, Any]:
    """Attach deterministic identity and source provenance to one chunk span.

    Args:
        chunk_set_id: Stable identifier of the owning reusable artifact.
        document: Canonical source document and offset metadata.
        ordinal: Zero-based position within this source document.
        span: Exact canonical-text slice returned by a chunker.
        page_index: Reusable index of the document's canonical page ranges.
        block_index: Reusable index of the document's canonical block ranges.

    Returns:
        Dictionary ready for atomic SQLite persistence.
    """
    # Find every parsed page whose canonical character range overlaps this chunk.
    # A chunk can cross a page boundary because chunkers follow token or paragraph
    # limits, not page boundaries. These pages become the persisted citation range.
    intersecting_pages = _find_intersecting_ranges(
        page_index,
        span.character_start_offset,
        span.character_end_offset,
    )
    # Find every parsed source block whose canonical character range overlaps this
    # chunk. Their ordinals preserve block-level provenance without copying block text.
    intersecting_blocks = [
        block["ordinal"]
        for block in _find_intersecting_ranges(
            block_index,
            span.character_start_offset,
            span.character_end_offset,
        )
    ]
    metadata = {
        "parse_id": document["parse_id"],
        "parser": {
            "name": document["parser_name"],
            "version": document["parser_version"],
        },
        "document": {
            "original_filename": document["original_filename"],
            "mime_type": document["mime_type"],
            "content_sha256": document["content_sha256"],
        },
        "parse_metadata": document["parse_metadata"],
        "block_ordinals": intersecting_blocks,
    }
    chunk_id = str(
        uuid5(
            _ARTIFACT_NAMESPACE,
            f"chunk:{chunk_set_id}:{document['id']}:{ordinal}",
        )
    )
    return {
        "id": chunk_id,
        "source_document_id": document["id"],
        "ordinal": ordinal,
        "text": span.text,
        "character_start_offset": span.character_start_offset,
        "character_end_offset": span.character_end_offset,
        "token_start_offset": span.token_start_offset,
        "token_end_offset": span.token_end_offset,
        "page_start": intersecting_pages[0]["page_number"]
        if intersecting_pages
        else None,
        "page_end": intersecting_pages[-1]["page_number"]
        if intersecting_pages
        else None,
        "source_metadata_json": _canonical_json(metadata),
    }


def _build_source_range_index(
    source_ranges: list[dict[str, Any]],
) -> _SourceRangeIndex:
    """Prepare one ordered page or block collection for repeated range queries.

    Args:
        source_ranges: Persisted page or block dictionaries in canonical source order.

    Returns:
        Immutable records, end offsets, and whether binary search is safe to use.
    """
    records = tuple(source_ranges)
    end_offsets = tuple(record["character_end_offset"] for record in records)

    # Canonical pages and blocks are normally ordered, disjoint ranges. Verify that
    # invariant once so unexpected legacy data can safely use the exact full-scan path.
    valid_ranges = all(
        record["character_start_offset"] <= record["character_end_offset"]
        for record in records
    )
    ordered_ranges = all(
        previous["character_end_offset"] <= current["character_start_offset"]
        for previous, current in pairwise(records)
    )
    supports_binary_search = valid_ranges and ordered_ranges
    return _SourceRangeIndex(
        records=records,
        end_offsets=end_offsets,
        supports_binary_search=supports_binary_search,
    )


def _find_intersecting_ranges(
    source_index: _SourceRangeIndex,
    chunk_start: int,
    chunk_end: int,
) -> list[dict[str, Any]]:
    """Return source ranges intersecting one half-open chunk character interval.

    Args:
        source_index: Reusable page or block range index for one document.
        chunk_start: Inclusive character start of the generated chunk.
        chunk_end: Exclusive character end of the generated chunk.

    Returns:
        Intersecting source records in their original persisted order.
    """
    # Unexpected overlapping or out-of-order legacy ranges cannot use a single binary
    # search safely. Retain the previous full-scan semantics for those documents.
    if not source_index.supports_binary_search:
        return [
            record
            for record in source_index.records
            if _ranges_intersect(
                chunk_start,
                chunk_end,
                record["character_start_offset"],
                record["character_end_offset"],
            )
        ]

    # Ranges ending at chunk_start do not intersect a half-open chunk, so bisect_right
    # skips them and positions the scan at the first possible overlapping source range.
    first_candidate = bisect_right(source_index.end_offsets, chunk_start)
    intersections: list[dict[str, Any]] = []

    # Only inspect nearby ranges. Ordered ranges after chunk_end cannot intersect and
    # let the loop stop without visiting the remainder of a large document's metadata.
    for record_index in range(first_candidate, len(source_index.records)):
        record = source_index.records[record_index]

        # Ordered source starts permit an immediate stop after the chunk's exclusive
        # end; no later record can move backward into the chunk.
        if record["character_start_offset"] >= chunk_end:
            break

        # The binary search and stop boundary normally imply intersection. Keep the
        # shared predicate here so zero-width or unusual legacy ranges remain exact.
        if _ranges_intersect(
            chunk_start,
            chunk_end,
            record["character_start_offset"],
            record["character_end_offset"],
        ):
            intersections.append(record)

    return intersections


def _ranges_intersect(
    first_start: int,
    first_end: int,
    second_start: int,
    second_end: int,
) -> bool:
    """Test whether two half-open source ranges share at least one character.

    Args:
        first_start: Inclusive start of the first range.
        first_end: Exclusive end of the first range.
        second_start: Inclusive start of the second range.
        second_end: Exclusive end of the second range.

    Returns:
        True when the half-open ranges overlap.
    """
    # Half-open comparison excludes pages or blocks touching only at a boundary.
    return first_start < second_end and second_start < first_end


def _canonical_json(value: Any) -> str:
    """Serialize a value deterministically for fingerprints and SQLite.

    Args:
        value: JSON-compatible configuration or provenance value.

    Returns:
        Compact JSON with stable key ordering and Unicode preserved.
    """
    # Stable serialization makes fingerprints independent of dictionary insertion order.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _utc_timestamp() -> str:
    """Create the UTC timestamp format used by persisted artifacts.

    Args:
        None.

    Returns:
        ISO-8601 UTC timestamp ending in ``Z``.
    """
    # Timezone-aware values prevent local timezone ambiguity in artifact history.
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace(
            "+00:00",
            "Z",
        )
    )
