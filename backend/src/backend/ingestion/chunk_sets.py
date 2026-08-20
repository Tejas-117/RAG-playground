"""Orchestrate deterministic chunk-set construction, reuse, and persistence."""

import hashlib
import json
from datetime import datetime, timezone
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

# UUID5 combines this permanent application namespace with stable artifact inputs
# to create deterministic chunk-set and chunk IDs. It must remain hardcoded and
# unchanged so identical inputs keep the same IDs across processes and installations.
_ARTIFACT_NAMESPACE = UUID("cfce009a-7f87-4a7c-86aa-76f2f613e0b4")


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
) -> dict[str, Any]:
    """Build or reuse the complete chunk artifact for one immutable corpus.

    Args:
        corpus_id: Stable corpus identifier whose documents should be chunked.
        config: Validated chunking configuration with resolved defaults.
        tokenizer: Optional injected tokenizer for deterministic unit tests.

    Returns:
        Ready chunk-set artifact with ordered persisted chunks.

    Raises:
        ChunkingCorpusNotFoundError: If the corpus does not exist.
        EmptyChunkingCorpusError: If the corpus contains no documents.
        MissingParseArtifactError: If any document lacks canonical text.
    """
    corpus = load_corpus_chunking_inputs(corpus_id)

    # Fail with a domain-specific error before tokenizer or persistence work begins.
    if corpus is None:
        raise ChunkingCorpusNotFoundError(corpus_id)

    documents = corpus["documents"]

    # A reusable artifact cannot represent an input set with no documents.
    if not documents:
        raise EmptyChunkingCorpusError(corpus_id)

    missing_parse_ids = [
        document["id"] for document in documents if document["parse_id"] is None
    ]

    # Report every missing parse together so the caller can repair the corpus once.
    if missing_parse_ids:
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
        return existing

    started_counter = perf_counter()
    started_at = _utc_timestamp()
    chunk_set_id = str(uuid5(_ARTIFACT_NAMESPACE, f"chunk-set:{fingerprint}"))
    chunks: list[dict[str, Any]] = []

    # Chunk each document independently so overlap never crosses a source boundary.
    for document in documents:
        spans = chunker.chunk(document["normalized_text"], config, resolved_tokenizer)

        # Reset ordinals for each document while keeping globally stable chunk IDs.
        for ordinal, span in enumerate(spans):
            chunks.append(
                _build_persistable_chunk(
                    chunk_set_id,
                    document,
                    ordinal,
                    span,
                )
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
    save_ready_chunk_set(chunk_set, chunks)

    # Read through the repository boundary so new and reused results share one shape.
    persisted = get_ready_chunk_set(fingerprint)
    if persisted is None:
        raise RuntimeError("The ready chunk set could not be read after persistence.")
    return persisted


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
) -> dict[str, Any]:
    """Attach deterministic identity and source provenance to one chunk span.

    Args:
        chunk_set_id: Stable identifier of the owning reusable artifact.
        document: Canonical source document and offset metadata.
        ordinal: Zero-based position within this source document.
        span: Exact canonical-text slice returned by a chunker.

    Returns:
        Dictionary ready for atomic SQLite persistence.
    """
    # Find every parsed page whose canonical character range overlaps this chunk.
    # A chunk can cross a page boundary because chunkers follow token or paragraph
    # limits, not page boundaries. These pages become the persisted citation range.
    intersecting_pages = [
        page
        for page in document["pages"]
        if _ranges_intersect(
            span.character_start_offset,
            span.character_end_offset,
            page["character_start_offset"],
            page["character_end_offset"],
        )
    ]
    # Find every parsed source block whose canonical character range overlaps this
    # chunk. Their ordinals preserve block-level provenance without copying block text.
    intersecting_blocks = [
        block["ordinal"]
        for block in document["blocks"]
        if _ranges_intersect(
            span.character_start_offset,
            span.character_end_offset,
            block["character_start_offset"],
            block["character_end_offset"],
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
