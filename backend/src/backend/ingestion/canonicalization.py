"""Build canonical persisted text and source offsets from parser output."""

import json
from dataclasses import dataclass, field
from typing import Any

from backend.ingestion.parsers.models import ParsedDocument

# Bound the expanded parsed text independently from the compressed upload size.
MAX_NORMALIZED_TEXT_SIZE_BYTES = 50 * 1024 * 1024
CANONICAL_SEPARATOR = "\n\n"


class ParsedDocumentValidationError(ValueError):
    """Raised when parser output cannot become a safe persisted artifact."""

    def __init__(self, code: str, message: str) -> None:
        """Create a structured canonicalization error.

        Args:
            code: Stable machine-readable failure code.
            message: Safe human-readable failure explanation.

        Returns:
            None. The exception retains both values for API translation.
        """
        # Retain structured fields so the transport layer does not parse text.
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class CanonicalBlock:
    """Offsets and source metadata for one canonical text block."""

    ordinal: int
    page_number: int
    source_block_index: int | None
    character_start_offset: int
    character_end_offset: int
    bounding_box: tuple[float, float, float, float] | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalPage:
    """Offsets and source metadata for one physical or logical page."""

    page_number: int
    character_start_offset: int
    character_end_offset: int
    metadata: dict[str, Any] = field(default_factory=dict)
    blocks: list[CanonicalBlock] = field(default_factory=list)


@dataclass(frozen=True)
class CanonicalDocument:
    """A complete normalized document ready for transactional persistence."""

    normalized_text: str
    parser_name: str
    parser_version: str
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    pages: list[CanonicalPage] = field(default_factory=list)

    @property
    def utf8_size_bytes(self) -> int:
        """Return the encoded storage size of normalized text.

        Args:
            None.

        Returns:
            The UTF-8 byte length of the canonical text.
        """
        return len(self.normalized_text.encode("utf-8"))

    @property
    def block_count(self) -> int:
        """Return the total number of canonical blocks.

        Args:
            None.

        Returns:
            The number of blocks across all stored pages.
        """
        return sum(len(page.blocks) for page in self.pages)


def canonicalize_parsed_document(parsed: ParsedDocument) -> CanonicalDocument:
    """Create canonical text plus page and block offsets from parser output.

    Args:
        parsed: Transient output returned by a format-specific parser.

    Returns:
        A validated canonical document suitable for SQLite persistence.

    Raises:
        ParsedDocumentValidationError: If text or parser identity is unusable.
    """
    # Require parser identity because it is part of artifact provenance.
    if not parsed.parser_name.strip() or not parsed.parser_version.strip():
        raise ParsedDocumentValidationError(
            "invalid_parser_output",
            "The parser did not provide its name and version.",
        )

    # Page-aware parsers use their blocks as the canonical reading-order text.
    if parsed.pages:
        normalized_text, pages = _canonicalize_pages(parsed)
    else:
        normalized_text = parsed.text
        pages = []

    # Whitespace-only extraction is unusable for retrieval and chunking.
    if not normalized_text.strip():
        raise ParsedDocumentValidationError(
            "empty_parsed_document",
            "The document did not contain any extractable text.",
        )

    utf8_size_bytes = len(normalized_text.encode("utf-8"))

    # Reject expanded parser output that exceeds the explicit storage boundary.
    if utf8_size_bytes > MAX_NORMALIZED_TEXT_SIZE_BYTES:
        raise ParsedDocumentValidationError(
            "parsed_document_too_large",
            "The extracted document text exceeds the 50 MiB limit.",
        )

    # Validate every metadata value before a database transaction is attempted.
    try:
        json.dumps(parsed.metadata, allow_nan=False)
        json.dumps(parsed.warnings, allow_nan=False)
        for page in pages:
            json.dumps(page.metadata, allow_nan=False)
            for block in page.blocks:
                json.dumps(block.metadata, allow_nan=False)
                json.dumps(block.bounding_box, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ParsedDocumentValidationError(
            "invalid_parser_output",
            "The parser returned metadata that cannot be stored as JSON.",
        ) from error

    return CanonicalDocument(
        normalized_text=normalized_text,
        parser_name=parsed.parser_name,
        parser_version=parsed.parser_version,
        metadata=parsed.metadata,
        warnings=parsed.warnings,
        pages=pages,
    )


def _canonicalize_pages(
    parsed: ParsedDocument,
) -> tuple[str, list[CanonicalPage]]:
    """Join page content and calculate authoritative source offsets.

    Args:
        parsed: Parser output containing physical or logical pages.

    Returns:
        Canonical document text and page records with nested block offsets.
    """
    text_parts: list[str] = []
    canonical_pages: list[CanonicalPage] = []
    character_cursor = 0
    block_ordinal = 0
    has_content = False

    # Preserve every page record, including empty pages that carry warnings.
    for page in parsed.pages:
        page_text, block_specs = _build_page_text(page.blocks, page.text)

        # Add a neutral separator only between non-empty canonical page contents.
        if page_text and has_content:
            text_parts.append(CANONICAL_SEPARATOR)
            character_cursor += len(CANONICAL_SEPARATOR)

        page_start = character_cursor
        text_parts.append(page_text)
        character_cursor += len(page_text)
        canonical_blocks: list[CanonicalBlock] = []

        # Translate page-local block locations into document-global offsets.
        for block, local_start, local_end in block_specs:
            canonical_blocks.append(
                CanonicalBlock(
                    ordinal=block_ordinal,
                    page_number=page.page_number,
                    source_block_index=block.block_index,
                    character_start_offset=page_start + local_start,
                    character_end_offset=page_start + local_end,
                    bounding_box=block.bbox,
                    metadata=block.metadata,
                )
            )
            block_ordinal += 1

        canonical_pages.append(
            CanonicalPage(
                page_number=page.page_number,
                character_start_offset=page_start,
                character_end_offset=character_cursor,
                metadata=page.metadata,
                blocks=canonical_blocks,
            )
        )

        # Empty pages do not cause consecutive separators in normalized text.
        if page_text:
            has_content = True

    return "".join(text_parts), canonical_pages


def _build_page_text(
    blocks: list[Any],
    fallback_text: str,
) -> tuple[str, list[tuple[Any, int, int]]]:
    """Build one page's canonical text and block-local offsets.

    Args:
        blocks: Ordered source blocks returned for the page.
        fallback_text: Page text used when no readable blocks exist.

    Returns:
        Page text and tuples containing blocks with local start/end offsets.
    """
    readable_blocks = [block for block in blocks if block.text.strip()]

    # Preserve parser page text when no source-aware blocks were produced.
    if not readable_blocks:
        return fallback_text, []

    parts: list[str] = []
    offsets: list[tuple[Any, int, int]] = []
    cursor = 0

    # Join blocks with a neutral separator while retaining internal newlines.
    for block in readable_blocks:
        if parts:
            parts.append(CANONICAL_SEPARATOR)
            cursor += len(CANONICAL_SEPARATOR)

        text = block.text.strip()
        start = cursor
        parts.append(text)
        cursor += len(text)
        offsets.append((block, start, cursor))

    return "".join(parts), offsets
