"""Tests for canonical parsed-document text and source offsets."""

import pytest

from backend.ingestion.canonicalization import (
    ParsedDocumentValidationError,
    canonicalize_parsed_document,
)
from backend.ingestion.parsers.models import ParsedBlock, ParsedDocument, ParsedPage


def test_canonicalization_preserves_newlines_and_tracks_exact_offsets() -> None:
    """Verify separators do not replace internal block or paragraph newlines.

    Args:
        None.

    Returns:
        None. Assertions verify canonical text and end-exclusive offsets.
    """
    # Build page-aware parser output with both internal and canonical separators.
    parsed = ParsedDocument(
        text="ignored parser aggregate",
        pages=[
            ParsedPage(
                page_number=1,
                text="ignored page aggregate",
                blocks=[
                    ParsedBlock("First paragraph\ncontinued", block_index=4),
                    ParsedBlock("Second paragraph", block_index=8),
                ],
            ),
            ParsedPage(
                page_number=2,
                text="Final page",
                blocks=[],
            ),
        ],
        parser_name="test_parser",
        parser_version="1.0",
    )

    canonical = canonicalize_parsed_document(parsed)

    # Verify original internal newlines survive and neutral separators are inserted.
    assert canonical.normalized_text == (
        "First paragraph\ncontinued\n\nSecond paragraph\n\nFinal page"
    )
    first_block = canonical.pages[0].blocks[0]
    second_block = canonical.pages[0].blocks[1]
    assert (
        canonical.normalized_text[
            first_block.character_start_offset : first_block.character_end_offset
        ]
        == "First paragraph\ncontinued"
    )
    assert (
        canonical.normalized_text[
            second_block.character_start_offset : second_block.character_end_offset
        ]
        == "Second paragraph"
    )
    assert canonical.pages[1].character_start_offset == 45


def test_canonicalization_keeps_empty_page_without_extra_separator() -> None:
    """Verify empty page provenance is retained without changing canonical text.

    Args:
        None.

    Returns:
        None. Assertions verify empty-page and later-page offsets.
    """
    # Place an empty physical page between two readable pages.
    parsed = ParsedDocument(
        text="First\n\nLast",
        pages=[
            ParsedPage(page_number=1, text="First"),
            ParsedPage(page_number=2, text=""),
            ParsedPage(page_number=3, text="Last"),
        ],
        warnings=["Page 2 did not return selectable text"],
        parser_name="test_parser",
        parser_version="1.0",
    )

    canonical = canonicalize_parsed_document(parsed)

    # Page rows, rather than separator count, remain the boundary authority.
    assert canonical.normalized_text == "First\n\nLast"
    assert canonical.pages[1].character_start_offset == 5
    assert canonical.pages[1].character_end_offset == 5
    assert canonical.pages[2].character_start_offset == 7


def test_canonicalization_rejects_whitespace_only_document() -> None:
    """Verify documents without retrievable text receive a structured error.

    Args:
        None.

    Returns:
        None. Assertions verify the stable error code.
    """
    # Canonicalize a parser result that contains only whitespace.
    with pytest.raises(ParsedDocumentValidationError) as error_info:
        canonicalize_parsed_document(
            ParsedDocument(
                text=" \n\t",
                parser_name="test_parser",
                parser_version="1.0",
            )
        )

    assert error_info.value.code == "empty_parsed_document"
