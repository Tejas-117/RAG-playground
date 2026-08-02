"""Normalized parser output shared by every supported ingestion format.

ParsedDocument represents the complete result from one uploaded file. Its `text`
is the parser's full normalized text and its metadata identifies the source file
and parser that produced the result.

ParsedPage represents a page-like unit within that file. For PDF, EPUB, and MOBI,
it represents one page supplied by PyMuPDF. For PPTX, it represents one slide.
DOCX has no reliable page layout through python-docx, so it is represented as one
logical page. TXT and Markdown do not have pages and therefore return no pages.

ParsedBlock represents the smallest source-aware text unit emitted by a parser.
For PDF, EPUB, and MOBI, it is a positioned PyMuPDF text block with a bounding
box. For DOCX, it is a paragraph or table row. For PPTX, it is a text shape or
table row. TXT and Markdown currently expose their text directly on
ParsedDocument and do not create blocks.

The chunking stage consumes this normalized output to create stored document
chunks. These parser models are transient ingestion data and are not persisted as
database records in their current form.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParsedBlock:
    """A source-location-aware text unit produced by a parser."""

    text: str
    page_number: int | None = None
    block_index: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedPage:
    """A parsed page or page-like unit from a document."""

    page_number: int
    text: str
    blocks: list[ParsedBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDocument:
    """Normalized parser output consumed by chunking and indexing."""

    text: str
    pages: list[ParsedPage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    parser_name: str = ""
    parser_version: str = ""
