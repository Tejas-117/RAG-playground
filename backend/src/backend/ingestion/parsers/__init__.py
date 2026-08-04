"""Parser interfaces and adapters for ingestion document extraction."""

from backend.ingestion.parsers.base import DocumentParser
from backend.ingestion.parsers.docx import DocxParser
from backend.ingestion.parsers.errors import (
    ParserDependencyError,
    UnsupportedFileTypeError,
)
from backend.ingestion.parsers.models import ParsedBlock, ParsedDocument, ParsedPage
from backend.ingestion.parsers.pdf import PyMuPDFParser
from backend.ingestion.parsers.pptx import PptxParser
from backend.ingestion.parsers.registry import (
    ParserRegistry,
    build_default_parser_registry,
)
from backend.ingestion.parsers.text import MarkdownParser, PlainTextParser

__all__ = [
    "DocumentParser",
    "DocxParser",
    "MarkdownParser",
    "ParsedBlock",
    "ParsedDocument",
    "ParsedPage",
    "ParserDependencyError",
    "ParserRegistry",
    "PlainTextParser",
    "PptxParser",
    "PyMuPDFParser",
    "UnsupportedFileTypeError",
    "build_default_parser_registry",
]
