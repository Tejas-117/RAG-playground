"""Parser interfaces and adapters for ingestion document extraction."""

from backend.ingestion.parser.base import DocumentParser
from backend.ingestion.parser.docx import DocxParser
from backend.ingestion.parser.errors import (
    ParserDependencyError,
    UnsupportedFileTypeError,
)
from backend.ingestion.parser.models import ParsedBlock, ParsedDocument, ParsedPage
from backend.ingestion.parser.pdf import PyMuPDFParser
from backend.ingestion.parser.pptx import PptxParser
from backend.ingestion.parser.registry import (
    ParserRegistry,
    build_default_parser_registry,
)
from backend.ingestion.parser.text import MarkdownParser, PlainTextParser

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
