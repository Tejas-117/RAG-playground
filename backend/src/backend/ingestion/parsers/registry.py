from pathlib import Path

from backend.ingestion.parsers.base import DocumentParser
from backend.ingestion.parsers.docx import DocxParser
from backend.ingestion.parsers.errors import UnsupportedFileTypeError
from backend.ingestion.parsers.pdf import PyMuPDFParser
from backend.ingestion.parsers.pptx import PptxParser
from backend.ingestion.parsers.text import MarkdownParser, PlainTextParser


class ParserRegistry:
    """Lookup table that selects the correct parser for an uploaded file."""

    def __init__(self, parsers: list[DocumentParser]) -> None:
        """Create a registry from parser adapter instances.

        Args:
            parsers: Parser adapters available to ingestion.

        Returns:
            None. The registry stores parser instances for later lookup.
        """
        self._parsers = parsers

    def get_parser_by_extension(self, file_path: Path) -> DocumentParser:
        """Find the parser that should process a file extension.

        Args:
            file_path: Local path to the uploaded file.

        Returns:
            A parser adapter matching the file extension.

        Raises:
            UnsupportedFileTypeError: No registered parser supports the file.
        """
        extension = file_path.suffix.lower()

        # Select the parser only from the extension
        for parser in self._parsers:
            if extension in parser.supported_extensions:
                return parser

        # Raise a structured parser error when no extension adapter is registered.
        raise UnsupportedFileTypeError(extension=extension)


def build_default_parser_registry() -> ParserRegistry:
    """Build the parser registry used by the ingestion pipeline.

    Args:
        None.

    Returns:
        A ParserRegistry containing the parser adapters supported by the current
        local ingestion implementation.
    """
    return ParserRegistry(
        parsers=[
            PyMuPDFParser(),
            DocxParser(),
            PptxParser(),
            PlainTextParser(),
            MarkdownParser(),
        ]
    )
