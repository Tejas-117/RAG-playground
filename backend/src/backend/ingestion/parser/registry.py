from pathlib import Path

from backend.ingestion.parser.base import DocumentParser
from backend.ingestion.parser.docx import DocxParser
from backend.ingestion.parser.errors import UnsupportedFileTypeError
from backend.ingestion.parser.pdf import PyMuPDFParser
from backend.ingestion.parser.pptx import PptxParser
from backend.ingestion.parser.text import MarkdownParser, PlainTextParser


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

    def get_parser(
        self, file_path: Path, mime_type: str | None = None
    ) -> DocumentParser:
        """Find the parser that should process a file.

        Args:
            file_path: Local path to the uploaded file.
            mime_type: Optional MIME type reported by upload handling.

        Returns:
            A parser adapter matching the MIME type or file extension.

        Raises:
            UnsupportedFileTypeError: No registered parser supports the file.
        """
        extension = file_path.suffix.lower()

        # try to get the file type based on its mime type
        if mime_type:
            for parser in self._parsers:
                if mime_type in parser.supported_mime_types:
                    return parser

        # fallback to using file extension
        for parser in self._parsers:
            if extension in parser.supported_extensions:
                return parser

        # else raise error
        raise UnsupportedFileTypeError(mime_type=mime_type, extension=extension)


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
