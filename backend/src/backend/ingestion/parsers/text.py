from importlib import metadata
from pathlib import Path

from backend.ingestion.parsers.models import ParsedDocument


class PlainTextParser:
    """Plain text parser for simple UTF-8 compatible files."""

    parser_name = "plain_text"
    supported_extensions = frozenset({".txt"})
    supported_mime_types = frozenset({"text/plain"})

    def parse(self, file_path: Path) -> ParsedDocument:
        """Read a plain text file into the normalized parser output shape.

        Args:
            file_path: Absolute or relative path to a text file stored by ingestion.

        Returns:
            A ParsedDocument containing the file text and parser metadata.
        """
        text = file_path.read_text(encoding="utf-8")

        return ParsedDocument(
            text=text,
            metadata={"source_path": str(file_path)},
            parser_name=self.parser_name,
            parser_version=self._get_parser_version(),
        )

    def _get_parser_version(self) -> str:
        """Read the package version that owns the plain text parser.

        Args:
            None.

        Returns:
            The backend package version when installed, otherwise "stdlib".
        """
        try:
            return metadata.version("backend")
        except metadata.PackageNotFoundError:
            return "stdlib"


class MarkdownParser(PlainTextParser):
    """Markdown parser that preserves Markdown source text for later chunking."""

    parser_name = "markdown"
    supported_extensions = frozenset({".md", ".markdown"})
    supported_mime_types = frozenset({"text/markdown", "text/x-markdown"})
