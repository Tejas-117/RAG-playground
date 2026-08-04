from pathlib import Path
from typing import Protocol

from backend.ingestion.parsers.models import ParsedDocument


class DocumentParser(Protocol):
    """Common interface implemented by every ingestion parser adapter."""

    parser_name: str
    supported_extensions: frozenset[str]
    supported_mime_types: frozenset[str]

    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a local file into the normalized ingestion document shape.

        Args:
            file_path: Absolute or relative path to the file stored by ingestion.

        Returns:
            A ParsedDocument containing normalized text plus optional page/block
            metadata that downstream chunking can use for provenance.
        """
        ...
