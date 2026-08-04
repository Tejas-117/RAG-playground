from importlib import metadata
from pathlib import Path
from typing import Any

from backend.ingestion.parsers.errors import ParserDependencyError
from backend.ingestion.parsers.models import ParsedBlock, ParsedDocument, ParsedPage


class PyMuPDFParser:
    """Document parser adapter backed by PyMuPDF."""

    parser_name = "pymupdf"
    supported_extensions = frozenset({".pdf", ".epub", ".mobi"})
    supported_mime_types = frozenset(
        {
            "application/pdf",
            "application/epub+zip",
            "application/vnd.amazon.ebook",
            "application/x-mobipocket-ebook",
        }
    )

    def parse(self, file_path: Path) -> ParsedDocument:
        """Extract text and layout blocks from a PyMuPDF-supported file.

        Args:
            file_path: Absolute or relative path to a supported file stored by
                ingestion.

        Returns:
            A ParsedDocument with full document text, page text, page blocks, and
            parser metadata suitable for chunk provenance.
        """
        try:
            import fitz
        except ImportError as exc:
            raise ParserDependencyError(self.parser_name, "pymupdf") from exc

        pages: list[ParsedPage] = []
        warnings: list[str] = []

        with fitz.open(file_path) as document:
            for page_index in range(document.page_count):
                page = document[page_index]
                page_number = page_index + 1
                page_text = page.get_text("text")
                blocks = self._parse_blocks(page.get_text("blocks"), page_number)

                if not page_text.strip():
                    warnings.append(
                        f"Page {page_number} did not return selectable text"
                    )

                pages.append(
                    ParsedPage(
                        page_number=page_number,
                        text=page_text,
                        blocks=blocks,
                        metadata={
                            "width": page.rect.width,
                            "height": page.rect.height,
                        },
                    )
                )

        return ParsedDocument(
            text="\n\n".join(page.text for page in pages).strip(),
            pages=pages,
            metadata={"source_path": str(file_path)},
            warnings=warnings,
            parser_name=self.parser_name,
            parser_version=self._get_parser_version(),
        )

    def _parse_blocks(
        self,
        raw_blocks: list[tuple[Any, ...]],
        page_number: int,
    ) -> list[ParsedBlock]:
        """Convert PyMuPDF block tuples into normalized ParsedBlock objects.

        Args:
            raw_blocks: Block tuples returned by PyMuPDF page.get_text("blocks").
            page_number: One-based page number that owns the extracted blocks.

        Returns:
            A list of ParsedBlock instances with text, block index, and bounding
            box coordinates retained for source provenance.
        """
        parsed_blocks: list[ParsedBlock] = []

        for block_index, block in enumerate(raw_blocks):
            x0, y0, x1, y1, text, *_ = block
            normalized_text = str(text).strip()

            if not normalized_text:
                continue

            parsed_blocks.append(
                ParsedBlock(
                    text=normalized_text,
                    page_number=page_number,
                    block_index=block_index,
                    bbox=(float(x0), float(y0), float(x1), float(y1)),
                )
            )

        return parsed_blocks

    def _get_parser_version(self) -> str:
        """Read the installed PyMuPDF package version.

        Args:
            None.

        Returns:
            The installed PyMuPDF version, or "unknown" if package metadata is
            unavailable in the current environment.
        """
        try:
            return metadata.version("pymupdf")
        except metadata.PackageNotFoundError:
            return "unknown"
