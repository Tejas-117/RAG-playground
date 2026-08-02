class UnsupportedFileTypeError(ValueError):
    """Raised when ingestion cannot find a parser for an uploaded file type."""

    def __init__(self, mime_type: str | None, extension: str) -> None:
        """Build an unsupported file type error with detection details.

        Args:
            mime_type: MIME type reported by upload handling or guessed locally.
            extension: Lowercase file extension from the uploaded path.

        Returns:
            None. The initialized exception carries a user-readable message.
        """
        self.mime_type = mime_type
        self.extension = extension
        super().__init__(
            f"No parser registered for mime_type={mime_type!r}, extension={extension!r}"
        )


class ParserDependencyError(RuntimeError):
    """Raised when a selected parser needs an optional package that is missing."""

    def __init__(self, parser_name: str, package_name: str) -> None:
        """Build a parser dependency error with the missing package name.

        Args:
            parser_name: Name of the parser adapter that could not run.
            package_name: Python package that must be installed for the parser.

        Returns:
            None. The initialized exception carries installation guidance.
        """
        self.parser_name = parser_name
        self.package_name = package_name
        super().__init__(
            f"{parser_name} requires the optional package {package_name!r}"
        )
