import codecs
import hashlib
import sqlite3
import uuid
import zipfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.db.repositories.corpora import create_upload_batch
from backend.ingestion.parsers.base import DocumentParser
from backend.ingestion.parsers.errors import UnsupportedFileTypeError
from backend.ingestion.parsers.registry import (
    ParserRegistry,
    build_default_parser_registry,
)

router = APIRouter()

# Keep the limit in bytes so validation is exact and independent of display units.
MAX_FILE_SIZE_BYTES = 30 * 1024 * 1024
UPLOADS_DIRECTORY = Path(__file__).resolve().parents[4] / "uploads"
READ_CHUNK_SIZE_BYTES = 1024 * 1024


def _validate_filename(filename: str | None) -> str:
    """Validate and return a safe client-provided filename.

    Args:
        filename: Filename supplied in the multipart upload.

    Returns:
        The validated filename.

    Raises:
        HTTPException: If the filename is missing or contains a path component.
    """

    # Reject missing names and path components instead of silently rewriting them.
    if not filename or filename in {".", ".."} or Path(filename).name != filename:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_filename",
                "message": "Each uploaded file must have a plain filename.",
            },
        )

    return filename


async def _get_file_size(file: UploadFile) -> int:
    """Count an uploaded file without persisting it and rewind it for saving.

    Args:
        file: Multipart upload whose temporary body should be measured.

    Returns:
        The number of bytes in the uploaded file.
    """
    # Read in bounded chunks so validation does not duplicate the whole request in memory.
    size = 0
    while True:
        chunk = await file.read(READ_CHUNK_SIZE_BYTES)

        # Stop once the temporary upload body has been consumed.
        if not chunk:
            break

        size += len(chunk)

        # Stop early after the limit is exceeded; the file will not be saved.
        if size > MAX_FILE_SIZE_BYTES:
            break

    # Rewind the UploadFile so the already-validated body can be copied later.
    await file.seek(0)
    return size


async def _save_file(file: UploadFile, destination: Path) -> tuple[int, str]:
    """Copy one validated upload and calculate its content hash.

    Args:
        file: Multipart upload whose size and format have already been validated.
        destination: Exact UUID-based destination path for the stored file.

    Returns:
        The stored byte count and SHA-256 digest of the file contents.
    """
    # Hash while copying so the file is read only once after validation.
    digest = hashlib.sha256()
    size = 0
    with destination.open("wb") as output_file:
        while True:
            chunk = await file.read(READ_CHUNK_SIZE_BYTES)

            # Finish once the entire validated upload body has been copied.
            if not chunk:
                break

            output_file.write(chunk)
            digest.update(chunk)
            size += len(chunk)

    # Rewind so request cleanup remains predictable for callers and tests.
    await file.seek(0)
    return size, digest.hexdigest()


def _get_parser_for_filename(registry: ParserRegistry, filename: str) -> DocumentParser:
    """Resolve a parser from a validated filename.

    Args:
        registry: Registry containing the currently supported parser adapters.
        filename: Validated client-provided filename.

    Returns:
        The parser registered for the filename extension.

    Raises:
        HTTPException: If the extension is unsupported.
    """
    try:
        # Resolve a parser by extension
        return registry.get_parser_by_extension(Path(filename))
    except UnsupportedFileTypeError as error:
        # Expose a stable API error code without leaking a traceback to the client.
        raise HTTPException(
            status_code=415,
            detail={
                "code": "unsupported_file_type",
                "message": "This file format is not supported.",
                "extension": error.extension,
            },
        ) from error


async def _validate_file_content(
    file: UploadFile,
    filename: str,
    parser: DocumentParser,
) -> None:
    """Verify that the uploaded bytes match the selected document format.

    Args:
        file: Multipart upload whose bytes should be inspected.
        filename: Validated client-provided filename.
        parser: Parser selected from the filename extension and MIME type.

    Returns:
        None. The upload is rewound after content inspection.

    Raises:
        HTTPException: If the file bytes do not match the selected format.
    """
    extension = Path(filename).suffix.lower()
    is_valid = False

    # Inspect the first bytes for formats with a fixed file signature.
    header = await file.read(8)
    await file.seek(0)

    if extension == ".pdf":
        is_valid = header.startswith(b"%PDF-")
    elif extension in {".docx", ".pptx", ".epub"}:
        # OOXML and EPUB files are ZIP containers, but their internal files
        # distinguish the supported document format from an arbitrary ZIP archive.
        if header.startswith(b"PK"):
            try:
                with zipfile.ZipFile(file.file) as archive:
                    names = set(archive.namelist())
                    if extension == ".docx":
                        is_valid = {
                            "[Content_Types].xml",
                            "word/document.xml",
                        }.issubset(names)
                    elif extension == ".pptx":
                        is_valid = {
                            "[Content_Types].xml",
                            "ppt/presentation.xml",
                        }.issubset(names)
                    else:
                        is_valid = "mimetype" in names
            except (OSError, zipfile.BadZipFile):
                is_valid = False
        await file.seek(0)
    elif extension == ".mobi":
        # MOBI files identify themselves with BOOKMOBI at byte offset 60.
        header = await file.read(68)
        is_valid = len(header) >= 68 and header[60:68] == b"BOOKMOBI"
        await file.seek(0)
    elif extension in {".txt", ".md", ".markdown"}:
        # Decode text incrementally so binary files renamed as text are rejected
        # without loading the entire upload into memory at once.
        decoder = codecs.getincrementaldecoder("utf-8")()
        is_valid = True
        while True:
            chunk = await file.read(READ_CHUNK_SIZE_BYTES)

            # Finish after all uploaded bytes have been decoded.
            if not chunk:
                break

            # Reject NUL bytes and invalid UTF-8 commonly found in binary files.
            if b"\x00" in chunk:
                is_valid = False
                break

            try:
                decoder.decode(chunk)
            except UnicodeDecodeError:
                is_valid = False
                break

        if is_valid:
            try:
                decoder.decode(b"", final=True)
            except UnicodeDecodeError:
                is_valid = False
        await file.seek(0)

    # Reject content that does not match the selected parser format.
    if not is_valid:
        await file.seek(0)
        raise HTTPException(
            status_code=415,
            detail={
                "code": "invalid_file_content",
                "message": "The file contents do not match the declared format.",
                "filename": filename,
                "parser": parser.parser_name,
            },
        )


async def _upload_files(
    files: list[UploadFile],
    registry: ParserRegistry,
    corpus_name: str,
) -> dict[str, Any]:
    """Validate, store, and persist every uploaded file in one corpus.

    Args:
        files: Multipart files submitted under the ``files`` field.
        registry: Parser registry used to select each file's parser.
        corpus_name: Required user-provided name for the newly created corpus.

    Returns:
        A success message, persisted corpus details, and original filenames.

    Raises:
        HTTPException: If validation or persistence fails.
    """
    # Normalize the required name before validating files or creating persisted state.
    resolved_corpus_name = corpus_name.strip()

    # Reject whitespace-only names because they cannot identify a corpus meaningfully.
    if not resolved_corpus_name:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_corpus_name",
                "message": "Corpus name must not be blank.",
            },
        )

    validated_files: list[tuple[UploadFile, str, DocumentParser, str]] = []

    # Validate every file before creating or modifying a stored upload.
    for file in files:
        filename = _validate_filename(file.filename)

        # Resolve the parser before reading or saving the body.
        parser = _get_parser_for_filename(registry, filename)
        file_size = await _get_file_size(file)

        # Reject oversized files before any upload is written to disk.
        if file_size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "file_too_large",
                    "message": "Each file must be 30 MB or smaller.",
                    "filename": filename,
                    "max_size_bytes": MAX_FILE_SIZE_BYTES,
                },
            )

        # Verify the actual bytes after size validation and before any file is saved.
        await _validate_file_content(file, filename, parser)
        stored_name = f"{uuid.uuid4()}{Path(filename).suffix.lower()}"
        validated_files.append((file, filename, parser, stored_name))

    # Create the requested storage directory only after the complete request is valid.
    UPLOADS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    stored_paths: list[Path] = []
    document_metadata: list[dict[str, Any]] = []

    try:
        # Persist files with UUID names and collect metadata for database insertion.
        for file, filename, parser, stored_name in validated_files:
            stored_path = UPLOADS_DIRECTORY / stored_name
            size_bytes, content_sha256 = await _save_file(file, stored_path)
            stored_paths.append(stored_path)
            document_metadata.append(
                {
                    "original_filename": filename,
                    "storage_path": str(
                        stored_path.relative_to(UPLOADS_DIRECTORY.parent)
                    ),
                    "mime_type": None,
                    "size_bytes": size_bytes,
                    "content_sha256": content_sha256,
                }
            )
            # Print the original filename and selected parser as requested by ingestion.
            print(f"{filename} -> {parser.parser_name}", flush=True)

        persisted_corpus = create_upload_batch(resolved_corpus_name, document_metadata)
    except sqlite3.Error as error:
        # Remove files when database persistence fails so storage cannot drift from metadata.
        for stored_path in stored_paths:
            stored_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "persistence_error",
                "message": "The uploaded files could not be saved.",
            },
        ) from error

    # Return machine-readable success data while preserving original filenames.
    return {
        "message": "Files uploaded successfully.",
        "filenames": [filename for _, filename, _, _ in validated_files],
        "corpus": persisted_corpus,
    }


@router.post("/uploads")
async def upload_files(
    files: Annotated[list[UploadFile], File(...)],
    corpus_name: Annotated[str, Form(alias="corpusName")],
) -> dict[str, Any]:
    """Handle one or more validated document uploads.

    Args:
        files: Multipart file fields submitted under the name ``files``.
        corpus_name: Required multipart field naming the new corpus.

    Returns:
        A success message and every successfully stored filename.
    """
    registry = build_default_parser_registry()

    try:
        return await _upload_files(files, registry, corpus_name)
    finally:
        # Close all temporary multipart file handles regardless of validation outcome.
        for file in files:
            await file.close()
