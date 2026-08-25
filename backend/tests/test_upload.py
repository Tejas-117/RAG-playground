import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from uuid import UUID
from zipfile import ZipFile

from httpx import ASGITransport, AsyncClient

from backend.app import app
from backend.db.repositories.parsed_documents import get_parsed_document
from backend.ingestion.parsers.errors import ParserDependencyError
from backend.ingestion.parsers.registry import ParserRegistry


async def _run_parser_inline(function: object, *args: object) -> object:
    """Execute a parser inline when worker threads are unavailable in tests.

    Args:
        function: Parser callable normally delegated to the worker pool.
        *args: Positional arguments supplied to the parser callable.

    Returns:
        The parser callable's result.
    """
    # The executor boundary is mocked; real parser and persistence code still run.
    return function(*args)  # type: ignore[operator]


def _create_pdf_bytes(page_texts: list[str]) -> bytes:
    """Create a small deterministic PDF fixture entirely in memory.

    Args:
        page_texts: Text inserted into each page; blank values create empty pages.

    Returns:
        Serialized PDF bytes accepted by PyMuPDF.
    """
    import fitz

    document = fitz.open()

    # Add each requested page and insert selectable text when supplied.
    for text in page_texts:
        page = document.new_page()

        # Leave blank fixture pages physically present for warning tests.
        if text:
            page.insert_text((72, 72), text)

    body = document.tobytes()
    document.close()
    return body


class UploadFilesTestCase(unittest.IsolatedAsyncioTestCase):
    """Regression tests for the multipart upload endpoint."""

    def setUp(self) -> None:
        """Create an isolated database for each upload test.

        Returns:
            None. The repository database path is patched for this test case.
        """
        # Keep API tests deterministic and prevent test data from reaching the local database.
        self.database_directory = TemporaryDirectory()
        self.database_path = Path(self.database_directory.name) / "test.sqlite3"
        self.database_patch = patch(
            "backend.db.connection.DATABASE_PATH", self.database_path
        )
        self.database_patch.start()
        self.addCleanup(self.database_patch.stop)
        self.addCleanup(self.database_directory.cleanup)
        self.threadpool_patch = patch(
            "backend.api.routers.uploads.run_in_threadpool",
            side_effect=_run_parser_inline,
        )
        self.threadpool_patch.start()
        self.addCleanup(self.threadpool_patch.stop)

    async def test_upload_stores_files_and_logs_stage_lifecycle(self) -> None:
        """Verify valid files are stored and upload/parsing stages are logged.

        Returns:
            None. Assertions fail the test when endpoint behavior changes.
        """
        transport = ASGITransport(app=app)
        pdf_body = _create_pdf_bytes(["PDF notes"])

        with TemporaryDirectory() as directory:
            uploads_directory = Path(directory)
            with (
                patch(
                    "backend.api.routers.uploads.UPLOADS_DIRECTORY",
                    uploads_directory,
                ),
                self.assertLogs(
                    "backend.api.routers.uploads",
                    level="INFO",
                ) as captured_logs,
            ):
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    response = await client.post(
                        "/uploads",
                        data={"corpusName": "Project notes"},
                        files=[
                            (
                                "files",
                                ("notes.txt", BytesIO(b"notes"), "text/plain"),
                            ),
                            (
                                "files",
                                (
                                    "report.pdf",
                                    BytesIO(pdf_body),
                                    "application/pdf",
                                ),
                            ),
                        ],
                    )

            self.assertEqual(response.status_code, 200)
            response_data = response.json()
            self.assertEqual(
                response_data["message"],
                "Files uploaded and parsed successfully.",
            )
            self.assertEqual(response_data["filenames"], ["notes.txt", "report.pdf"])
            corpus = response_data["corpus"]
            UUID(corpus["id"])
            self.assertEqual(corpus["name"], "Project notes")
            self.assertEqual(
                [document["original_filename"] for document in corpus["documents"]],
                ["notes.txt", "report.pdf"],
            )
            self.assertEqual(
                [document["mime_type"] for document in corpus["documents"]],
                [None, None],
            )
            combined_logs = "\n".join(captured_logs.output)
            self.assertIn("upload_started file_count=2", combined_logs)
            self.assertIn(
                "parsing_completed filename='notes.txt' parser=plain_text",
                combined_logs,
            )
            self.assertIn(
                "parsing_completed filename='report.pdf' parser=pymupdf",
                combined_logs,
            )
            self.assertIn("upload_completed corpus_id=", combined_logs)
            stored_files = sorted(uploads_directory.iterdir())
            self.assertEqual(len(stored_files), 2)
            self.assertEqual({path.suffix for path in stored_files}, {".txt", ".pdf"})
            self.assertEqual(
                {path.read_bytes() for path in stored_files},
                {b"notes", pdf_body},
            )

            # Verify each source has one persisted parse summary with provenance.
            parse_summaries = [document["parse"] for document in corpus["documents"]]
            self.assertEqual(
                [summary["parser_name"] for summary in parse_summaries],
                ["plain_text", "pymupdf"],
            )
            self.assertEqual(parse_summaries[1]["page_count"], 1)
            self.assertGreaterEqual(parse_summaries[1]["block_count"], 1)

            # Verify persisted records are available through the read endpoint.
            async with AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                listing_response = await client.get("/corpora/")
            self.assertEqual(listing_response.status_code, 200)
            self.assertEqual(listing_response.json()["corpora"][0]["id"], corpus["id"])
            listed_documents = listing_response.json()["corpora"][0]["documents"]
            self.assertEqual(
                listed_documents[1]["parse"]["parser_name"],
                "pymupdf",
            )

            # Read through the chunking-facing boundary and resolve stored offsets.
            parsed_pdf = get_parsed_document(corpus["documents"][1]["id"])
            self.assertIsNotNone(parsed_pdf)
            assert parsed_pdf is not None
            first_block = parsed_pdf["pages"][0]["blocks"][0]
            self.assertEqual(
                parsed_pdf["normalized_text"][
                    first_block["character_start_offset"] : first_block[
                        "character_end_offset"
                    ]
                ],
                "PDF notes",
            )

    async def test_upload_accepts_partial_empty_pdf_with_persisted_warning(
        self,
    ) -> None:
        """Verify an empty page warns without rejecting other usable PDF text."""
        transport = ASGITransport(app=app)
        pdf_body = _create_pdf_bytes(["Readable", ""])

        with TemporaryDirectory() as directory:
            uploads_directory = Path(directory)
            with patch(
                "backend.api.routers.uploads.UPLOADS_DIRECTORY",
                uploads_directory,
            ):
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    response = await client.post(
                        "/uploads",
                        data={"corpusName": "Warnings"},
                        files=[
                            (
                                "files",
                                ("mixed.pdf", BytesIO(pdf_body), "application/pdf"),
                            )
                        ],
                    )

        self.assertEqual(response.status_code, 200)
        summary = response.json()["corpus"]["documents"][0]["parse"]
        self.assertEqual(summary["page_count"], 2)
        self.assertEqual(
            summary["warnings"],
            ["Page 2 did not return selectable text"],
        )

    async def test_parse_failure_rolls_back_entire_upload_batch(self) -> None:
        """Verify one malformed document removes all files and database rows."""
        transport = ASGITransport(app=app)

        with TemporaryDirectory() as directory:
            uploads_directory = Path(directory)
            with patch(
                "backend.api.routers.uploads.UPLOADS_DIRECTORY",
                uploads_directory,
            ):
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    response = await client.post(
                        "/uploads",
                        data={"corpusName": "Atomic parsing"},
                        files=[
                            ("files", ("good.txt", BytesIO(b"good"), "text/plain")),
                            (
                                "files",
                                (
                                    "broken.pdf",
                                    BytesIO(b"%PDF-1.7\nnot-a-real-pdf"),
                                    "application/pdf",
                                ),
                            ),
                        ],
                    )
                    listing_response = await client.get("/corpora/")

            self.assertEqual(response.status_code, 422)
            self.assertEqual(
                response.json()["detail"]["code"],
                "document_parse_failed",
            )
            self.assertEqual(list(uploads_directory.iterdir()), [])
            self.assertEqual(listing_response.json()["corpora"], [])

    async def test_upload_rejects_document_without_extractable_text(self) -> None:
        """Verify whitespace-only text does not create files or corpus records."""
        transport = ASGITransport(app=app)

        with TemporaryDirectory() as directory:
            uploads_directory = Path(directory)
            with patch(
                "backend.api.routers.uploads.UPLOADS_DIRECTORY",
                uploads_directory,
            ):
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    response = await client.post(
                        "/uploads",
                        data={"corpusName": "Empty"},
                        files=[
                            ("files", ("empty.txt", BytesIO(b" \n\t"), "text/plain"))
                        ],
                    )

            self.assertEqual(response.status_code, 422)
            self.assertEqual(
                response.json()["detail"]["code"],
                "empty_parsed_document",
            )
            self.assertEqual(list(uploads_directory.iterdir()), [])

    async def test_upload_rejects_oversized_extracted_text(self) -> None:
        """Verify expanded parser text uses a separate structured size limit."""
        transport = ASGITransport(app=app)

        with TemporaryDirectory() as directory:
            uploads_directory = Path(directory)
            with (
                patch(
                    "backend.api.routers.uploads.UPLOADS_DIRECTORY",
                    uploads_directory,
                ),
                patch(
                    "backend.ingestion.canonicalization.MAX_NORMALIZED_TEXT_SIZE_BYTES",
                    3,
                ),
            ):
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    response = await client.post(
                        "/uploads",
                        data={"corpusName": "Expanded text"},
                        files=[
                            ("files", ("large.txt", BytesIO(b"four"), "text/plain"))
                        ],
                    )

            self.assertEqual(response.status_code, 413)
            self.assertEqual(
                response.json()["detail"]["code"],
                "parsed_document_too_large",
            )
            self.assertEqual(list(uploads_directory.iterdir()), [])

    async def test_upload_returns_structured_parser_dependency_error(self) -> None:
        """Verify an unavailable parser rolls back storage with a safe error."""
        transport = ASGITransport(app=app)
        parser = Mock()
        parser.parser_name = "missing_parser"
        parser.supported_extensions = frozenset({".txt"})
        parser.parse.side_effect = ParserDependencyError(
            "missing_parser",
            "missing-package",
        )
        registry = ParserRegistry([parser])

        with TemporaryDirectory() as directory:
            uploads_directory = Path(directory)
            with (
                patch(
                    "backend.api.routers.uploads.UPLOADS_DIRECTORY",
                    uploads_directory,
                ),
                patch(
                    "backend.api.routers.uploads.build_default_parser_registry",
                    return_value=registry,
                ),
            ):
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    response = await client.post(
                        "/uploads",
                        data={"corpusName": "Unavailable parser"},
                        files=[
                            ("files", ("notes.txt", BytesIO(b"notes"), "text/plain"))
                        ],
                    )

            self.assertEqual(response.status_code, 500)
            self.assertEqual(
                response.json()["detail"]["code"],
                "parser_unavailable",
            )
            self.assertEqual(list(uploads_directory.iterdir()), [])

    async def test_upload_allows_duplicate_original_names_with_uuid_storage(
        self,
    ) -> None:
        """Verify repeated original names are persisted as separate UUID files."""
        transport = ASGITransport(app=app)

        with TemporaryDirectory() as directory:
            uploads_directory = Path(directory)
            with patch(
                "backend.api.routers.uploads.UPLOADS_DIRECTORY",
                uploads_directory,
            ):
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    response = await client.post(
                        "/uploads",
                        data={"corpusName": "Duplicate files"},
                        files=[
                            ("files", ("same.txt", BytesIO(b"one"), "text/plain")),
                            ("files", ("same.txt", BytesIO(b"two"), "text/plain")),
                        ],
                    )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["filenames"], ["same.txt", "same.txt"])
            stored_files = list(uploads_directory.iterdir())
            self.assertEqual(len(stored_files), 2)
            self.assertEqual(
                {path.read_bytes() for path in stored_files}, {b"one", b"two"}
            )

    async def test_upload_requires_a_nonblank_corpus_name(self) -> None:
        """Verify absent and whitespace-only corpus names are rejected before storage.

        Returns:
            None. Assertions fail when corpus-name validation changes.
        """
        transport = ASGITransport(app=app)

        with TemporaryDirectory() as directory:
            uploads_directory = Path(directory)
            with patch(
                "backend.api.routers.uploads.UPLOADS_DIRECTORY",
                uploads_directory,
            ):
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    # Omit the multipart field entirely to verify FastAPI treats it as required.
                    missing_response = await client.post(
                        "/uploads",
                        files=[
                            (
                                "files",
                                ("notes.txt", BytesIO(b"notes"), "text/plain"),
                            )
                        ],
                    )

                    # Submit only whitespace to verify application-level name validation.
                    blank_response = await client.post(
                        "/uploads",
                        data={"corpusName": "  \t  "},
                        files=[
                            (
                                "files",
                                ("notes.txt", BytesIO(b"notes"), "text/plain"),
                            )
                        ],
                    )

            self.assertEqual(missing_response.status_code, 422)
            self.assertEqual(blank_response.status_code, 422)
            self.assertEqual(
                blank_response.json()["detail"]["code"], "invalid_corpus_name"
            )
            self.assertEqual(list(uploads_directory.iterdir()), [])

    async def test_upload_rejects_unsupported_file_without_writing_any_file(
        self,
    ) -> None:
        """Verify unsupported formats fail before any request file is persisted."""
        transport = ASGITransport(app=app)

        with TemporaryDirectory() as directory:
            uploads_directory = Path(directory)
            with patch(
                "backend.api.routers.uploads.UPLOADS_DIRECTORY",
                uploads_directory,
            ):
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    response = await client.post(
                        "/uploads",
                        data={"corpusName": "Validation corpus"},
                        files=[
                            ("files", ("valid.txt", BytesIO(b"valid"), "text/plain")),
                            (
                                "files",
                                ("archive.zip", BytesIO(b"zip"), "application/zip"),
                            ),
                        ],
                    )

            self.assertEqual(response.status_code, 415)
            self.assertEqual(response.json()["detail"]["code"], "unsupported_file_type")
            self.assertEqual(list(uploads_directory.iterdir()), [])

    async def test_upload_ignores_client_mime_type_for_parser_selection(self) -> None:
        """Verify client MIME metadata cannot override extension parser selection."""
        transport = ASGITransport(app=app)

        with TemporaryDirectory() as directory:
            uploads_directory = Path(directory)
            with patch(
                "backend.api.routers.uploads.UPLOADS_DIRECTORY",
                uploads_directory,
            ):
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    response = await client.post(
                        "/uploads",
                        data={"corpusName": "MIME test"},
                        files=[
                            (
                                "files",
                                ("notes.txt", BytesIO(b"notes"), "application/pdf"),
                            )
                        ],
                    )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["filenames"], ["notes.txt"])
            self.assertEqual(
                response.json()["corpus"]["documents"][0]["mime_type"],
                None,
            )
            self.assertEqual(
                [path.read_bytes() for path in uploads_directory.iterdir()],
                [b"notes"],
            )

    async def test_upload_rejects_zip_renamed_as_text_without_writing(self) -> None:
        """Verify binary ZIP content cannot pass as a text upload."""
        archive_body = BytesIO()
        with ZipFile(archive_body, "w") as archive:
            archive.writestr("payload.txt", "not a plain text upload")
        archive_body.seek(0)
        transport = ASGITransport(app=app)

        with TemporaryDirectory() as directory:
            uploads_directory = Path(directory)
            with patch(
                "backend.api.routers.uploads.UPLOADS_DIRECTORY",
                uploads_directory,
            ):
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    response = await client.post(
                        "/uploads",
                        data={"corpusName": "Archive validation"},
                        files=[
                            (
                                "files",
                                ("test_zip_to_pdf.txt", archive_body, "text/plain"),
                            )
                        ],
                    )

            self.assertEqual(response.status_code, 415)
            self.assertEqual(response.json()["detail"]["code"], "invalid_file_content")
            self.assertEqual(list(uploads_directory.iterdir()), [])

    async def test_upload_rejects_file_over_30_mb_without_writing(self) -> None:
        """Verify a file larger than 30 MiB receives a structured size error."""
        transport = ASGITransport(app=app)
        oversized_content = b"x" * (30 * 1024 * 1024 + 1)

        with TemporaryDirectory() as directory:
            uploads_directory = Path(directory)
            with patch(
                "backend.api.routers.uploads.UPLOADS_DIRECTORY",
                uploads_directory,
            ):
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    response = await client.post(
                        "/uploads",
                        data={"corpusName": "Size validation"},
                        files=[
                            (
                                "files",
                                ("large.txt", BytesIO(oversized_content), "text/plain"),
                            )
                        ],
                    )

            self.assertEqual(response.status_code, 413)
            self.assertEqual(response.json()["detail"]["code"], "file_too_large")
            self.assertEqual(list(uploads_directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
