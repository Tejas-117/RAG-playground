import unittest
from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import UUID
from zipfile import ZipFile

from httpx import ASGITransport, AsyncClient

from backend.app import app


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

    async def test_upload_stores_files_and_prints_selected_parsers(self) -> None:
        """Verify valid files are stored and their selected parsers are printed.

        Returns:
            None. Assertions fail the test when endpoint behavior changes.
        """
        output = StringIO()
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
                    with redirect_stdout(output):
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
                                        BytesIO(b"%PDF-1.7\n"),
                                        "application/pdf",
                                    ),
                                ),
                            ],
                        )

            self.assertEqual(response.status_code, 200)
            response_data = response.json()
            self.assertEqual(response_data["message"], "Files uploaded successfully.")
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
            self.assertEqual(
                output.getvalue().splitlines(),
                ["notes.txt -> plain_text", "report.pdf -> pymupdf"],
            )
            stored_files = sorted(uploads_directory.iterdir())
            self.assertEqual(len(stored_files), 2)
            self.assertEqual({path.suffix for path in stored_files}, {".txt", ".pdf"})
            self.assertEqual(
                {path.read_bytes() for path in stored_files},
                {b"notes", b"%PDF-1.7\n"},
            )

            # Verify persisted records are available through the read endpoint.
            async with AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                listing_response = await client.get("/corpora/")
            self.assertEqual(listing_response.status_code, 200)
            self.assertEqual(listing_response.json()["corpora"][0]["id"], corpus["id"])

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
