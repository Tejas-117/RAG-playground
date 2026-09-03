"""API and persistence tests for imported evaluation datasets."""

import json
import sqlite3
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient

from backend.app import app
from backend.db.connection import connect
from backend.db.repositories.evaluation_datasets import (
    EvaluationDatasetInUseError,
    create_evaluation_dataset,
)


class EvaluationDatasetTestCase(unittest.IsolatedAsyncioTestCase):
    """Exercise dataset imports against an isolated SQLite database."""

    def setUp(self) -> None:
        """Create one isolated corpus with unique and ambiguous document names.

        Returns:
            None. The database path is patched until each test completes.
        """
        self.database_directory = TemporaryDirectory()
        self.database_path = Path(self.database_directory.name) / "test.sqlite3"
        self.database_patch = patch(
            "backend.db.connection.DATABASE_PATH",
            self.database_path,
        )
        self.database_patch.start()
        self.addCleanup(self.database_patch.stop)
        self.addCleanup(self.database_directory.cleanup)

        # Seed immutable corpus documents directly; parsing is irrelevant to labels.
        with connect() as connection:
            connection.execute(
                "INSERT INTO corpus VALUES (?, ?, ?, ?, ?)",
                (
                    "corpus-1",
                    "Product docs",
                    None,
                    "2026-08-31T00:00:00Z",
                    "2026-08-31T00:00:00Z",
                ),
            )
            connection.executemany(
                "INSERT INTO document VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        "document-guide",
                        "corpus-1",
                        "guide.pdf",
                        "uploads/guide.pdf",
                        "application/pdf",
                        10,
                        "a" * 64,
                        "2026-08-31T00:00:01Z",
                    ),
                    (
                        "document-copy-1",
                        "corpus-1",
                        "copy.pdf",
                        "uploads/copy-1.pdf",
                        "application/pdf",
                        10,
                        "b" * 64,
                        "2026-08-31T00:00:02Z",
                    ),
                    (
                        "document-copy-2",
                        "corpus-1",
                        "copy.pdf",
                        "uploads/copy-2.pdf",
                        "application/pdf",
                        10,
                        "c" * 64,
                        "2026-08-31T00:00:03Z",
                    ),
                ],
            )

    async def _post_dataset(
        self,
        body: object,
        name: str = "Support questions",
        corpus_id: str = "corpus-1",
        filename: str = "support.json",
    ) -> object:
        """Submit one JSON-compatible body to the multipart import endpoint.

        Args:
            body: Value serialized as the uploaded JSON document.
            name: Multipart dataset display name.
            corpus_id: Multipart selected corpus identifier.
            filename: Client-provided upload filename.

        Returns:
            HTTPX response returned by the ASGI application.
        """
        transport = ASGITransport(app=app)
        source_body = json.dumps(body).encode("utf-8")

        # Use the real API stack while keeping all storage local to this test.
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.post(
                "/datasets",
                data={"name": name, "corpus_id": corpus_id},
                files={"file": (filename, BytesIO(source_body), "application/json")},
            )

    async def test_import_preserves_duplicates_and_reports_skipped_documents(
        self,
    ) -> None:
        """Verify duplicate examples remain and bad labels become warnings.

        Returns:
            None. Assertions verify the complete successful import response.
        """
        example = {
            "question": "Where is setup documented?",
            "reference_answer": "In the guide.",
            "relevant_documents": ["guide.pdf", "missing.pdf", "copy.pdf"],
        }
        response = await self._post_dataset({"examples": [example, example]})

        assert response.status_code == 201
        payload = response.json()
        assert payload["name"] == "Support questions"
        assert payload["example_count"] == 2
        assert payload["resolved_document_count"] == 2
        assert payload["warning_count"] == 4
        assert [item["ordinal"] for item in payload["examples"]] == [0, 1]
        assert payload["examples"][0]["relevant_documents"] == [
            {"id": "document-guide", "filename": "guide.pdf"}
        ]
        assert {warning["code"] for warning in payload["warnings"]} == {
            "document_not_found",
            "ambiguous_document_name",
        }
        assert all(warning["example_id"] for warning in payload["warnings"])

    async def test_import_keeps_question_when_all_document_names_are_wrong(
        self,
    ) -> None:
        """Verify unresolved labels do not cause their evaluation example to be lost.

        Returns:
            None. Assertions verify the question remains with an empty relevance set.
        """
        response = await self._post_dataset(
            {
                "examples": [
                    {
                        "question": "Still a useful question?",
                        "relevant_documents": ["unknown.pdf"],
                    }
                ]
            }
        )

        assert response.status_code == 201
        payload = response.json()
        assert payload["examples"][0]["question"] == "Still a useful question?"
        assert payload["examples"][0]["relevant_documents"] == []
        assert payload["warning_count"] == 1

    async def test_import_rejects_invalid_schema_and_unknown_corpus(self) -> None:
        """Verify malformed examples and missing corpora return structured errors.

        Returns:
            None. Assertions verify both failures occur before dataset persistence.
        """
        invalid_response = await self._post_dataset({"examples": []})
        missing_corpus_response = await self._post_dataset(
            {"examples": [{"question": "Valid question"}]},
            corpus_id="missing-corpus",
        )

        assert invalid_response.status_code == 422
        assert invalid_response.json()["detail"]["code"] == "invalid_dataset_schema"
        assert missing_corpus_response.status_code == 404
        assert missing_corpus_response.json()["detail"]["code"] == "corpus_not_found"

    async def test_import_rejects_non_json_filename(self) -> None:
        """Verify the importer enforces the initial JSON-only contract.

        Returns:
            None. Assertions verify a structured unsupported-media response.
        """
        response = await self._post_dataset(
            {"examples": [{"question": "Question"}]},
            filename="dataset.csv",
        )

        assert response.status_code == 415
        assert response.json()["detail"]["code"] == "unsupported_dataset_file_type"

    async def test_import_rejects_invalid_json_and_oversized_file(self) -> None:
        """Verify malformed and oversized uploads return distinct structured errors.

        Returns:
            None. Assertions verify invalid files never reach persistence.
        """
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            invalid_json_response = await client.post(
                "/datasets",
                data={"name": "Broken", "corpus_id": "corpus-1"},
                files={"file": ("broken.json", BytesIO(b"{"), "application/json")},
            )

            # Patch the limit rather than allocating the production 30 MB boundary.
            with patch("backend.api.routers.datasets.MAX_DATASET_SIZE_BYTES", 4):
                oversized_response = await client.post(
                    "/datasets",
                    data={"name": "Large", "corpus_id": "corpus-1"},
                    files={
                        "file": ("large.json", BytesIO(b"12345"), "application/json")
                    },
                )

        assert invalid_json_response.status_code == 422
        assert invalid_json_response.json()["detail"]["code"] == (
            "invalid_dataset_json"
        )
        assert oversized_response.status_code == 413
        assert oversized_response.json()["detail"]["code"] == ("dataset_file_too_large")

    async def test_list_filter_detail_and_delete_dataset(self) -> None:
        """Verify the dataset inventory supports filtering, inspection, and deletion.

        Returns:
            None. Assertions verify all management endpoints share stable IDs.
        """
        created_response = await self._post_dataset(
            {"examples": [{"question": "Question"}]},
            name="Alpha Support",
        )
        dataset_id = created_response.json()["id"]
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            list_response = await client.get(
                "/datasets",
                params={"corpus_id": "corpus-1", "search": "SUPPORT"},
            )
            detail_response = await client.get(f"/datasets/{dataset_id}")
            delete_response = await client.delete(f"/datasets/{dataset_id}")
            missing_response = await client.get(f"/datasets/{dataset_id}")

        assert list_response.status_code == 200
        assert [dataset["id"] for dataset in list_response.json()] == [dataset_id]
        assert detail_response.status_code == 200
        assert detail_response.json()["examples"][0]["question"] == "Question"
        assert delete_response.status_code == 204
        assert missing_response.status_code == 404

    async def test_delete_reports_missing_and_protected_datasets(self) -> None:
        """Verify deletion distinguishes unknown IDs from protected history.

        Returns:
            None. Assertions verify stable 404 and 409 machine-readable errors.
        """
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            missing_response = await client.delete("/datasets/missing-dataset")

            # Simulate the foreign-key conflict future benchmark tables will enforce.
            with patch(
                "backend.api.routers.datasets.delete_evaluation_dataset",
                side_effect=EvaluationDatasetInUseError("protected-dataset"),
            ):
                protected_response = await client.delete("/datasets/protected-dataset")

        assert missing_response.status_code == 404
        assert missing_response.json()["detail"]["code"] == "dataset_not_found"
        assert protected_response.status_code == 409
        assert protected_response.json()["detail"]["code"] == "dataset_in_use"

    def test_persistence_failure_rolls_back_parent_and_examples(self) -> None:
        """Verify an invalid resolved document cannot leave a partial dataset.

        Returns:
            None. Assertions verify SQLite rolls back the complete write transaction.
        """
        with self.assertRaises(sqlite3.IntegrityError):
            create_evaluation_dataset(
                "Broken dataset",
                "corpus-1",
                "broken.json",
                "d" * 64,
                [
                    {
                        "ordinal": 0,
                        "question": "Question",
                        "reference_answer": None,
                        "relevant_document_ids": ["missing-document"],
                    }
                ],
                [],
            )

        # A transactional rollback removes the parent inserted before the bad label.
        with connect() as connection:
            dataset_count = connection.execute(
                "SELECT COUNT(*) FROM evaluation_dataset"
            ).fetchone()[0]
        assert dataset_count == 0
