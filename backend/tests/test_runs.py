"""API tests for immutable single-question pipeline run persistence."""

import json
import re
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient, Response

from backend.app import app
from backend.db.connection import connect
from backend.db.repositories.runs import InvalidRunStateError, complete_run
from backend.ingestion.chunkers.models import TokenizedText, TokenOffset
from backend.ingestion.chunkers.tokenizer import TokenizerAssetError
from backend.pipeline.execution import PipelineExecutor, get_pipeline_executor


class RunTestTokenizer:
    """Provide deterministic offline token offsets to API execution tests."""

    identifier = "run-test-tokenizer"
    revision = "1"
    asset_sha256 = "run-test-digest"
    special_tokens_policy = "none"

    def encode(self, text: str) -> TokenizedText:
        """Map every non-whitespace test word to its source character range.

        Args:
            text: Canonical document text submitted by the chunking service.

        Returns:
            Ordered offsets for deterministic word-like test tokens.
        """
        # Regex offsets exercise production provenance without loading a model asset.
        return TokenizedText(
            offsets=tuple(
                TokenOffset(match.start(), match.end())
                for match in re.finditer(r"\S+", text)
            )
        )


class UnavailableRunTestTokenizer(RunTestTokenizer):
    """Simulate a production tokenizer asset that cannot be loaded."""

    def encode(self, text: str) -> TokenizedText:
        """Raise the operational tokenizer failure expected by the executor.

        Args:
            text: Canonical text that cannot be encoded by the missing asset.

        Returns:
            Never returns because the configured asset is unavailable.

        Raises:
            TokenizerAssetError: Always, to exercise structured failure handling.
        """
        # The error text is intentionally hidden by the public API mapping.
        raise TokenizerAssetError("simulated missing tokenizer")


def _run_payload() -> dict[str, object]:
    """Create a valid request using defaults for optional pipeline settings.

    Args:
        None.

    Returns:
        A fresh dictionary suitable for ``POST /runs`` tests.
    """
    # Keep only required selections in the request so tests verify resolved defaults.
    return {
        "corpus_id": "corpus-1",
        "question": "  What is the refund policy?  ",
        "configuration": {
            "chunking": {},
            "embedding": {
                "provider": "ollama",
                "model": "nomic-embed-text",
            },
            "retrieval": {},
            "generation": {
                "provider": "ollama",
                "model": "llama3.2:3b",
            },
        },
    }


class PipelineRunRouteTestCase(unittest.IsolatedAsyncioTestCase):
    """Exercise the public single-question run endpoint with isolated SQLite data."""

    def setUp(self) -> None:
        """Create and patch an isolated SQLite database for one test.

        Args:
            None.

        Returns:
            None. The test instance owns the temporary database until teardown.
        """
        # Redirect repository connections away from the developer's local database.
        self.database_directory = TemporaryDirectory()
        self.database_path = Path(self.database_directory.name) / "runs.sqlite3"
        self.database_patch = patch(
            "backend.db.connection.DATABASE_PATH",
            self.database_path,
        )
        self.database_patch.start()
        self.executor = PipelineExecutor(tokenizer=RunTestTokenizer())
        app.dependency_overrides[get_pipeline_executor] = self._get_executor

        # Seed one immutable corpus and completed canonical parse for valid runs.
        with connect() as connection:
            connection.execute(
                "INSERT INTO corpus VALUES (?, ?, ?, ?, ?)",
                (
                    "corpus-1",
                    "Product docs",
                    None,
                    "2026-08-02T00:00:00Z",
                    "2026-08-02T00:00:00Z",
                ),
            )
            connection.execute(
                "INSERT INTO document VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "document-1",
                    "corpus-1",
                    "refunds.txt",
                    "uploads/refunds.txt",
                    "text/plain",
                    42,
                    "a" * 64,
                    "2026-08-02T00:00:01Z",
                ),
            )
            connection.execute(
                """
                INSERT INTO document_parse VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    "parse-1",
                    "document-1",
                    "Refunds are processed within five business days.",
                    48,
                    48,
                    "text-parser",
                    "1.0.0",
                    "{}",
                    "[]",
                    0,
                    0,
                    1,
                    "2026-08-02T00:00:02Z",
                ),
            )

    def _get_executor(self) -> PipelineExecutor:
        """Return the deterministic executor used by this isolated API test.

        Args:
            None.

        Returns:
            Pipeline executor configured with the offline test tokenizer.
        """
        # FastAPI dependency overrides call this method for every test request.
        return self.executor

    def tearDown(self) -> None:
        """Restore the database path and remove temporary test data.

        Args:
            None.

        Returns:
            None. Resources created by setup are released.
        """
        # Restore the production dependency before releasing the isolated database.
        app.dependency_overrides.pop(get_pipeline_executor, None)

        # Stop patching before deleting the temporary directory it references.
        self.database_patch.stop()
        self.database_directory.cleanup()

    async def _post_run(self, payload: dict[str, object]) -> Response:
        """Submit a run request through the complete FastAPI application.

        Args:
            payload: JSON-compatible request body sent to the runs endpoint.

        Returns:
            The HTTPX response returned by the application.
        """
        transport = ASGITransport(app=app)

        # Use an in-process ASGI client so tests remain fast and offline.
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post("/runs", json=payload)

    async def test_create_run_saves_trimmed_question_and_resolved_defaults(
        self,
    ) -> None:
        """Verify a valid request stores its normalized immutable snapshot.

        Args:
            None.

        Returns:
            None. Assertions verify the API response and SQLite row.
        """
        response = await self._post_run(_run_payload())

        # Confirm creation returns the normalized question and backend defaults.
        self.assertEqual(response.status_code, 201)
        response_data = response.json()
        self.assertEqual(response_data["question"], "What is the refund policy?")
        self.assertEqual(response_data["status"], "completed")
        self.assertEqual(response_data["chunking"]["status"], "ready")
        self.assertGreater(response_data["chunking"]["chunk_count"], 0)
        self.assertFalse(response_data["chunking"]["reused"])
        self.assertEqual(response_data["configuration"]["retrieval"]["top_k"], 10)
        self.assertEqual(
            response_data["configuration"]["generation"]["max_output_tokens"],
            1000,
        )
        self.assertEqual(
            response_data["configuration"]["evaluation"],
            {"retrieval_metrics": [], "answer_metrics": []},
        )

        # Read the stored JSON to confirm persistence matches the response snapshot.
        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT question, effective_config_json FROM pipeline_run"
            ).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], "What is the refund policy?")
        self.assertEqual(json.loads(row[1]), response_data["configuration"])

    async def test_create_run_saves_complete_frontend_configuration(self) -> None:
        """Verify the complete frontend payload is persisted as the run snapshot.

        Args:
            None.

        Returns:
            None. Assertions verify the response and database contain the UI selections.
        """
        payload = {
            "corpus_id": "corpus-1",
            "question": "How long do refunds take?",
            "configuration": {
                "chunking": {
                    "strategy": "recursive",
                    "chunk_size_tokens": 800,
                    "chunk_overlap_tokens": 100,
                },
                "embedding": {
                    "provider": "ollama",
                    "model": "all-minilm",
                    "distance_metric": "cosine",
                },
                "retrieval": {"top_k": 10},
                "generation": {
                    "provider": "ollama",
                    "model": "llama3.2:3b",
                    "temperature": 0.2,
                    "max_output_tokens": 1000,
                },
                "evaluation": {
                    "retrieval_metrics": [],
                    "answer_metrics": ["groundedness", "answer_relevance"],
                },
            },
        }

        response = await self._post_run(payload)

        # Verify FastAPI accepted and returned every selection submitted by the UI.
        self.assertEqual(response.status_code, 201)
        response_data = response.json()
        self.assertEqual(response_data["configuration"], payload["configuration"])

        with sqlite3.connect(self.database_path) as connection:
            stored_row = connection.execute(
                "SELECT corpus_id, question, effective_config_json FROM pipeline_run"
            ).fetchone()

        # Confirm the immutable SQLite row matches the selected source, question, and config.
        self.assertIsNotNone(stored_row)
        self.assertEqual(stored_row[0], payload["corpus_id"])
        self.assertEqual(stored_row[1], payload["question"])
        self.assertEqual(json.loads(stored_row[2]), payload["configuration"])

    async def test_completed_run_rejects_second_completion(self) -> None:
        """Verify terminal run history cannot be overwritten by a retry.

        Args:
            None.

        Returns:
            None. Assertions verify the repository lifecycle guard.
        """
        response = await self._post_run(_run_payload())
        response_data = response.json()

        # Repeating a terminal transition must fail instead of mutating history.
        with self.assertRaises(InvalidRunStateError):
            complete_run(
                response_data["id"],
                response_data["chunking"]["chunk_set_id"],
                True,
                1,
            )

    async def test_create_run_rejects_blank_question(self) -> None:
        """Verify whitespace-only questions produce a structured validation error.

        Args:
            None.

        Returns:
            None. Assertions verify no unusable run is accepted.
        """
        payload = _run_payload()
        payload["question"] = "   "
        response = await self._post_run(payload)

        # Require the stable error contract used by the frontend.
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["detail"],
            {
                "code": "invalid_question",
                "message": "Question must not be blank.",
            },
        )

    async def test_create_run_rejects_unknown_corpus(self) -> None:
        """Verify a run cannot reference a corpus that does not exist.

        Args:
            None.

        Returns:
            None. Assertions verify the structured not-found response.
        """
        payload = _run_payload()
        payload["corpus_id"] = "missing-corpus"
        response = await self._post_run(payload)

        # Distinguish a missing corpus from general persistence failures.
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"]["code"], "corpus_not_found")

    async def test_create_run_records_missing_parse_failure(self) -> None:
        """Verify incomplete canonical inputs leave one structured failed run.

        Args:
            None.

        Returns:
            None. Assertions verify the API error and persisted lifecycle state.
        """
        # Remove only the parse so the selected corpus remains valid but unchunkable.
        with connect() as connection:
            connection.execute("DELETE FROM document_parse WHERE id = ?", ("parse-1",))

        response = await self._post_run(_run_payload())

        # The failure identifies both the auditable run and repairable document.
        self.assertEqual(response.status_code, 409)
        detail = response.json()["detail"]
        self.assertEqual(detail["code"], "missing_parse_artifact")
        self.assertEqual(detail["document_ids"], ["document-1"])

        with sqlite3.connect(self.database_path) as connection:
            failed_run = connection.execute(
                """
                SELECT id, status, chunk_set_id, error_code
                FROM pipeline_run
                """
            ).fetchone()

        self.assertEqual(failed_run[0], detail["run_id"])
        self.assertEqual(failed_run[1:], ("failed", None, "missing_parse_artifact"))

    async def test_create_run_records_empty_corpus_failure(self) -> None:
        """Verify a known corpus without documents becomes a failed run.

        Args:
            None.

        Returns:
            None. Assertions verify conflict transport and auditable failure state.
        """
        # Insert a valid immutable corpus that intentionally has no document rows.
        with connect() as connection:
            connection.execute(
                "INSERT INTO corpus VALUES (?, ?, ?, ?, ?)",
                (
                    "empty-corpus",
                    "Empty corpus",
                    None,
                    "2026-08-02T00:00:00Z",
                    "2026-08-02T00:00:00Z",
                ),
            )

        payload = _run_payload()
        payload["corpus_id"] = "empty-corpus"
        response = await self._post_run(payload)

        # A known but unusable corpus is a state conflict rather than a not-found error.
        self.assertEqual(response.status_code, 409)
        detail = response.json()["detail"]
        self.assertEqual(detail["code"], "empty_corpus")

        with sqlite3.connect(self.database_path) as connection:
            failed_run = connection.execute(
                """
                SELECT status, error_code
                FROM pipeline_run
                WHERE id = ?
                """,
                (detail["run_id"],),
            ).fetchone()

        self.assertEqual(failed_run, ("failed", "empty_corpus"))

    async def test_create_run_records_tokenizer_failure(self) -> None:
        """Verify tokenizer asset failures are sanitized and persisted.

        Args:
            None.

        Returns:
            None. Assertions verify safe transport and database failure records.
        """
        # Replace the normal offline tokenizer with a deterministic failing adapter.
        self.executor = PipelineExecutor(tokenizer=UnavailableRunTestTokenizer())
        response = await self._post_run(_run_payload())

        # Internal asset details must not leak through the structured API response.
        self.assertEqual(response.status_code, 500)
        detail = response.json()["detail"]
        self.assertEqual(detail["code"], "chunking_tokenizer_unavailable")
        self.assertNotIn("simulated", detail["message"])

        with sqlite3.connect(self.database_path) as connection:
            failed_run = connection.execute(
                "SELECT status, error_code FROM pipeline_run"
            ).fetchone()

        self.assertEqual(
            failed_run,
            ("failed", "chunking_tokenizer_unavailable"),
        )

    async def test_create_run_rejects_unsupported_provider(self) -> None:
        """Verify the backend catalog controls executable provider identifiers.

        Args:
            None.

        Returns:
            None. Assertions verify semantic compatibility validation.
        """
        payload = _run_payload()
        configuration = payload["configuration"]
        configuration["embedding"]["provider"] = "unknown"
        response = await self._post_run(payload)

        # Return the incompatible field through a stable machine-readable error.
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["detail"],
            {
                "code": "invalid_pipeline_configuration",
                "message": "Provider 'unknown' is not supported.",
                "field": "configuration.embedding.provider",
            },
        )

    async def test_create_run_saves_multiple_answer_metrics(self) -> None:
        """Verify one run snapshot retains every selected answer metric.

        Args:
            None.

        Returns:
            None. Assertions verify multi-metric API and persistence behavior.
        """
        payload = _run_payload()
        configuration = payload["configuration"]
        configuration["evaluation"] = {
            "retrieval_metrics": [],
            "answer_metrics": ["groundedness", "answer_relevance"],
        }
        response = await self._post_run(payload)

        # Return both compatible metrics in the immutable effective configuration.
        self.assertEqual(response.status_code, 201)
        self.assertEqual(
            response.json()["configuration"]["evaluation"]["answer_metrics"],
            ["groundedness", "answer_relevance"],
        )

        with sqlite3.connect(self.database_path) as connection:
            stored_configuration = connection.execute(
                "SELECT effective_config_json FROM pipeline_run"
            ).fetchone()[0]

        self.assertEqual(
            json.loads(stored_configuration)["evaluation"]["answer_metrics"],
            ["groundedness", "answer_relevance"],
        )

    async def test_create_run_rejects_unknown_evaluation_metric(self) -> None:
        """Verify evaluation selections are controlled by the backend catalog.

        Args:
            None.

        Returns:
            None. Assertions verify a structured compatibility failure.
        """
        payload = _run_payload()
        configuration = payload["configuration"]
        configuration["evaluation"] = {
            "answer_metrics": ["unknown_metric"],
        }
        response = await self._post_run(payload)

        # Identify the incompatible metric list without exposing internal details.
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["detail"],
            {
                "code": "invalid_pipeline_configuration",
                "message": "Evaluation metric 'unknown_metric' is not supported.",
                "field": "configuration.evaluation.answer_metrics",
            },
        )

    async def test_create_run_rejects_metrics_without_required_labels(self) -> None:
        """Verify single-question runs reject metrics requiring dataset labels.

        Args:
            None.

        Returns:
            None. Assertions verify retrieval and reference-answer restrictions.
        """
        retrieval_payload = _run_payload()
        retrieval_configuration = retrieval_payload["configuration"]
        retrieval_configuration["evaluation"] = {
            "retrieval_metrics": ["hit_rate_at_k"],
        }
        retrieval_response = await self._post_run(retrieval_payload)

        # Retrieval evaluation needs manually labelled relevant documents.
        self.assertEqual(retrieval_response.status_code, 422)
        self.assertEqual(
            retrieval_response.json()["detail"]["field"],
            "configuration.evaluation.retrieval_metrics",
        )

        correctness_payload = _run_payload()
        correctness_configuration = correctness_payload["configuration"]
        correctness_configuration["evaluation"] = {
            "answer_metrics": ["answer_correctness"],
        }
        correctness_response = await self._post_run(correctness_payload)

        # Answer correctness cannot run without a labelled reference answer.
        self.assertEqual(correctness_response.status_code, 422)
        self.assertEqual(
            correctness_response.json()["detail"]["field"],
            "configuration.evaluation.answer_metrics",
        )

    async def test_create_run_does_not_deduplicate_identical_submissions(self) -> None:
        """Verify every user submission creates a distinct immutable run.

        Args:
            None.

        Returns:
            None. Assertions verify separate run identities and rows.
        """
        first_response = await self._post_run(_run_payload())
        second_response = await self._post_run(_run_payload())

        # Preserve click-level history even when the input and configuration match.
        self.assertEqual(first_response.status_code, 201)
        self.assertEqual(second_response.status_code, 201)
        self.assertNotEqual(first_response.json()["id"], second_response.json()["id"])
        self.assertEqual(
            first_response.json()["chunking"]["chunk_set_id"],
            second_response.json()["chunking"]["chunk_set_id"],
        )
        self.assertFalse(first_response.json()["chunking"]["reused"])
        self.assertTrue(second_response.json()["chunking"]["reused"])

        with sqlite3.connect(self.database_path) as connection:
            run_count = connection.execute(
                "SELECT COUNT(*) FROM pipeline_run"
            ).fetchone()[0]
            chunk_set_count = connection.execute(
                "SELECT COUNT(*) FROM chunk_set"
            ).fetchone()[0]

        self.assertEqual(run_count, 2)
        self.assertEqual(chunk_set_count, 1)
