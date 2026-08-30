"""API and executor tests for queued immutable pipeline runs."""

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
from backend.db.repositories.runs import (
    InvalidRunStateError,
    claim_next_pending_run,
    record_retrieval_result,
)
from backend.embedding.models import (
    EmbeddingBatch,
    EmbeddingInputPurpose,
    EmbeddingProviderUnavailableError,
    VectorSearchHit,
    VectorStoreError,
)
from backend.generation.models import (
    GenerationAuthenticationError,
    GenerationMessage,
    GenerationProviderResponse,
)
from backend.ingestion.chunkers.models import TokenizedText, TokenOffset
from backend.ingestion.chunkers.tokenizer import TokenizerAssetError
from backend.pipeline.execution import PipelineExecutor, PipelineRunExecutionError


class RunTestTokenizer:
    """Provide deterministic offline token offsets to pipeline execution tests."""

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


class RunTestEmbeddingProvider:
    """Return deterministic fixed-width vectors without a network call."""

    identifier = "run-test-embedding-provider"
    version = "1"

    def input_policy_version(self, model: str) -> str:
        """Return the stable text policy used by this fake provider.

        Args:
            model: Provider model identifier selected by the run.

        Returns:
            Fixed fake policy version.
        """
        # Every fake model uses the same unmodified text policy.
        return "raw-test-v1"

    def embed(
        self,
        model: str,
        texts: list[str],
        purpose: EmbeddingInputPurpose,
    ) -> EmbeddingBatch:
        """Create one deterministic three-dimensional vector per text.

        Args:
            model: Provider model identifier selected by the run.
            texts: Ordered chunk texts being embedded.
            purpose: Document or query purpose for the request.

        Returns:
            Three-dimensional vectors aligned with the submitted texts.
        """
        # Text length keeps vectors deterministic while purpose verifies the interface.
        purpose_coordinate = 1.0 if purpose is EmbeddingInputPurpose.DOCUMENT else 2.0
        vectors = tuple(
            (float(len(text)), float(index + 1), purpose_coordinate)
            for index, text in enumerate(texts)
        )
        return EmbeddingBatch(
            vectors=vectors,
            dimensions=3,
            provider_model=model,
            provider_revision="test-revision",
        )


class UnavailableRunTestEmbeddingProvider(RunTestEmbeddingProvider):
    """Simulate an embedding HTTP service that cannot be reached."""

    def embed(
        self,
        model: str,
        texts: list[str],
        purpose: EmbeddingInputPurpose,
    ) -> EmbeddingBatch:
        """Raise the provider-neutral unavailability failure.

        Args:
            model: Provider model identifier selected by the run.
            texts: Ordered chunk texts that cannot be embedded.
            purpose: Document or query purpose for the request.

        Returns:
            Never returns because the fake service is unavailable.

        Raises:
            EmbeddingProviderUnavailableError: Always.
        """
        # The executor must persist a safe error rather than expose transport internals.
        raise EmbeddingProviderUnavailableError("simulated connection failure")


class RunTestGenerationProvider:
    """Return one deterministic answer without a paid Groq API request."""

    identifier = "run-test-generation-provider"
    version = "1"

    def policy_version(self, model: str) -> str:
        """Return the stable request policy used by this fake provider.

        Args:
            model: Generation model identifier selected by the run.

        Returns:
            Fixed provider-policy version for persistence assertions.
        """
        # Every fake model uses the same request behavior.
        return "run-test-generation-v1"

    def generate(
        self,
        model: str,
        messages: tuple[GenerationMessage, ...],
        temperature: float,
        max_output_tokens: int,
    ) -> GenerationProviderResponse:
        """Return a source-citing answer for the supplied deterministic prompt.

        Args:
            model: Generation model identifier selected by the run.
            messages: Ordered prompt messages containing retrieved context.
            temperature: Sampling temperature retained by the request contract.
            max_output_tokens: Requested completion limit retained by the contract.

        Returns:
            Fixed answer and usage provenance suitable for pipeline tests.
        """
        # The final prompt must carry the persisted refund source for this fixture.
        if "Refunds are processed" not in messages[-1].content:
            raise AssertionError("The generation prompt omitted retrieved context.")

        return GenerationProviderResponse(
            answer_text="Refunds take five business days. [Source 1]",
            provider_model=model,
            finish_reason="stop",
            prompt_tokens=24,
            completion_tokens=8,
            total_tokens=32,
            provider_request_id="request-test-1",
            system_fingerprint="fingerprint-test-1",
        )


class UnavailableRunTestGenerationProvider(RunTestGenerationProvider):
    """Simulate missing or rejected Groq backend credentials."""

    def generate(
        self,
        model: str,
        messages: tuple[GenerationMessage, ...],
        temperature: float,
        max_output_tokens: int,
    ) -> GenerationProviderResponse:
        """Raise the provider-neutral authentication failure.

        Args:
            model: Generation model that cannot be requested.
            messages: Prompt messages that cannot be submitted.
            temperature: Sampling value that cannot be submitted.
            max_output_tokens: Completion limit that cannot be submitted.

        Returns:
            Never returns because credentials are unavailable.

        Raises:
            GenerationAuthenticationError: Always.
        """
        # The executor must preserve retrieval while sanitizing credential errors.
        raise GenerationAuthenticationError("simulated rejected key")


class RunTestVectorStore:
    """Persist explicit vectors in memory for deterministic executor tests."""

    identifier = "run-test-vector-store"
    version = "1"

    def __init__(self) -> None:
        """Create the isolated in-memory collection registry.

        Args:
            None.

        Returns:
            None. The store begins with no collections.
        """
        self.collections: dict[str, dict[str, list[object]]] = {}

    def create_collection(self, name: str, distance_metric: str) -> None:
        """Create one empty named test collection.

        Args:
            name: Unique collection identifier.
            distance_metric: Configured distance metric retained by production only.

        Returns:
            None. An empty collection is registered.
        """
        # Keep only record arrays required to verify aligned writes and counts.
        self.collections[name] = {"ids": [], "vectors": [], "metadata": []}

    def add(
        self,
        collection_name: str,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict[str, str | int | float | bool]],
    ) -> None:
        """Append one aligned vector batch to the named collection.

        Args:
            collection_name: Existing collection receiving records.
            ids: Stable chunk identifiers.
            vectors: Explicit provider-generated vectors.
            metadata: Scalar provenance aligned with the identifiers.

        Returns:
            None. Records are retained in memory.
        """
        collection = self.collections[collection_name]
        collection["ids"].extend(ids)
        collection["vectors"].extend(vectors)
        collection["metadata"].extend(metadata)

    def count(self, collection_name: str) -> int:
        """Return the number of vector identifiers in one collection.

        Args:
            collection_name: Existing collection to inspect.

        Returns:
            Number of stored records.
        """
        # IDs are the authoritative one-to-one vector record identity.
        return len(self.collections[collection_name]["ids"])

    def query(
        self,
        collection_name: str,
        vector: list[float],
        top_k: int,
    ) -> tuple[VectorSearchHit, ...]:
        """Return deterministic ranked chunk IDs from one test collection.

        Args:
            collection_name: Existing collection containing the indexed chunks.
            vector: Query vector whose shape was validated before this call.
            top_k: Maximum number of stored chunk IDs to return.

        Returns:
            Ranked vector hits with deterministic raw cosine distances.
        """
        # Preserve insertion order as deterministic nearest-neighbor order for tests.
        chunk_ids = self.collections[collection_name]["ids"][:top_k]
        return tuple(
            VectorSearchHit(
                chunk_id=str(chunk_id),
                raw_distance=float(index) / 10,
            )
            for index, chunk_id in enumerate(chunk_ids)
        )

    def delete_collection(self, name: str) -> None:
        """Delete one test collection during rollback or reuse races.

        Args:
            name: Exact collection identifier to remove.

        Returns:
            None. A missing collection is treated as already removed.
        """
        # Match the production adapter's idempotent cleanup behavior.
        self.collections.pop(name, None)


class UnavailableRunTestVectorStore(RunTestVectorStore):
    """Build indexes successfully but simulate a retrieval query failure."""

    def query(
        self,
        collection_name: str,
        vector: list[float],
        top_k: int,
    ) -> tuple[VectorSearchHit, ...]:
        """Raise the provider-neutral vector-store retrieval failure.

        Args:
            collection_name: Existing collection that cannot be searched.
            vector: Valid query vector submitted by the executor.
            top_k: Maximum result count requested by retrieval.

        Returns:
            Never returns because the simulated store is unavailable.

        Raises:
            VectorStoreError: Always, to exercise retrieval failure handling.
        """
        # Hide the simulated internal message behind the executor's safe error mapping.
        raise VectorStoreError("simulated vector query failure")


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
                "provider": "groq",
                "model": "openai/gpt-oss-20b",
            },
        },
    }


class PipelineRunRouteTestCase(unittest.IsolatedAsyncioTestCase):
    """Exercise queue, polling, and execution with isolated SQLite data."""

    def setUp(self) -> None:
        """Create isolated persistence and deterministic pipeline adapters.

        Args:
            None.

        Returns:
            None. Temporary resources remain owned until teardown.
        """
        # Redirect repository connections away from the developer's local database.
        self.database_directory = TemporaryDirectory()
        self.database_path = Path(self.database_directory.name) / "runs.sqlite3"
        self.database_patch = patch(
            "backend.db.connection.DATABASE_PATH",
            self.database_path,
        )
        self.database_patch.start()
        self.vector_store = RunTestVectorStore()
        self.executor = PipelineExecutor(
            tokenizer=RunTestTokenizer(),
            embedding_provider=RunTestEmbeddingProvider(),
            generation_provider=RunTestGenerationProvider(),
            vector_store=self.vector_store,
        )

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
                    48,
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

    def tearDown(self) -> None:
        """Restore the database path and remove temporary test data.

        Args:
            None.

        Returns:
            None. Resources created by setup are released.
        """
        # Stop patching before deleting the temporary directory it references.
        self.database_patch.stop()
        self.database_directory.cleanup()

    async def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> Response:
        """Send one request through the complete FastAPI application.

        Args:
            method: HTTP method used for the request.
            path: Application-relative route path.
            payload: Optional JSON-compatible request body.

        Returns:
            HTTPX response returned by the application.
        """
        transport = ASGITransport(app=app)

        # Use an in-process client so API contract tests remain fast and offline.
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.request(method, path, json=payload)

    def _execute_next_run(self) -> dict[str, object]:
        """Claim and synchronously execute the oldest pending test run.

        Args:
            None.

        Returns:
            Completed run produced by the deterministic executor.
        """
        claimed_run = claim_next_pending_run()

        # A test calling this helper must have enqueued exactly one available run.
        self.assertIsNotNone(claimed_run)
        return self.executor.execute(claimed_run["id"])

    async def test_enqueue_and_poll_completed_run(self) -> None:
        """Verify queued creation and completed chunk/vector provenance.

        Args:
            None.

        Returns:
            None. Assertions verify both API lifecycle representations.
        """
        enqueue_response = await self._request("POST", "/runs", _run_payload())

        # Enqueueing returns immediately with resolved immutable configuration.
        self.assertEqual(enqueue_response.status_code, 202)
        pending_run = enqueue_response.json()
        self.assertEqual(pending_run["question"], "What is the refund policy?")
        self.assertEqual(pending_run["status"], "pending")
        self.assertEqual(pending_run["chunking"]["status"], "pending")
        self.assertEqual(pending_run["embedding"]["status"], "pending")
        self.assertEqual(pending_run["configuration"]["retrieval"]["top_k"], 10)

        self._execute_next_run()
        poll_response = await self._request("GET", f"/runs/{pending_run['id']}")
        completed_run = poll_response.json()

        # Polling exposes both exact reusable artifacts only after they are ready.
        self.assertEqual(poll_response.status_code, 200)
        self.assertEqual(completed_run["status"], "completed")
        self.assertEqual(completed_run["chunking"]["status"], "completed")
        self.assertGreater(completed_run["chunking"]["chunk_count"], 0)
        self.assertEqual(completed_run["embedding"]["status"], "completed")
        self.assertEqual(completed_run["embedding"]["dimensions"], 3)
        self.assertEqual(
            completed_run["embedding"]["vector_count"],
            completed_run["chunking"]["chunk_count"],
        )
        self.assertEqual(completed_run["retrieval"]["status"], "completed")
        self.assertEqual(completed_run["retrieval"]["returned_count"], 1)
        self.assertEqual(
            completed_run["retrieval"]["chunks"][0]["original_filename"],
            "refunds.txt",
        )
        self.assertEqual(completed_run["generation"]["status"], "completed")
        self.assertEqual(
            completed_run["generation"]["answer"],
            "Refunds take five business days. [Source 1]",
        )
        self.assertEqual(completed_run["generation"]["total_tokens"], 32)
        self.assertEqual(
            completed_run["generation"]["context_chunks"][0]["retrieval_rank"],
            1,
        )

        # Retrieval must persist ranked references before the run becomes completed.
        with connect() as connection:
            retrieval_result = connection.execute(
                """
                SELECT requested_top_k, returned_count, distance_metric, duration_ms
                FROM retrieval_result WHERE pipeline_run_id = ?
                """,
                (completed_run["id"],),
            ).fetchone()
            retrieved_chunks = connection.execute(
                """
                SELECT rank, chunk_id, raw_distance FROM retrieved_chunk
                ORDER BY rank
                """
            ).fetchall()

        self.assertEqual(retrieval_result["requested_top_k"], 10)
        self.assertEqual(retrieval_result["returned_count"], 1)
        self.assertEqual(retrieval_result["distance_metric"], "cosine")
        self.assertGreaterEqual(retrieval_result["duration_ms"], 0)
        self.assertEqual(len(retrieved_chunks), 1)
        self.assertEqual(retrieved_chunks[0]["rank"], 1)

        # The saved snapshot must exactly match the resolved API configuration.
        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT question, effective_config_json FROM pipeline_run"
            ).fetchone()

        self.assertEqual(row[0], "What is the refund policy?")
        self.assertEqual(json.loads(row[1]), completed_run["configuration"])

    async def test_poll_rejects_unknown_run(self) -> None:
        """Verify polling an unknown stable ID returns a structured 404.

        Args:
            None.

        Returns:
            None. Assertions verify the public not-found contract.
        """
        response = await self._request("GET", "/runs/missing-run")

        # Keep missing run identity distinct from provider or execution failures.
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"]["code"], "run_not_found")

    async def test_poll_accepts_active_retrieval_stage(self) -> None:
        """Verify the runs API contract represents retrieval lifecycle state.

        Args:
            None.

        Returns:
            None. Assertions verify retrieval is accepted as a current stage.
        """
        enqueue_response = await self._request("POST", "/runs", _run_payload())
        claimed_run = claim_next_pending_run()

        # Isolate the transport enum without running provider work in this API test.
        with connect() as connection:
            connection.execute(
                """
                UPDATE pipeline_run SET current_stage = 'retrieval'
                WHERE id = ? AND status = 'running'
                """,
                (claimed_run["id"],),
            )

        poll_response = await self._request(
            "GET",
            f"/runs/{enqueue_response.json()['id']}",
        )

        self.assertEqual(poll_response.status_code, 200)
        self.assertEqual(poll_response.json()["current_stage"], "retrieval")

    async def test_poll_accepts_active_generation_stage(self) -> None:
        """Verify the API contract represents the active generation lifecycle.

        Args:
            None.

        Returns:
            None. Assertions verify generation is an accepted current stage.
        """
        enqueue_response = await self._request("POST", "/runs", _run_payload())
        claimed_run = claim_next_pending_run()

        # Isolate the transport enum without invoking any external provider.
        with connect() as connection:
            connection.execute(
                """
                UPDATE pipeline_run SET current_stage = 'generation'
                WHERE id = ? AND status = 'running'
                """,
                (claimed_run["id"],),
            )

        poll_response = await self._request(
            "GET",
            f"/runs/{enqueue_response.json()['id']}",
        )

        self.assertEqual(poll_response.status_code, 200)
        self.assertEqual(poll_response.json()["current_stage"], "generation")
        self.assertEqual(poll_response.json()["generation"]["status"], "running")

    async def test_enqueue_rejects_invalid_inputs(self) -> None:
        """Verify blank questions and unknown corpora never enter the queue.

        Args:
            None.

        Returns:
            None. Assertions verify request-level error categories.
        """
        blank_payload = _run_payload()
        blank_payload["question"] = "   "
        blank_response = await self._request("POST", "/runs", blank_payload)
        missing_payload = _run_payload()
        missing_payload["corpus_id"] = "missing-corpus"
        missing_response = await self._request("POST", "/runs", missing_payload)

        # Neither invalid request should create an auditable execution row.
        self.assertEqual(blank_response.status_code, 422)
        self.assertEqual(blank_response.json()["detail"]["code"], "invalid_question")
        self.assertEqual(missing_response.status_code, 404)
        self.assertEqual(
            missing_response.json()["detail"]["code"],
            "corpus_not_found",
        )

    async def test_enqueue_rejects_unsupported_provider(self) -> None:
        """Verify the backend catalog controls executable provider identifiers.

        Args:
            None.

        Returns:
            None. Assertions verify semantic compatibility validation.
        """
        payload = _run_payload()
        configuration = payload["configuration"]
        configuration["embedding"]["provider"] = "unknown"
        response = await self._request("POST", "/runs", payload)

        # Reject an unregistered adapter before persisting an unusable run.
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["detail"],
            {
                "code": "invalid_pipeline_configuration",
                "message": "Provider 'unknown' is not supported.",
                "field": "configuration.embedding.provider",
            },
        )

    async def test_chunking_failure_is_persisted_for_polling(self) -> None:
        """Verify missing canonical input becomes a safe terminal run failure.

        Args:
            None.

        Returns:
            None. Assertions verify stage state and structured error provenance.
        """
        # Remove only the parse so the selected corpus remains valid but unchunkable.
        with connect() as connection:
            connection.execute("DELETE FROM document_parse WHERE id = ?", ("parse-1",))

        enqueue_response = await self._request("POST", "/runs", _run_payload())
        run_id = enqueue_response.json()["id"]
        claimed_run = claim_next_pending_run()

        # The executor raises to its worker while retaining a pollable failed row.
        with self.assertRaises(PipelineRunExecutionError):
            self.executor.execute(claimed_run["id"])

        poll_response = await self._request("GET", f"/runs/{run_id}")
        failed_run = poll_response.json()
        self.assertEqual(failed_run["status"], "failed")
        self.assertEqual(failed_run["chunking"]["status"], "failed")
        self.assertEqual(failed_run["embedding"]["status"], "pending")
        self.assertEqual(failed_run["error"]["code"], "missing_parse_artifact")
        self.assertEqual(failed_run["error"]["stage"], "chunking")

    async def test_tokenizer_failure_is_sanitized(self) -> None:
        """Verify tokenizer details do not leak through the polling contract.

        Args:
            None.

        Returns:
            None. Assertions verify safe persisted failure text.
        """
        self.executor = PipelineExecutor(
            tokenizer=UnavailableRunTestTokenizer(),
            embedding_provider=RunTestEmbeddingProvider(),
            vector_store=self.vector_store,
        )
        enqueue_response = await self._request("POST", "/runs", _run_payload())
        claimed_run = claim_next_pending_run()

        # A missing backend asset is terminal for this run but not for the worker.
        with self.assertRaises(PipelineRunExecutionError):
            self.executor.execute(claimed_run["id"])

        poll_response = await self._request(
            "GET",
            f"/runs/{enqueue_response.json()['id']}",
        )
        failure = poll_response.json()["error"]
        self.assertEqual(failure["code"], "chunking_tokenizer_unavailable")
        self.assertNotIn("simulated", failure["message"])

    async def test_embedding_provider_failure_preserves_ready_chunks(self) -> None:
        """Verify an unreachable model API fails embedding without losing chunks.

        Args:
            None.

        Returns:
            None. Assertions verify graceful failure and upstream provenance.
        """
        self.executor = PipelineExecutor(
            tokenizer=RunTestTokenizer(),
            embedding_provider=UnavailableRunTestEmbeddingProvider(),
            vector_store=self.vector_store,
        )
        enqueue_response = await self._request("POST", "/runs", _run_payload())
        claimed_run = claim_next_pending_run()

        # Provider availability is checked only by the real embedding request.
        with self.assertRaises(PipelineRunExecutionError):
            self.executor.execute(claimed_run["id"])

        poll_response = await self._request(
            "GET",
            f"/runs/{enqueue_response.json()['id']}",
        )
        failed_run = poll_response.json()
        self.assertEqual(failed_run["status"], "failed")
        self.assertEqual(failed_run["chunking"]["status"], "completed")
        self.assertEqual(failed_run["embedding"]["status"], "failed")
        self.assertEqual(failed_run["error"]["stage"], "embedding")
        self.assertEqual(failed_run["error"]["code"], "embedding_provider_unavailable")
        self.assertIsNone(failed_run["embedding"]["vector_index_id"])

    async def test_identical_runs_reuse_compatible_artifacts(self) -> None:
        """Verify click history stays distinct while artifacts remain reusable.

        Args:
            None.

        Returns:
            None. Assertions verify chunk and vector compatibility reuse.
        """
        first_response = await self._request("POST", "/runs", _run_payload())
        first_run = self._execute_next_run()
        second_response = await self._request("POST", "/runs", _run_payload())
        second_run = self._execute_next_run()

        # Every click has a run ID while identical compatibility inputs share artifacts.
        self.assertNotEqual(first_response.json()["id"], second_response.json()["id"])
        self.assertEqual(
            first_run["chunking"]["chunk_set_id"],
            second_run["chunking"]["chunk_set_id"],
        )
        self.assertEqual(
            first_run["embedding"]["vector_index_id"],
            second_run["embedding"]["vector_index_id"],
        )
        self.assertFalse(first_run["chunking"]["reused"])
        self.assertTrue(second_run["chunking"]["reused"])
        self.assertFalse(first_run["embedding"]["reused"])
        self.assertTrue(second_run["embedding"]["reused"])

    async def test_retrieval_failure_preserves_ready_upstream_artifacts(self) -> None:
        """Verify a vector query failure leaves no partial retrieval result.

        Args:
            None.

        Returns:
            None. Assertions verify retrieval failure state and artifact retention.
        """
        failing_store = UnavailableRunTestVectorStore()
        self.executor = PipelineExecutor(
            tokenizer=RunTestTokenizer(),
            embedding_provider=RunTestEmbeddingProvider(),
            vector_store=failing_store,
        )
        enqueue_response = await self._request("POST", "/runs", _run_payload())
        claimed_run = claim_next_pending_run()

        # Retrieval fails only after ready chunk and vector artifacts are attached.
        with self.assertRaises(PipelineRunExecutionError):
            self.executor.execute(claimed_run["id"])

        poll_response = await self._request(
            "GET",
            f"/runs/{enqueue_response.json()['id']}",
        )
        failed_run = poll_response.json()
        self.assertEqual(failed_run["status"], "failed")
        self.assertEqual(failed_run["chunking"]["status"], "completed")
        self.assertEqual(failed_run["embedding"]["status"], "completed")
        self.assertEqual(failed_run["error"]["stage"], "retrieval")
        self.assertEqual(
            failed_run["error"]["code"],
            "retrieval_vector_store_unavailable",
        )

        # A failed search must not leave a result parent or ranked children behind.
        with connect() as connection:
            result_count = connection.execute(
                "SELECT COUNT(*) FROM retrieval_result"
            ).fetchone()[0]
            hit_count = connection.execute(
                "SELECT COUNT(*) FROM retrieved_chunk"
            ).fetchone()[0]
        self.assertEqual((result_count, hit_count), (0, 0))

    async def test_generation_failure_preserves_retrieval_result(self) -> None:
        """Verify rejected Groq credentials retain all completed upstream output.

        Args:
            None.

        Returns:
            None. Assertions cover structured failure and retrieval preservation.
        """
        self.executor = PipelineExecutor(
            tokenizer=RunTestTokenizer(),
            embedding_provider=RunTestEmbeddingProvider(),
            generation_provider=UnavailableRunTestGenerationProvider(),
            vector_store=self.vector_store,
        )
        enqueue_response = await self._request("POST", "/runs", _run_payload())
        claimed_run = claim_next_pending_run()

        # Generation fails only after the complete retrieval result is committed.
        with self.assertRaises(PipelineRunExecutionError):
            self.executor.execute(claimed_run["id"])

        poll_response = await self._request(
            "GET",
            f"/runs/{enqueue_response.json()['id']}",
        )
        failed_run = poll_response.json()
        self.assertEqual(failed_run["status"], "failed")
        self.assertEqual(failed_run["retrieval"]["status"], "completed")
        self.assertEqual(failed_run["retrieval"]["returned_count"], 1)
        self.assertEqual(failed_run["generation"]["status"], "failed")
        self.assertEqual(failed_run["error"]["stage"], "generation")
        self.assertEqual(
            failed_run["error"]["code"],
            "generation_authentication_failed",
        )

        # No answer rows may remain after a terminal generation failure.
        with connect() as connection:
            generation_count = connection.execute(
                "SELECT COUNT(*) FROM generation_result"
            ).fetchone()[0]
        self.assertEqual(generation_count, 0)

    async def test_generation_context_write_failure_rolls_back_answer(self) -> None:
        """Verify answer and context rows share the completion transaction.

        Args:
            None.

        Returns:
            None. Assertions cover rollback and retained retrieval provenance.
        """
        # Abort the child context write after its answer parent is inserted.
        with connect() as connection:
            connection.execute(
                """
                CREATE TRIGGER fail_generation_context
                BEFORE INSERT ON generation_context_chunk
                BEGIN
                    SELECT RAISE(ABORT, 'simulated context write failure');
                END
                """
            )

        enqueue_response = await self._request("POST", "/runs", _run_payload())
        claimed_run = claim_next_pending_run()

        # The executor converts the relational failure into a safe terminal state.
        with self.assertRaises(PipelineRunExecutionError):
            self.executor.execute(claimed_run["id"])

        failed_run = (
            await self._request(
                "GET",
                f"/runs/{enqueue_response.json()['id']}",
            )
        ).json()
        self.assertEqual(failed_run["error"]["code"], "generation_persistence_failed")
        self.assertEqual(failed_run["retrieval"]["status"], "completed")
        self.assertEqual(failed_run["generation"]["status"], "failed")

        # Transaction rollback removes the parent created before the failed child.
        with connect() as connection:
            counts = (
                connection.execute("SELECT COUNT(*) FROM generation_result").fetchone()[
                    0
                ],
                connection.execute(
                    "SELECT COUNT(*) FROM generation_context_chunk"
                ).fetchone()[0],
            )
        self.assertEqual(counts, (0, 0))

    async def test_completed_run_rejects_second_completion(self) -> None:
        """Verify terminal run history cannot be overwritten by a retry.

        Args:
            None.

        Returns:
            None. Assertions verify the repository lifecycle guard.
        """
        await self._request("POST", "/runs", _run_payload())
        completed_run = self._execute_next_run()

        # Repeating an earlier retrieval transition must fail on terminal history.
        with self.assertRaises(InvalidRunStateError):
            record_retrieval_result(
                completed_run["id"],
                completed_run["embedding"]["vector_index_id"],
                10,
                "cosine",
                (),
                1,
            )
