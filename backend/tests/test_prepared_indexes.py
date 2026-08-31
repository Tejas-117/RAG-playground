"""API, repository, queue, and executor tests for named prepared indexes."""

import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient, Response

from backend.app import app
from backend.db.connection import connect
from backend.db.repositories.chunk_sets import save_ready_chunk_set
from backend.db.repositories.prepared_indexes import (
    claim_next_pending_prepared_index,
    get_prepared_index,
)
from backend.db.repositories.vector_indexes import save_ready_vector_index
from backend.db.repositories.work_queue import claim_next_pending_work_item
from backend.embedding.models import EmbeddingProviderUnavailableError
from backend.embedding.service import VectorIndexBuildResult
from backend.ingestion.chunk_sets import ChunkSetBuildResult
from backend.pipeline.preparation import (
    PreparedIndexExecutionError,
    PreparedIndexExecutor,
)


def _index_payload(name: str = "Product index") -> dict[str, object]:
    """Build one valid preparation API payload with resolved default inputs.

    Args:
        name: User-facing prepared-index label.

    Returns:
        JSON-compatible request body for ``POST /indexes``.
    """
    # Keep the fixture aligned with the backend-owned pipeline option catalog.
    return {
        "name": name,
        "corpus_id": "corpus-1",
        "configuration": {
            "chunking": {
                "strategy": "recursive",
                "chunk_size_tokens": 800,
                "chunk_overlap_tokens": 100,
            },
            "embedding": {
                "provider": "ollama",
                "model": "nomic-embed-text",
                "distance_metric": "cosine",
            },
        },
    }


class PreparedIndexTestArtifacts:
    """Create deterministic ready artifacts without model or Chroma calls."""

    def build_chunk_set(
        self,
        corpus_id: str,
        configuration: Any,
        tokenizer: Any,
    ) -> ChunkSetBuildResult:
        """Persist and return one ready chunk artifact.

        Args:
            corpus_id: Corpus selected by the preparation executor.
            configuration: Typed chunking configuration from the snapshot.
            tokenizer: Optional tokenizer override, unused by this fake.

        Returns:
            Newly-created ready chunk artifact marked as non-reused.
        """
        # Validate that the executor forwards the immutable preparation inputs.
        if corpus_id != "corpus-1" or configuration.chunk_size_tokens != 800:
            raise AssertionError("The preparation executor changed chunking inputs.")

        artifact = {
            "id": "chunk-set-prepared-1",
            "corpus_id": corpus_id,
            "fingerprint": "prepared-chunk-fingerprint",
            "chunking_config_json": configuration.model_dump_json(),
            "chunker_name": "test-chunker",
            "chunker_version": "1",
            "created_at": "2026-08-31T00:00:01Z",
            "started_at": "2026-08-31T00:00:01Z",
            "completed_at": "2026-08-31T00:00:02Z",
            "duration_ms": 1,
        }
        chunks = [
            {
                "id": "prepared-chunk-1",
                "source_document_id": "document-1",
                "ordinal": 0,
                "text": "Prepared index content.",
                "character_start_offset": 0,
                "character_end_offset": 23,
                "token_start_offset": 0,
                "token_end_offset": 3,
                "page_start": None,
                "page_end": None,
                "source_metadata_json": "{}",
            }
        ]
        save_ready_chunk_set(artifact, chunks)

        # Services return materialized chunks for the next embedding stage.
        return ChunkSetBuildResult(
            artifact={**artifact, "status": "ready", "chunks": chunks},
            reused=False,
        )

    def build_vector_index(
        self,
        chunk_set: dict[str, Any],
        configuration: Any,
        provider: Any,
        vector_store: Any,
    ) -> VectorIndexBuildResult:
        """Persist and return one compatible ready vector artifact.

        Args:
            chunk_set: Exact ready chunk artifact produced by the first stage.
            configuration: Typed embedding configuration from the snapshot.
            provider: Optional provider override, unused by this fake.
            vector_store: Optional vector-store override, unused by this fake.

        Returns:
            Newly-created ready vector artifact marked as non-reused.
        """
        # Preserve exact chunk lineage in the relational artifact fixture.
        artifact = {
            "id": "vector-index-prepared-1",
            "chunk_set_id": chunk_set["id"],
            "fingerprint": "prepared-vector-fingerprint",
            "embedding_config_json": configuration.model_dump_json(),
            "provider": configuration.provider,
            "model": configuration.model,
            "provider_model": configuration.model,
            "provider_revision": "test-revision",
            "dimensions": 3,
            "distance_metric": configuration.distance_metric.value,
            "input_policy_version": "test-policy-v1",
            "indexer_name": "test-store",
            "indexer_version": "1",
            "collection_name": "prepared-test-collection",
            "vector_count": 1,
            "created_at": "2026-08-31T00:00:02Z",
            "started_at": "2026-08-31T00:00:02Z",
            "completed_at": "2026-08-31T00:00:03Z",
            "duration_ms": 1,
        }
        save_ready_vector_index(artifact)
        return VectorIndexBuildResult(
            artifact={**artifact, "status": "ready"},
            reused=False,
        )

    def fail_vector_index(
        self,
        chunk_set: dict[str, Any],
        configuration: Any,
        provider: Any,
        vector_store: Any,
    ) -> VectorIndexBuildResult:
        """Simulate an unavailable embedding provider after chunking succeeds.

        Args:
            chunk_set: Ready upstream artifact retained by the failed request.
            configuration: Embedding configuration that cannot be executed.
            provider: Optional provider override, unused by this fake.
            vector_store: Optional vector-store override, unused by this fake.

        Returns:
            Never returns because the provider is unavailable.

        Raises:
            EmbeddingProviderUnavailableError: Always for failure mapping.
        """
        # Reference inputs so strict static checks recognize this intentional fake.
        _ = (chunk_set, configuration, provider, vector_store)
        raise EmbeddingProviderUnavailableError("simulated unavailable provider")


class PreparedIndexRouteTestCase(unittest.IsolatedAsyncioTestCase):
    """Exercise named-index routes and execution with isolated SQLite data."""

    def setUp(self) -> None:
        """Create isolated persistence and seed one immutable parsed corpus.

        Args:
            None.

        Returns:
            None. Temporary resources remain owned until teardown.
        """
        self.database_directory = TemporaryDirectory()
        self.database_path = Path(self.database_directory.name) / "indexes.sqlite3"
        self.database_patch = patch(
            "backend.db.connection.DATABASE_PATH",
            self.database_path,
        )
        self.database_patch.start()

        # Seed the source hierarchy required by fake reusable chunk artifacts.
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
            connection.execute(
                "INSERT INTO document VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "document-1",
                    "corpus-1",
                    "product.txt",
                    "uploads/product.txt",
                    "text/plain",
                    23,
                    "a" * 64,
                    "2026-08-31T00:00:01Z",
                ),
            )

    def tearDown(self) -> None:
        """Restore the database path and release temporary resources.

        Args:
            None.

        Returns:
            None. Test persistence is deleted with its temporary directory.
        """
        # Stop patching before the referenced temporary directory is removed.
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
            method: HTTP method for the request.
            path: Application-relative resource path.
            payload: Optional JSON-compatible request body.

        Returns:
            HTTPX response returned by the in-process application.
        """
        transport = ASGITransport(app=app)

        # The in-process client keeps API contract tests offline and deterministic.
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, json=payload)

    async def test_create_list_and_read_duplicate_named_indexes(self) -> None:
        """Verify duplicate labels create distinct selectable preparation records.

        Args:
            None.

        Returns:
            None. Assertions verify create, list, filter, and detail contracts.
        """
        first_response = await self._request("POST", "/indexes", _index_payload())
        second_response = await self._request("POST", "/indexes", _index_payload())

        # Duplicate names remain distinct because stable IDs are the selection value.
        self.assertEqual(first_response.status_code, 202)
        self.assertEqual(second_response.status_code, 202)
        first_index = first_response.json()
        second_index = second_response.json()
        self.assertNotEqual(first_index["id"], second_index["id"])
        self.assertEqual(first_index["name"], second_index["name"])
        self.assertEqual(first_index["status"], "pending")
        self.assertEqual(first_index["chunking"]["status"], "pending")
        self.assertEqual(first_index["embedding"]["provider"], "ollama")

        list_response = await self._request("GET", "/indexes?status=pending")
        detail_response = await self._request(
            "GET",
            f"/indexes/{first_index['id']}",
        )

        # Listing includes both identities and detail polling preserves the snapshot.
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(len(list_response.json()), 2)
        self.assertEqual(detail_response.status_code, 200)
        self.assertEqual(
            detail_response.json()["configuration"], _index_payload()["configuration"]
        )

    async def test_create_rejects_blank_name_and_unknown_corpus(self) -> None:
        """Verify create validation returns stable safe API errors.

        Args:
            None.

        Returns:
            None. Assertions verify name and corpus validation boundaries.
        """
        blank_response = await self._request(
            "POST",
            "/indexes",
            _index_payload("   "),
        )
        missing_payload = _index_payload()
        missing_payload["corpus_id"] = "missing-corpus"
        missing_response = await self._request(
            "POST",
            "/indexes",
            missing_payload,
        )

        # Whitespace and missing parents fail before any durable job is enqueued.
        self.assertEqual(blank_response.status_code, 422)
        self.assertEqual(blank_response.json()["detail"]["code"], "invalid_index_name")
        self.assertEqual(missing_response.status_code, 404)
        self.assertEqual(missing_response.json()["detail"]["code"], "corpus_not_found")

    async def test_executor_completes_prepared_index_with_artifact_links(self) -> None:
        """Verify claimed preparation reaches ready with exact artifact provenance.

        Args:
            None.

        Returns:
            None. Assertions verify lifecycle transitions and summaries.
        """
        create_response = await self._request("POST", "/indexes", _index_payload())
        prepared_index_id = create_response.json()["id"]
        claimed_work = claim_next_pending_work_item()
        artifacts = PreparedIndexTestArtifacts()

        # The shared queue must route this request to the preparation executor.
        self.assertEqual(
            claimed_work,
            {"kind": "prepared_index", "id": prepared_index_id},
        )
        executor = PreparedIndexExecutor(
            chunk_set_builder=artifacts.build_chunk_set,
            vector_index_builder=artifacts.build_vector_index,
        )
        ready_index = executor.execute(prepared_index_id)

        # The named row references technical artifacts without duplicating their data.
        self.assertEqual(ready_index["status"], "ready")
        self.assertEqual(ready_index["chunking"]["chunk_count"], 1)
        self.assertEqual(
            ready_index["embedding"]["vector_index_id"],
            "vector-index-prepared-1",
        )
        self.assertFalse(ready_index["chunking"]["reused"])
        self.assertFalse(ready_index["embedding"]["reused"])

        poll_response = await self._request(
            "GET",
            f"/indexes/{prepared_index_id}",
        )

        # Polling derives completed stage states from the durable artifact links.
        self.assertEqual(poll_response.status_code, 200)
        self.assertEqual(poll_response.json()["chunking"]["status"], "completed")
        self.assertEqual(poll_response.json()["embedding"]["status"], "completed")

    async def test_shared_queue_claims_oldest_job_across_job_types(self) -> None:
        """Verify legacy runs and preparations share one creation-ordered queue.

        Args:
            None.

        Returns:
            None. Assertions verify global FIFO selection across both tables.
        """
        # Insert an older legacy row directly so this test focuses on queue ordering.
        with connect() as connection:
            connection.execute(
                """
                INSERT INTO pipeline_run (
                    id, corpus_id, question, effective_config_json,
                    status, created_at
                ) VALUES (?, ?, ?, ?, 'pending', ?)
                """,
                (
                    "older-pipeline-run",
                    "corpus-1",
                    "What is older?",
                    "{}",
                    "2026-01-01T00:00:00Z",
                ),
            )

        create_response = await self._request("POST", "/indexes", _index_payload())
        claimed_work = claim_next_pending_work_item()

        # The worker must not prioritize resource type over durable insertion time.
        self.assertEqual(create_response.status_code, 202)
        self.assertEqual(
            claimed_work,
            {"kind": "pipeline_run", "id": "older-pipeline-run"},
        )

    async def test_embedding_failure_keeps_ready_chunk_artifact(self) -> None:
        """Verify a provider failure is safe and preserves completed chunking work.

        Args:
            None.

        Returns:
            None. Assertions verify structured terminal failure and provenance.
        """
        create_response = await self._request("POST", "/indexes", _index_payload())
        prepared_index_id = create_response.json()["id"]
        claimed_index = claim_next_pending_prepared_index()
        artifacts = PreparedIndexTestArtifacts()
        executor = PreparedIndexExecutor(
            chunk_set_builder=artifacts.build_chunk_set,
            vector_index_builder=artifacts.fail_vector_index,
        )

        # The direct preparation claim puts the durable request into chunking.
        self.assertEqual(claimed_index["id"], prepared_index_id)
        with self.assertRaises(PreparedIndexExecutionError):
            executor.execute(prepared_index_id)

        failed_index = get_prepared_index(prepared_index_id)

        # Embedding failure retains upstream work and never exposes raw exception text.
        self.assertEqual(failed_index["status"], "failed")
        self.assertEqual(
            failed_index["error"]["code"],
            "embedding_provider_unavailable",
        )
        self.assertEqual(failed_index["error"]["stage"], "embedding")
        self.assertEqual(
            failed_index["chunking"]["chunk_set_id"],
            "chunk-set-prepared-1",
        )
        self.assertIsNone(failed_index["embedding"]["vector_index_id"])


def test_schema_allows_duplicate_prepared_names_with_distinct_ids() -> None:
    """Verify the relational model treats names as labels rather than identity.

    Args:
        None.

    Returns:
        None. Assertions verify duplicate label semantics directly in SQLite.
    """
    schema_path = Path(__file__).parents[1] / "src" / "backend" / "db" / "schema.sql"
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(schema_path.read_text(encoding="utf-8"))
    connection.execute(
        "INSERT INTO corpus VALUES (?, ?, ?, ?, ?)",
        ("corpus-1", "Docs", None, "created", "updated"),
    )
    rows = [
        (
            prepared_index_id,
            "Shared label",
            "corpus-1",
            '{"chunking":{},"embedding":{}}',
            "created",
        )
        for prepared_index_id in ("prepared-1", "prepared-2")
    ]

    # Insert identical labels with distinct stable identifiers as the API permits.
    connection.executemany(
        """
        INSERT INTO prepared_index (
            id, name, corpus_id, effective_config_json, status, created_at
        ) VALUES (?, ?, ?, ?, 'pending', ?)
        """,
        rows,
    )

    # Both rows remain available for unambiguous ID-based selection.
    count = connection.execute(
        "SELECT COUNT(*) FROM prepared_index WHERE name = 'Shared label'"
    ).fetchone()[0]
    assert count == 2
