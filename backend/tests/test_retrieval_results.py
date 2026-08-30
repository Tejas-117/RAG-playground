"""Offline tests for immutable retrieval-result persistence."""

import sqlite3
from pathlib import Path

import pytest

from backend.db.connection import connect
from backend.db.repositories.retrieval_results import (
    InvalidRetrievalResultError,
    RetrievalArtifactMismatchError,
    save_retrieval_result,
)
from backend.db.repositories.runs import record_retrieval_result
from backend.retrieval.models import HydratedVectorSearchHit


@pytest.fixture
def isolated_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect retrieval persistence to one initialized temporary database.

    Args:
        tmp_path: Pytest-owned temporary directory for the SQLite file.
        monkeypatch: Helper that restores the production path after the test.

    Returns:
        Path to the isolated initialized database.
    """
    database_path = tmp_path / "retrieval-results.sqlite3"
    monkeypatch.setattr("backend.db.connection.DATABASE_PATH", database_path)

    # Opening one application connection initializes the complete production schema.
    with connect():
        pass
    return database_path


def _insert_artifact_fixtures() -> None:
    """Persist one run, index, and chunks plus one foreign chunk set.

    Args:
        None.

    Returns:
        None. Required retrieval persistence relationships are committed.
    """
    timestamp = "2026-08-29T00:00:00Z"

    # Insert the corpus and two documents used by expected and foreign chunks.
    with connect() as connection:
        connection.execute(
            "INSERT INTO corpus VALUES (?, ?, ?, ?, ?)",
            ("corpus-1", "Corpus", None, timestamp, timestamp),
        )
        connection.executemany(
            """
            INSERT INTO document (
                id, corpus_id, original_filename, storage_path, mime_type,
                size_bytes, content_sha256, uploaded_at
            ) VALUES (?, 'corpus-1', ?, ?, 'text/plain', 10, ?, ?)
            """,
            [
                ("document-1", "one.txt", "/test/one.txt", "sha-one", timestamp),
                ("document-2", "two.txt", "/test/two.txt", "sha-two", timestamp),
            ],
        )
        connection.executemany(
            """
            INSERT INTO chunk_set (
                id, corpus_id, fingerprint, chunking_config_json,
                chunker_name, chunker_version, status, chunk_count, created_at
            ) VALUES (?, 'corpus-1', ?, '{}', 'test', '1', 'ready', ?, ?)
            """,
            [
                ("chunk-set-1", "fingerprint-1", 2, timestamp),
                ("chunk-set-2", "fingerprint-2", 1, timestamp),
            ],
        )
        connection.executemany(
            """
            INSERT INTO chunk (
                id, chunk_set_id, source_document_id, ordinal, text,
                source_metadata_json
            ) VALUES (?, ?, ?, ?, ?, '{}')
            """,
            [
                ("chunk-1", "chunk-set-1", "document-1", 0, "First"),
                ("chunk-2", "chunk-set-1", "document-1", 1, "Second"),
                ("foreign-chunk", "chunk-set-2", "document-2", 0, "Foreign"),
            ],
        )
        connection.execute(
            """
            INSERT INTO vector_index (
                id, chunk_set_id, fingerprint, embedding_config_json,
                provider, model, dimensions, distance_metric,
                input_policy_version, indexer_name, indexer_version,
                collection_name, status, vector_count, created_at,
                started_at, completed_at, duration_ms
            ) VALUES (
                'vector-index-1', 'chunk-set-1', 'vector-fingerprint-1', '{}',
                'ollama', 'nomic-embed-text', 3, 'cosine', 'test-policy',
                'test-store', '1', 'collection-1', 'ready', 2, ?, ?, ?, 1
            )
            """,
            (timestamp, timestamp, timestamp),
        )
        connection.execute(
            """
            INSERT INTO pipeline_run (
                id, corpus_id, chunk_set_id, vector_index_id, question,
                effective_config_json, status, current_stage,
                chunk_set_reused, vector_index_reused,
                chunking_duration_ms, embedding_duration_ms,
                created_at, started_at
            ) VALUES (
                'run-1', 'corpus-1', 'chunk-set-1', 'vector-index-1',
                'Question', ?,
                'running', 'retrieval', 0, 0, 1, 1, ?, ?
            )
            """,
            (
                '{"embedding":{"provider":"ollama","model":"nomic-embed-text","distance_metric":"cosine"}}',
                timestamp,
                timestamp,
            ),
        )


def _hit(rank: int, chunk_id: str, distance: float) -> HydratedVectorSearchHit:
    """Create one complete hydrated hit for persistence tests.

    Args:
        rank: One-based retrieval position assigned by hydration.
        chunk_id: Stable persisted chunk identifier.
        distance: Raw vector-store distance retained by the result.

    Returns:
        Complete immutable hydrated hit accepted by persistence.
    """
    return HydratedVectorSearchHit(
        rank=rank,
        chunk_id=chunk_id,
        raw_distance=distance,
        source_document_id="document-1",
        ordinal=rank - 1,
        text=f"Text {rank}",
        character_start_offset=None,
        character_end_offset=None,
        token_start_offset=None,
        token_end_offset=None,
        page_start=None,
        page_end=None,
        section_path=None,
        source_metadata={},
    )


def _stored_counts() -> tuple[int, int]:
    """Count persisted retrieval parents and ranked child rows.

    Args:
        None.

    Returns:
        Retrieval-result and retrieved-chunk row counts.
    """
    # Read both counts from one connection for a consistent rollback assertion.
    with connect() as connection:
        result_count = connection.execute(
            "SELECT COUNT(*) FROM retrieval_result"
        ).fetchone()[0]
        hit_count = connection.execute(
            "SELECT COUNT(*) FROM retrieved_chunk"
        ).fetchone()[0]
    return result_count, hit_count


def test_save_retrieval_result_persists_ranked_references_atomically(
    isolated_database: Path,
) -> None:
    """Verify one result stores ordered IDs and raw distances without chunk text.

    Args:
        isolated_database: Initialized isolated application database.

    Returns:
        None. Assertions cover summary fields, ranking, and normalized storage.
    """
    _insert_artifact_fixtures()
    hits = (
        _hit(1, "chunk-2", 0.125),
        _hit(2, "chunk-1", 0.25),
    )

    result = save_retrieval_result(
        "run-1",
        "vector-index-1",
        10,
        "cosine",
        7,
        hits,
    )

    assert result["pipeline_run_id"] == "run-1"
    assert result["vector_index_id"] == "vector-index-1"
    assert result["requested_top_k"] == 10
    assert result["returned_count"] == 2
    assert result["distance_metric"] == "cosine"
    assert result["duration_ms"] == 7
    assert result["hits"] == [
        {"rank": 1, "chunk_id": "chunk-2", "raw_distance": 0.125},
        {"rank": 2, "chunk_id": "chunk-1", "raw_distance": 0.25},
    ]

    # Child storage references immutable chunks instead of duplicating their payload.
    with connect() as connection:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(retrieved_chunk)")
        }
    assert columns == {
        "retrieval_result_id",
        "rank",
        "chunk_id",
        "raw_distance",
    }


def test_save_retrieval_result_allows_empty_results(
    isolated_database: Path,
) -> None:
    """Verify a valid no-hit search persists an empty result summary.

    Args:
        isolated_database: Initialized isolated application database.

    Returns:
        None. Assertions distinguish a valid empty result from missing data.
    """
    _insert_artifact_fixtures()

    result = save_retrieval_result(
        "run-1",
        "vector-index-1",
        10,
        "cosine",
        0,
        (),
    )

    assert result["returned_count"] == 0
    assert result["hits"] == []
    assert _stored_counts() == (1, 0)


@pytest.mark.parametrize(
    "hits",
    [
        (_hit(2, "chunk-1", 0.1),),
        (_hit(1, "chunk-1", 0.1), _hit(2, "chunk-1", 0.2)),
        (_hit(1, "chunk-1", float("inf")),),
    ],
)
def test_save_retrieval_result_rejects_malformed_hits_before_writing(
    isolated_database: Path,
    hits: tuple[HydratedVectorSearchHit, ...],
) -> None:
    """Verify invalid rank, duplicate ID, and distance values leave no rows.

    Args:
        isolated_database: Initialized isolated application database.
        hits: Malformed ranked hits selected by the parametrized fixture.

    Returns:
        None. Assertions cover validation and all-or-nothing persistence.
    """
    _insert_artifact_fixtures()

    with pytest.raises(InvalidRetrievalResultError):
        save_retrieval_result(
            "run-1",
            "vector-index-1",
            10,
            "cosine",
            1,
            hits,
        )

    assert _stored_counts() == (0, 0)


@pytest.mark.parametrize("chunk_id", ["missing-chunk", "foreign-chunk"])
def test_save_retrieval_result_rejects_missing_or_foreign_chunks_atomically(
    isolated_database: Path,
    chunk_id: str,
) -> None:
    """Verify every hit must belong to the vector index's exact chunk set.

    Args:
        isolated_database: Initialized isolated application database.
        chunk_id: Missing or foreign chunk selected by parametrization.

    Returns:
        None. Assertions verify validation leaves no partial result parent.
    """
    _insert_artifact_fixtures()

    with pytest.raises(RetrievalArtifactMismatchError, match="chunk set"):
        save_retrieval_result(
            "run-1",
            "vector-index-1",
            10,
            "cosine",
            1,
            (_hit(1, chunk_id, 0.1),),
        )

    assert _stored_counts() == (0, 0)


def test_save_retrieval_result_rejects_metric_mismatch(
    isolated_database: Path,
) -> None:
    """Verify stored raw distances retain the vector index's metric label.

    Args:
        isolated_database: Initialized isolated application database.

    Returns:
        None. Assertions verify a mismatched label cannot be persisted.
    """
    _insert_artifact_fixtures()

    with pytest.raises(RetrievalArtifactMismatchError, match="metric"):
        save_retrieval_result(
            "run-1",
            "vector-index-1",
            10,
            "euclidean",
            1,
            (_hit(1, "chunk-1", 0.1),),
        )

    assert _stored_counts() == (0, 0)


def test_save_retrieval_result_enforces_one_result_per_run(
    isolated_database: Path,
) -> None:
    """Verify retries cannot overwrite an immutable run retrieval result.

    Args:
        isolated_database: Initialized isolated application database.

    Returns:
        None. Assertions verify the original parent and child remain unchanged.
    """
    _insert_artifact_fixtures()
    original_hits = (_hit(1, "chunk-1", 0.1),)
    save_retrieval_result(
        "run-1",
        "vector-index-1",
        10,
        "cosine",
        1,
        original_hits,
    )

    with pytest.raises(sqlite3.IntegrityError):
        save_retrieval_result(
            "run-1",
            "vector-index-1",
            10,
            "cosine",
            2,
            (_hit(1, "chunk-2", 0.2),),
        )

    assert _stored_counts() == (1, 1)


def test_save_retrieval_result_rolls_back_parent_when_child_insert_fails(
    isolated_database: Path,
) -> None:
    """Verify a child write failure rolls back the parent and earlier ranks.

    Args:
        isolated_database: Initialized isolated application database.

    Returns:
        None. Assertions verify the transaction exposes no partial retrieval data.
    """
    _insert_artifact_fixtures()

    # Simulate a storage-level failure on the second ranked child insertion.
    with connect() as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_second_retrieved_chunk
            BEFORE INSERT ON retrieved_chunk
            WHEN NEW.chunk_id = 'chunk-2'
            BEGIN
                SELECT RAISE(ABORT, 'simulated child write failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="simulated child"):
        save_retrieval_result(
            "run-1",
            "vector-index-1",
            10,
            "cosine",
            1,
            (
                _hit(1, "chunk-1", 0.1),
                _hit(2, "chunk-2", 0.2),
            ),
        )

    assert _stored_counts() == (0, 0)


def test_record_empty_retrieval_result_advances_to_generation(
    isolated_database: Path,
) -> None:
    """Verify a valid no-hit result advances the run to generation.

    Args:
        isolated_database: Initialized isolated application database.

    Returns:
        None. Assertions verify empty-result persistence and stage lifecycle.
    """
    _insert_artifact_fixtures()

    retrieval_result_id = record_retrieval_result(
        "run-1",
        "vector-index-1",
        10,
        "cosine",
        (),
        3,
    )

    # No nearest neighbors is valid context and still advances to generation.
    assert retrieval_result_id
    assert _stored_counts() == (1, 0)

    with connect() as connection:
        run_row = connection.execute(
            """
            SELECT status, current_stage, retrieval_duration_ms
            FROM pipeline_run WHERE id = 'run-1'
            """
        ).fetchone()
    assert run_row["status"] == "running"
    assert run_row["current_stage"] == "generation"
    assert run_row["retrieval_duration_ms"] == 3


def test_record_retrieval_rolls_back_result_when_ranked_write_fails(
    isolated_database: Path,
) -> None:
    """Verify retrieval rows and terminal transition share one transaction.

    Args:
        isolated_database: Initialized isolated application database.

    Returns:
        None. Assertions verify the run stays retrievable without partial rows.
    """
    _insert_artifact_fixtures()

    # Force a child persistence error after the result parent would be inserted.
    with connect() as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_pipeline_retrieved_chunk
            BEFORE INSERT ON retrieved_chunk
            BEGIN
                SELECT RAISE(ABORT, 'simulated pipeline result failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="simulated pipeline"):
        record_retrieval_result(
            "run-1",
            "vector-index-1",
            10,
            "cosine",
            (_hit(1, "chunk-1", 0.1),),
            3,
        )

    # The transaction restores both the stage and the empty result tables.
    assert _stored_counts() == (0, 0)
    with connect() as connection:
        run_row = connection.execute(
            """
            SELECT status, current_stage, retrieval_duration_ms
            FROM pipeline_run WHERE id = 'run-1'
            """
        ).fetchone()
    assert tuple(run_row) == ("running", "retrieval", None)
