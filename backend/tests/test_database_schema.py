"""Tests for the foundational SQLite corpus schema."""

import sqlite3
from pathlib import Path

# Locate the schema relative to this test file so it can run from any working directory.
SCHEMA_PATH = Path(__file__).parents[1] / "src" / "backend" / "db" / "schema.sql"


def test_schema_creates_corpus_and_document_tables() -> None:
    """Execute the schema and verify that the initial tables are present.

    Parameters:
        None.
    Returns:
        None. Assertions verify successful schema creation.
    """
    # Create an isolated in-memory database for deterministic schema validation.
    connection = sqlite3.connect(":memory:")

    # Enable SQLite foreign keys because application connections must do the same.
    connection.execute("PRAGMA foreign_keys = ON")

    # Execute the shipped schema exactly as a production database initializer would.
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    # Read SQLite's catalog to confirm that the required initial tables exist.
    table_names = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }

    # Verify source, reusable chunking-artifact, and immutable run tables exist.
    assert {
        "corpus",
        "document",
        "document_parse",
        "parsed_page",
        "parsed_block",
        "chunk_set",
        "chunk",
        "retrieval_result",
        "retrieved_chunk",
        "pipeline_run",
    }.issubset(table_names)
    assert not {"corpus_version", "corpus_version_document"}.intersection(table_names)


def test_schema_stores_parse_text_once_with_offset_provenance() -> None:
    """Verify canonical text, page offsets, and block offsets can be persisted.

    Parameters:
        None.
    Returns:
        None. Assertions verify the parsed-document relational model.
    """
    # Create an isolated database with production-equivalent foreign keys.
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    # Insert the immutable source hierarchy required by a parse artifact.
    connection.execute(
        "INSERT INTO corpus VALUES (?, ?, ?, ?, ?)",
        ("corpus-1", "Docs", None, "2026-08-02T00:00:00Z", "2026-08-02T00:00:00Z"),
    )
    connection.execute(
        "INSERT INTO document VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "document-1",
            "corpus-1",
            "guide.pdf",
            "uploads/guide.pdf",
            "application/pdf",
            100,
            "a" * 64,
            "2026-08-02T00:00:00Z",
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
            "First\n\nSecond",
            13,
            13,
            "test",
            "1",
            "{}",
            "[]",
            1,
            2,
            5,
            "2026-08-02T00:00:01Z",
        ),
    )
    connection.execute(
        "INSERT INTO parsed_page VALUES (?, ?, ?, ?, ?, ?)",
        ("page-1", "parse-1", 1, 0, 13, "{}"),
    )
    connection.executemany(
        "INSERT INTO parsed_block VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("block-1", "parse-1", "page-1", 0, 0, 0, 5, None, "{}"),
            ("block-2", "parse-1", "page-1", 1, 1, 7, 13, None, "{}"),
        ],
    )

    # Slice the one stored text value using the persisted block offsets.
    text = connection.execute(
        "SELECT normalized_text FROM document_parse WHERE id = 'parse-1'"
    ).fetchone()[0]
    offsets = connection.execute(
        """
        SELECT character_start_offset, character_end_offset
        FROM parsed_block ORDER BY ordinal
        """
    ).fetchall()

    # Confirm provenance resolves to exact text without duplicated block columns.
    assert [text[start:end] for start, end in offsets] == ["First", "Second"]


def test_schema_assigns_documents_to_corpora() -> None:
    """Verify that documents belong directly to their owning corpora.

    Parameters:
        None.
    Returns:
        None. Assertions verify direct corpus/document ownership.
    """
    # Create an isolated in-memory database for relationship validation.
    connection = sqlite3.connect(":memory:")

    # Enable referential checks to validate the foreign-key relationships.
    connection.execute("PRAGMA foreign_keys = ON")

    # Create the database structures under test.
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    # Create two independent corpora for separate document collections.
    connection.executemany(
        "INSERT INTO corpus VALUES (?, ?, ?, ?, ?)",
        [
            (
                "corpus-1",
                "Product docs",
                None,
                "2026-08-02T00:00:00Z",
                "2026-08-02T00:00:00Z",
            ),
            (
                "corpus-2",
                "Marketing docs",
                None,
                "2026-08-02T01:00:00Z",
                "2026-08-02T01:00:00Z",
            ),
        ],
    )

    # Store one document directly under each corpus.
    connection.executemany(
        "INSERT INTO document VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "document-1",
                "corpus-1",
                "guide.pdf",
                "data/documents/guide.pdf",
                "application/pdf",
                128,
                "a" * 64,
                "2026-08-02T00:00:00Z",
            ),
            (
                "document-2",
                "corpus-2",
                "launch.md",
                "data/documents/launch.md",
                "text/markdown",
                64,
                "b" * 64,
                "2026-08-02T01:00:00Z",
            ),
        ],
    )

    # Confirm that each document is associated with exactly its chosen corpus.
    corpus_documents = connection.execute(
        "SELECT corpus_id, COUNT(*) FROM document GROUP BY corpus_id ORDER BY corpus_id"
    ).fetchall()

    # Assert separate uploads create independent corpus document collections.
    assert corpus_documents == [("corpus-1", 1), ("corpus-2", 1)]


def test_schema_records_chunk_set_provenance_and_chunks() -> None:
    """Verify that chunks retain their chunk-set and source-document provenance.

    Parameters:
        None.
    Returns:
        None. Assertions verify the chunking artifact relationships and constraints.
    """
    # Create an isolated database with the same foreign-key enforcement as the app.
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    # Insert the immutable corpus and source document required by a chunk set.
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
            "guide.pdf",
            "data/documents/guide.pdf",
            "application/pdf",
            128,
            "a" * 64,
            "2026-08-02T00:00:00Z",
        ),
    )

    # Record the exact chunking artifact configuration and build result.
    connection.execute(
        """
        INSERT INTO chunk_set (
            id, corpus_id, fingerprint, chunking_config_json, chunker_name,
            chunker_version, status, chunk_count, created_at, started_at,
            completed_at, duration_ms, error_code, error_details_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "chunk-set-1",
            "corpus-1",
            "chunk-fingerprint-1",
            '{"strategy":"recursive","chunk_size_tokens":800,"chunk_overlap_tokens":100}',
            "recursive",
            "1.0.0",
            "ready",
            1,
            "2026-08-02T00:00:00Z",
            "2026-08-02T00:00:01Z",
            "2026-08-02T00:00:02Z",
            1000,
            None,
            None,
        ),
    )

    # Store the chunk text and offsets that downstream retrieval will reference.
    connection.execute(
        """
        INSERT INTO chunk (
            id, chunk_set_id, source_document_id, ordinal, text,
            character_start_offset, character_end_offset, token_start_offset,
            token_end_offset, page_start, page_end, section_path_json,
            source_metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "chunk-1",
            "chunk-set-1",
            "document-1",
            0,
            "Product overview",
            0,
            16,
            0,
            2,
            1,
            1,
            '["Overview"]',
            '{"parser":"pdf"}',
        ),
    )

    # Read the persisted chunk payload and its source document as an execution would.
    persisted_chunk = connection.execute(
        """
        SELECT text, source_document_id
        FROM chunk
        WHERE id = ?
        """,
        ("chunk-1",),
    ).fetchone()

    # Confirm the chunk retains both text and source-document provenance.
    assert persisted_chunk == ("Product overview", "document-1")

    # Reject a second chunk with the same source-document ordinal in one chunk set.
    try:
        connection.execute(
            """
            INSERT INTO chunk (
                id, chunk_set_id, source_document_id, ordinal, text,
                source_metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("chunk-duplicate", "chunk-set-1", "document-1", 0, "Duplicate", "{}"),
        )
    except sqlite3.IntegrityError:
        # The uniqueness constraint prevents ambiguous chunk ordering.
        pass
    else:
        # Fail explicitly if SQLite accepts an ambiguous chunk ordinal.
        raise AssertionError("chunk ordinals must be unique per source document")


def test_schema_records_single_question_pipeline_runs() -> None:
    """Verify run lifecycle, configuration, and chunk-set provenance constraints.

    Parameters:
        None.
    Returns:
        None. Assertions verify run persistence and integrity constraints.
    """
    # Create an isolated database with production-equivalent foreign-key enforcement.
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    # Insert the immutable corpus required by the run foreign key.
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

    # Insert the ready reusable artifact required by a completed run.
    connection.execute(
        """
        INSERT INTO chunk_set (
            id, corpus_id, fingerprint, chunking_config_json, chunker_name,
            chunker_version, status, chunk_count, created_at, started_at,
            completed_at, duration_ms, error_code, error_details_json
        ) VALUES (?, ?, ?, ?, ?, ?, 'ready', 0, ?, ?, ?, 1, NULL, NULL)
        """,
        (
            "chunk-set-1",
            "corpus-1",
            "fingerprint-1",
            '{"strategy":"recursive"}',
            "recursive",
            "1.0.0",
            "2026-08-02T00:00:00Z",
            "2026-08-02T00:00:00Z",
            "2026-08-02T00:00:01Z",
        ),
    )

    # Insert the ready vector artifact required by a completed embedding run.
    connection.execute(
        """
        INSERT INTO vector_index (
            id, chunk_set_id, fingerprint, embedding_config_json, provider,
            model, provider_model, provider_revision, dimensions,
            distance_metric, input_policy_version, indexer_name,
            indexer_version, collection_name, status, vector_count,
            created_at, started_at, completed_at, duration_ms
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, 'ready', ?, ?, ?, ?, ?
        )
        """,
        (
            "vector-index-1",
            "chunk-set-1",
            "vector-fingerprint-1",
            '{"provider":"ollama","model":"nomic-embed-text"}',
            "ollama",
            "nomic-embed-text",
            "nomic-embed-text",
            3,
            "cosine",
            "nomic-retrieval-prefix-v1",
            "chroma-persistent",
            "1",
            "rag_idx_test",
            1,
            "2026-08-02T00:00:01Z",
            "2026-08-02T00:00:01Z",
            "2026-08-02T00:00:02Z",
            1,
        ),
    )

    # Persist a completed run linked directly to the exact ready chunk artifact.
    connection.execute(
        """
        INSERT INTO pipeline_run (
            id, corpus_id, chunk_set_id, vector_index_id, question,
            effective_config_json, status, current_stage, chunk_set_reused,
            vector_index_reused, chunking_duration_ms, embedding_duration_ms,
            created_at, started_at, completed_at, duration_ms, error_code,
            error_details_json
        ) VALUES (
            ?, ?, ?, ?, ?, ?, 'completed', NULL, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL
        )
        """,
        (
            "run-1",
            "corpus-1",
            "chunk-set-1",
            "vector-index-1",
            "What is the refund policy?",
            '{"retrieval":{"top_k":10}}',
            0,
            0,
            1,
            1,
            "2026-08-02T00:00:01Z",
            "2026-08-02T00:00:01Z",
            "2026-08-02T00:00:02Z",
            1000,
        ),
    )
    persisted_run = connection.execute(
        """
        SELECT corpus_id, chunk_set_id, question, effective_config_json,
               status, chunk_set_reused
        FROM pipeline_run
        """
    ).fetchone()

    # Confirm immutable input and the selected reusable artifact remain distinguishable.
    assert persisted_run == (
        "corpus-1",
        "chunk-set-1",
        "What is the refund policy?",
        '{"retrieval":{"top_k":10}}',
        "completed",
        0,
    )

    # Exercise malformed input snapshots independently of lifecycle constraints.
    invalid_runs = [
        (
            "run-blank",
            "   ",
            "{}",
            "2026-08-02T00:00:02Z",
        ),
        (
            "run-json",
            "Question",
            "invalid",
            "2026-08-02T00:00:01Z",
        ),
    ]

    # Reject blank questions and malformed configuration snapshots at persistence time.
    for invalid_run in invalid_runs:
        try:
            connection.execute(
                """
                INSERT INTO pipeline_run (
                    id, corpus_id, question, effective_config_json,
                    status, created_at
                ) VALUES (?, 'corpus-1', ?, ?, 'pending', ?)
                """,
                invalid_run,
            )
        except sqlite3.IntegrityError:
            # The expected integrity failure proves the corresponding check is active.
            pass
        else:
            # Fail explicitly if SQLite accepts an unusable run record.
            raise AssertionError(
                "pipeline runs require a question and valid JSON config"
            )

    # A completed run cannot omit the artifact and lifecycle values it claims exist.
    try:
        connection.execute(
            """
            INSERT INTO pipeline_run (
                id, corpus_id, question, effective_config_json,
                status, created_at
            ) VALUES (?, ?, ?, ?, 'completed', ?)
            """,
            (
                "run-incomplete",
                "corpus-1",
                "Question",
                "{}",
                "2026-08-02T00:00:04Z",
            ),
        )
    except sqlite3.IntegrityError:
        # Lifecycle checks prevent a completed run without a ready artifact reference.
        pass
    else:
        # Fail explicitly if SQLite accepts contradictory completed-run state.
        raise AssertionError("completed pipeline runs require chunk provenance")
