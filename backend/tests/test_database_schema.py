"""Tests for the foundational SQLite corpus schema."""

import sqlite3
from pathlib import Path

# Locate the schema relative to this test file so it can run from any working directory.
SCHEMA_PATH = Path(__file__).parents[1] / "src" / "backend" / "db" / "schema.sql"


def test_schema_creates_versioned_corpus_tables() -> None:
    """Execute the schema and verify that its four initial tables are present.

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

    # Verify the complete initial persistence boundary before later tables are added.
    assert {
        "corpus",
        "document",
        "corpus_version",
        "corpus_version_document",
    }.issubset(table_names)


def test_schema_allows_document_reuse_across_corpus_versions() -> None:
    """Verify that one stored source file can belong to multiple versions.

    Parameters:
        None.
    Returns:
        None. Assertions verify the many-to-many version membership design.
    """
    # Create an isolated in-memory database for relationship validation.
    connection = sqlite3.connect(":memory:")

    # Enable referential checks to validate the foreign-key relationships.
    connection.execute("PRAGMA foreign_keys = ON")

    # Create the database structures under test.
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    # Create the corpus that owns the two immutable document-set snapshots.
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

    # Store one physical source file once, outside the corpus-version records.
    connection.execute(
        "INSERT INTO document VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "document-1",
            "guide.pdf",
            "data/documents/document-1.pdf",
            "application/pdf",
            128,
            "a" * 64,
            "2026-08-02T00:00:00Z",
        ),
    )

    # Create the first and second complete corpus snapshots.
    connection.executemany(
        "INSERT INTO corpus_version VALUES (?, ?, ?, ?, ?)",
        [
            ("version-1", "corpus-1", 1, None, "2026-08-02T00:00:00Z"),
            ("version-2", "corpus-1", 2, "Added references", "2026-08-02T01:00:00Z"),
        ],
    )

    # Include the same stored document in both snapshots without copying its row.
    connection.executemany(
        "INSERT INTO corpus_version_document VALUES (?, ?, ?)",
        [
            ("version-1", "document-1", "2026-08-02T00:00:00Z"),
            ("version-2", "document-1", "2026-08-02T01:00:00Z"),
        ],
    )

    # Confirm that the one document row is reused by two version-membership rows.
    document_count = connection.execute("SELECT COUNT(*) FROM document").fetchone()[0]
    membership_count = connection.execute(
        "SELECT COUNT(*) FROM corpus_version_document WHERE document_id = ?",
        ("document-1",),
    ).fetchone()[0]

    # Assert the design guarantees file reuse while preserving distinct version membership.
    assert document_count == 1
    assert membership_count == 2
