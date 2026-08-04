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

    # Verify the simplified persistence boundary before later tables are added.
    assert {"corpus", "document"}.issubset(table_names)
    assert not {"corpus_version", "corpus_version_document"}.intersection(table_names)


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
