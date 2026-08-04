"""SQLite connection setup shared by database repositories."""

import sqlite3
from pathlib import Path

DATABASE_PATH = Path(__file__).resolve().parents[3] / "rag_playground.sqlite3"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect() -> sqlite3.Connection:
    """Open an initialized SQLite connection for repository operations.

    Args:
        None.

    Returns:
        A SQLite connection with row-name access and foreign keys enabled.
    """
    # Ensure the parent directory exists before SQLite creates the database file.
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    # Enforce the corpus/document relationship for every application connection.
    connection.execute("PRAGMA foreign_keys = ON")

    # Create the foundational tables when the application starts using persistence.
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return connection
