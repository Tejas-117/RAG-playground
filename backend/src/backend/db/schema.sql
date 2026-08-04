-- Core SQLite schema for corpus identity and uploaded source files. The
-- application must enable foreign-key enforcement:
-- PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS corpus (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document (
    id TEXT PRIMARY KEY,
    corpus_id TEXT NOT NULL REFERENCES corpus(id) ON DELETE RESTRICT,
    original_filename TEXT NOT NULL,
    storage_path TEXT NOT NULL UNIQUE,
    mime_type TEXT,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    content_sha256 TEXT NOT NULL,
    uploaded_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_document_corpus_id
    ON document (corpus_id);
