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

-- A reusable artifact containing every chunk generated from one immutable corpus
-- with one exact chunking configuration and chunker implementation.
CREATE TABLE IF NOT EXISTS chunk_set (
    id TEXT PRIMARY KEY,
    corpus_id TEXT NOT NULL REFERENCES corpus(id) ON DELETE RESTRICT,
    fingerprint TEXT NOT NULL UNIQUE,
    chunking_config_json TEXT NOT NULL,
    chunker_name TEXT NOT NULL,
    chunker_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'ready', 'failed')),
    chunk_count INTEGER NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    duration_ms INTEGER CHECK (duration_ms >= 0),
    error_code TEXT,
    error_details_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunk_set_corpus_id
    ON chunk_set (corpus_id);

-- Individual chunk text and source provenance. Embeddings themselves live in
-- the vector store and reference this stable application-level chunk ID.
CREATE TABLE IF NOT EXISTS chunk (
    id TEXT PRIMARY KEY,
    chunk_set_id TEXT NOT NULL REFERENCES chunk_set(id) ON DELETE CASCADE,
    source_document_id TEXT NOT NULL REFERENCES document(id) ON DELETE RESTRICT,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    text TEXT NOT NULL,
    character_start_offset INTEGER CHECK (character_start_offset >= 0),
    character_end_offset INTEGER CHECK (
        character_end_offset >= character_start_offset
    ),
    token_start_offset INTEGER CHECK (token_start_offset >= 0),
    token_end_offset INTEGER CHECK (token_end_offset >= token_start_offset),
    page_start INTEGER CHECK (page_start >= 1),
    page_end INTEGER CHECK (page_end >= page_start),
    section_path_json TEXT,
    source_metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (chunk_set_id, source_document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_chunk_chunk_set_id
    ON chunk (chunk_set_id);

CREATE INDEX IF NOT EXISTS idx_chunk_source_document_id
    ON chunk (source_document_id);

-- An immutable record of one user-submitted question and its fully resolved
-- pipeline configuration. Query-specific stage results will reference this row
-- when pipeline execution is implemented.
CREATE TABLE IF NOT EXISTS pipeline_run (
    id TEXT PRIMARY KEY,
    corpus_id TEXT NOT NULL REFERENCES corpus(id) ON DELETE RESTRICT,
    question TEXT NOT NULL CHECK (length(trim(question)) > 0),
    effective_config_json TEXT NOT NULL CHECK (json_valid(effective_config_json)),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pipeline_run_corpus_id
    ON pipeline_run (corpus_id);
