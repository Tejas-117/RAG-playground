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

-- One immutable canonical parse artifact for each immutable uploaded document.
-- Page and block records reference this text with zero-based, end-exclusive
-- Unicode character offsets instead of duplicating extracted text.
CREATE TABLE IF NOT EXISTS document_parse (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL UNIQUE
        REFERENCES document(id) ON DELETE CASCADE,
    normalized_text TEXT NOT NULL CHECK (length(trim(normalized_text)) > 0),
    utf8_size_bytes INTEGER NOT NULL CHECK (utf8_size_bytes > 0),
    character_count INTEGER NOT NULL CHECK (character_count > 0),
    parser_name TEXT NOT NULL CHECK (length(trim(parser_name)) > 0),
    parser_version TEXT NOT NULL CHECK (length(trim(parser_version)) > 0),
    document_metadata_json TEXT NOT NULL DEFAULT '{}'
        CHECK (
            json_valid(document_metadata_json)
            AND json_type(document_metadata_json) = 'object'
        ),
    warnings_json TEXT NOT NULL DEFAULT '[]' CHECK (
        json_valid(warnings_json) AND json_type(warnings_json) = 'array'
    ),
    page_count INTEGER NOT NULL DEFAULT 0 CHECK (page_count >= 0),
    block_count INTEGER NOT NULL DEFAULT 0 CHECK (block_count >= 0),
    duration_ms INTEGER NOT NULL CHECK (duration_ms >= 0),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_document_parse_document_id
    ON document_parse (document_id);

-- Physical or logical page boundaries within the canonical document text.
CREATE TABLE IF NOT EXISTS parsed_page (
    id TEXT PRIMARY KEY,
    parse_id TEXT NOT NULL REFERENCES document_parse(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL CHECK (page_number >= 1),
    character_start_offset INTEGER NOT NULL
        CHECK (character_start_offset >= 0),
    character_end_offset INTEGER NOT NULL
        CHECK (character_end_offset >= character_start_offset),
    metadata_json TEXT NOT NULL DEFAULT '{}' CHECK (
        json_valid(metadata_json) AND json_type(metadata_json) = 'object'
    ),
    UNIQUE (parse_id, page_number),
    UNIQUE (id, parse_id)
);

CREATE INDEX IF NOT EXISTS idx_parsed_page_parse_id
    ON parsed_page (parse_id);

-- Ordered source-aware blocks within the canonical document text. Blocks may
-- omit a page when a parser provides document text without physical pages.
CREATE TABLE IF NOT EXISTS parsed_block (
    id TEXT PRIMARY KEY,
    parse_id TEXT NOT NULL REFERENCES document_parse(id) ON DELETE CASCADE,
    page_id TEXT,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    source_block_index INTEGER CHECK (source_block_index >= 0),
    character_start_offset INTEGER NOT NULL
        CHECK (character_start_offset >= 0),
    character_end_offset INTEGER NOT NULL
        CHECK (character_end_offset >= character_start_offset),
    bounding_box_json TEXT CHECK (
        bounding_box_json IS NULL OR json_valid(bounding_box_json)
    ),
    metadata_json TEXT NOT NULL DEFAULT '{}' CHECK (
        json_valid(metadata_json) AND json_type(metadata_json) = 'object'
    ),
    UNIQUE (parse_id, ordinal),
    FOREIGN KEY (page_id, parse_id)
        REFERENCES parsed_page(id, parse_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_parsed_block_parse_id
    ON parsed_block (parse_id);

CREATE INDEX IF NOT EXISTS idx_parsed_block_page_id
    ON parsed_block (page_id);

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
