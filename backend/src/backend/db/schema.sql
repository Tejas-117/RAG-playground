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

-- A reusable vector-search artifact built from one exact chunk set, embedding
-- configuration, provider adapter policy, and Chroma distance space.
CREATE TABLE IF NOT EXISTS vector_index (
    id TEXT PRIMARY KEY,
    chunk_set_id TEXT NOT NULL REFERENCES chunk_set(id) ON DELETE RESTRICT,
    fingerprint TEXT NOT NULL UNIQUE,
    embedding_config_json TEXT NOT NULL CHECK (
        json_valid(embedding_config_json)
        AND json_type(embedding_config_json) = 'object'
    ),
    provider TEXT NOT NULL CHECK (length(trim(provider)) > 0),
    model TEXT NOT NULL CHECK (length(trim(model)) > 0),
    provider_model TEXT,
    provider_revision TEXT,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    distance_metric TEXT NOT NULL CHECK (
        distance_metric IN ('cosine', 'dot_product', 'euclidean')
    ),
    input_policy_version TEXT NOT NULL,
    indexer_name TEXT NOT NULL,
    indexer_version TEXT NOT NULL,
    collection_name TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status = 'ready'),
    vector_count INTEGER NOT NULL CHECK (vector_count > 0),
    created_at TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    duration_ms INTEGER NOT NULL CHECK (duration_ms >= 0)
);

CREATE INDEX IF NOT EXISTS idx_vector_index_chunk_set_id
    ON vector_index (chunk_set_id);

-- One immutable pipeline execution, including its resolved configuration,
-- lifecycle, reusable chunk artifact, timing, and structured failure state.
CREATE TABLE IF NOT EXISTS pipeline_run (
    id TEXT PRIMARY KEY,
    corpus_id TEXT NOT NULL REFERENCES corpus(id) ON DELETE RESTRICT,
    chunk_set_id TEXT REFERENCES chunk_set(id) ON DELETE RESTRICT,
    vector_index_id TEXT REFERENCES vector_index(id) ON DELETE RESTRICT,
    question TEXT NOT NULL CHECK (length(trim(question)) > 0),
    effective_config_json TEXT NOT NULL CHECK (json_valid(effective_config_json)),
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'running', 'completed', 'failed')
    ),
    current_stage TEXT CHECK (
        current_stage IS NULL OR
        current_stage IN ('chunking', 'embedding', 'retrieval', 'generation')
    ),
    chunk_set_reused INTEGER CHECK (chunk_set_reused IN (0, 1)),
    vector_index_reused INTEGER CHECK (vector_index_reused IN (0, 1)),
    chunking_duration_ms INTEGER CHECK (chunking_duration_ms >= 0),
    embedding_duration_ms INTEGER CHECK (embedding_duration_ms >= 0),
    retrieval_duration_ms INTEGER CHECK (retrieval_duration_ms >= 0),
    generation_duration_ms INTEGER CHECK (generation_duration_ms >= 0),
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    duration_ms INTEGER CHECK (duration_ms >= 0),
    error_code TEXT,
    error_details_json TEXT CHECK (
        error_details_json IS NULL OR (
            json_valid(error_details_json)
            AND json_type(error_details_json) = 'object'
        )
    ),
    CHECK (
        status != 'running' OR (
            started_at IS NOT NULL
            AND current_stage IS NOT NULL
        )
    ),
    CHECK (
        status != 'completed' OR (
            chunk_set_id IS NOT NULL
            AND chunk_set_reused IS NOT NULL
            AND vector_index_id IS NOT NULL
            AND vector_index_reused IS NOT NULL
            AND chunking_duration_ms IS NOT NULL
            AND embedding_duration_ms IS NOT NULL
            AND retrieval_duration_ms IS NOT NULL
            AND generation_duration_ms IS NOT NULL
            AND current_stage IS NULL
            AND started_at IS NOT NULL
            AND completed_at IS NOT NULL
            AND duration_ms IS NOT NULL
        )
    ),
    CHECK (
        status != 'failed' OR (
            started_at IS NOT NULL
            AND completed_at IS NOT NULL
            AND duration_ms IS NOT NULL
            AND error_code IS NOT NULL
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_pipeline_run_corpus_id
    ON pipeline_run (corpus_id);

CREATE INDEX IF NOT EXISTS idx_pipeline_run_chunk_set_id
    ON pipeline_run (chunk_set_id);

CREATE INDEX IF NOT EXISTS idx_pipeline_run_vector_index_id
    ON pipeline_run (vector_index_id);

CREATE INDEX IF NOT EXISTS idx_pipeline_run_queue
    ON pipeline_run (status, created_at);

-- One immutable retrieval result for one pipeline run and its exact vector
-- index. Text and source metadata remain normalized in the chunk table.
CREATE TABLE IF NOT EXISTS retrieval_result (
    id TEXT PRIMARY KEY,
    pipeline_run_id TEXT NOT NULL UNIQUE
        REFERENCES pipeline_run(id) ON DELETE CASCADE,
    vector_index_id TEXT NOT NULL
        REFERENCES vector_index(id) ON DELETE RESTRICT,
    requested_top_k INTEGER NOT NULL CHECK (requested_top_k > 0),
    returned_count INTEGER NOT NULL CHECK (
        returned_count >= 0 AND returned_count <= requested_top_k
    ),
    distance_metric TEXT NOT NULL CHECK (
        distance_metric IN ('cosine', 'dot_product', 'euclidean')
    ),
    duration_ms INTEGER NOT NULL CHECK (duration_ms >= 0),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_retrieval_result_vector_index_id
    ON retrieval_result (vector_index_id);

-- Ranked references to immutable chunks returned by vector search. Raw
-- distances are retained without converting distance into similarity.
CREATE TABLE IF NOT EXISTS retrieved_chunk (
    retrieval_result_id TEXT NOT NULL
        REFERENCES retrieval_result(id) ON DELETE CASCADE,
    rank INTEGER NOT NULL CHECK (rank > 0),
    chunk_id TEXT NOT NULL REFERENCES chunk(id) ON DELETE RESTRICT,
    raw_distance REAL NOT NULL,
    PRIMARY KEY (retrieval_result_id, rank),
    UNIQUE (retrieval_result_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_retrieved_chunk_chunk_id
    ON retrieved_chunk (chunk_id);

-- One immutable answer generated from one persisted retrieval result. The full
-- prompt is reconstructable from its version, run question, and context links.
CREATE TABLE IF NOT EXISTS generation_result (
    id TEXT PRIMARY KEY,
    pipeline_run_id TEXT NOT NULL UNIQUE
        REFERENCES pipeline_run(id) ON DELETE CASCADE,
    retrieval_result_id TEXT NOT NULL UNIQUE
        REFERENCES retrieval_result(id) ON DELETE RESTRICT,
    provider TEXT NOT NULL CHECK (length(trim(provider)) > 0),
    model TEXT NOT NULL CHECK (length(trim(model)) > 0),
    provider_model TEXT,
    prompt_template_version TEXT NOT NULL
        CHECK (length(trim(prompt_template_version)) > 0),
    provider_policy_version TEXT NOT NULL
        CHECK (length(trim(provider_policy_version)) > 0),
    generation_config_json TEXT NOT NULL CHECK (
        json_valid(generation_config_json)
        AND json_type(generation_config_json) = 'object'
    ),
    answer_text TEXT NOT NULL CHECK (length(trim(answer_text)) > 0),
    finish_reason TEXT NOT NULL CHECK (length(trim(finish_reason)) > 0),
    prompt_tokens INTEGER CHECK (prompt_tokens >= 0),
    completion_tokens INTEGER CHECK (completion_tokens >= 0),
    total_tokens INTEGER CHECK (total_tokens >= 0),
    provider_request_id TEXT,
    system_fingerprint TEXT,
    provider_called INTEGER NOT NULL CHECK (provider_called IN (0, 1)),
    duration_ms INTEGER NOT NULL CHECK (duration_ms >= 0),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_generation_result_retrieval_result_id
    ON generation_result (retrieval_result_id);

-- Exact retrieval ranks included in the generation prompt. Lower-ranked
-- retrieval hits may be omitted when the context budget is exhausted.
CREATE TABLE IF NOT EXISTS generation_context_chunk (
    generation_result_id TEXT NOT NULL
        REFERENCES generation_result(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal > 0),
    retrieval_result_id TEXT NOT NULL,
    retrieval_rank INTEGER NOT NULL CHECK (retrieval_rank > 0),
    PRIMARY KEY (generation_result_id, ordinal),
    UNIQUE (generation_result_id, retrieval_rank),
    FOREIGN KEY (retrieval_result_id, retrieval_rank)
        REFERENCES retrieved_chunk(retrieval_result_id, rank) ON DELETE RESTRICT
);
