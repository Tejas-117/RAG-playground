# RAG Pipeline Architecture

This document describes the RAG Playground as a **versioned, artifact-based pipeline with fingerprint-based caching**.

The aim is to let users change an experiment configuration freely while reusing prior work whenever the upstream inputs are identical. This avoids unnecessarily parsing, chunking, and embedding the same documents again.

## 1. Immutable corpora

A corpus is a fixed, named collection of uploaded documents.

- A corpus name is required.
- Documents cannot be added to or removed from an existing corpus.
- To change the document set or document content, the user creates a new corpus.

This makes every experiment reproducible: a corpus always refers to the same source files and content. If a corpus could change over time, an old experiment could silently use different documents and no longer be comparable.

A corpus should have a stable ID and a manifest/content hash derived from its documents.

## 2. The pipeline

When the user selects an experiment configuration and clicks **Run**, the backend executes this logical pipeline:

```text
Immutable corpus
  -> parse documents
  -> chunk parsed text
  -> embed chunks
  -> build or reuse vector index
  -> retrieve for a query
  -> optionally rerank or transform query
  -> generate answer
  -> optionally evaluate
```

Stages do not necessarily run from scratch. The backend checks for a valid artifact built from the same inputs and configuration, then reuses it when possible.

Examples:

- Changing `top_k` from 10 to 20 reuses parsing, chunks, and the vector index; retrieval and generation run again.
- Changing chunk size from 500 to 1,000 reuses parsing, but creates a new chunk set and index before retrieval and generation.
- Changing the embedding model reuses parsing and chunks, but creates a new embedding/index artifact.

## 3. Stage artifacts

Each significant pipeline-stage output is a persisted, versioned artifact that later runs can reuse.

```text
Corpus
  -> Parsed documents
       -> Chunk set
            -> Vector index
                 -> Run
                      -> Retrieval result
                      -> Generation result
                      -> Evaluation result
```

### Upload artifact

The corpus is the immutable root input. Store:

- corpus ID and required name
- document IDs
- file names and storage locations
- content hashes
- document metadata
- corpus manifest hash
- upload timestamps

### Parsing artifact

Parsing produces a normalized representation of an uploaded source document. Store:

- original document ID
- parser configuration and parser version
- extracted text
- structure when available: pages, headings, sections, source offsets, and tables
- parse status, latency, and structured errors
- parser fingerprint

Parsed output should be separate from the raw document record:

- `documents` represents the original uploaded file.
- `parsed_documents` represents extracted text and usable structure.

This separation lets multiple chunking configurations reuse one parse result.

### Chunking artifact: `chunk_set`

A `chunk_set` is the complete collection of chunks produced from a corpus using one chunking configuration.

```text
Corpus: Product documentation v1
Chunking configuration:
  strategy: recursive
  chunk size: 800 tokens
  overlap: 100 tokens

Result:
  chunk_set: cs_123
  chunks: 2,412
```

Each chunk set is tied to:

- immutable corpus version
- exact parsed document versions used
- chunking configuration
- chunker implementation/version
- fingerprint
- build status and timestamps

Each individual chunk stores:

- chunk ID
- chunk set ID
- source document ID
- ordinal position
- text
- character/token offsets where possible
- page/section metadata
- source metadata

Different chunking strategies, sizes, overlap, tokenizers, or chunker versions can create different chunks. Chunks therefore cannot be one permanent list directly attached to a document. The same parsed document can belong to many chunk sets.

### Embedding and vector-index artifact

Chunks are structured application data and belong in SQLite. Embeddings and vector-search indexes belong in the vector store, initially Chroma.

A `vector_index` represents one compatible indexed embedding space. Store in SQLite:

- vector index ID
- source chunk set ID
- embedding provider and model identifier
- embedding dimensions
- distance metric
- embedding configuration
- Chroma collection name/reference
- build status, timings, and errors
- index fingerprint

Store in Chroma:

- vectors
- vector IDs mapped to stable chunk IDs
- metadata required for filtering and source display

An index must never be silently reused if any compatibility-defining input changes:

- chunk set
- embedding model
- embedding dimensions
- distance metric
- vector-store/index configuration

Cosine, dot-product, and Euclidean values have different meanings. Scores must be labelled accurately or normalized before comparison/display.

### Run artifact

A run is an immutable record of what the user asked the system to execute at a specific moment. Store:

- run ID and corpus ID
- immutable snapshot of the effective configuration, including resolved defaults
- links to reused or newly-created parse/chunk/index artifacts
- query
- retrieval parameters such as `top_k`
- reranker and query-transformation configuration
- LLM provider/model, prompt/template version, and generation parameters
- stage timings, timestamps, statuses, and structured errors

A configuration may be edited later; a run snapshot must remain historical evidence of exactly what happened.

### Retrieval, generation, and evaluation artifacts

These are query-specific downstream results and must not overwrite ingestion artifacts.

Retrieval stores the query or transformed query, retrieved chunk IDs, raw and display/normalized scores, rank, filters, parameters, and latency.

Generation stores the run ID, prompt or reconstructable prompt reference, retrieved context IDs, provider/model, prompt version, sampling settings, answer, latency, and errors.

Evaluation stores the run or generated-answer ID, evaluator configuration, metric, evaluator model where applicable, prompt/rubric version, optional rationale, and latency.

Evaluation is separate from generation so an answer can be evaluated again without paying to rerun ingestion, retrieval, or generation. An LLM-judge score is contextual evidence, not objective truth; its provenance must be retained.

## 4. MVP chunking strategies

The MVP should include:

1. **Recursive chunking** as the default: split using a hierarchy such as sections, paragraphs, lines, sentences, and finally character/token boundaries.
2. **Fixed-size chunking with overlap** as a simple, predictable baseline.
3. **Paragraph/section-aware chunking** to preserve authored structure when possible.

Defer semantic, proposition, late, parent-child, graph, and hierarchical chunking. The MVP strategies are easier to explain, test, compare, and operate.

### Fixed-size overlap and sliding windows

Fixed-size chunking creates chunks of a chosen length with a selected overlap. Sliding-window chunking describes repeatedly advancing a window by a fixed stride. In practice, the two terms commonly refer to the same implementation pattern.

For a 500-token window with a 400-token stride:

```text
Chunk 1: tokens 0-499
Chunk 2: tokens 400-899
Chunk 3: tokens 800-1299
```

The overlap is 100 tokens.

### Paragraph/section-based and recursive chunking

Paragraph/section chunking uses document structure as its primary unit:

```text
Section -> paragraphs -> split only when needed
```

Recursive chunking uses a fallback hierarchy:

```text
Section boundary
  -> paragraph boundary
    -> sentence boundary
      -> word/token boundary
```

Recursive chunking is more resilient for irregular or oversized content. Paragraph/section chunking is more focused on retaining authored structure.

### Late chunking

Normal chunking works as follows:

```text
Split text into chunks -> embed each chunk independently
```

Late chunking first encodes a much longer document/context, then pools the resulting token-level representations over each chunk span:

```text
Embed long document/context first -> split token representations into chunk vectors
```

The intended benefit is that a chunk vector reflects surrounding context, such as nearby paragraphs and headings. It requires token-level model representations, span pooling, and compatible embedding models, so it is not an MVP priority.

## 5. Configuration, runs, and fingerprints

There are two separate concepts.

### Editable pipeline configuration

This is the configuration the user edits in the UI, for example:

```text
Chunking: recursive, 800 tokens, 100 overlap
Embedding: model A
Metric: cosine
Retrieval: top_k 10
LLM: model B
```

It may be saved and named, such as “Baseline Recursive”.

### Immutable run

A run records the exact effective configuration used for an execution. Named editable configurations are useful for UX, but they are not necessary for cache correctness; caching uses the effective configuration snapshot and stage fingerprints.

The first backend run slice supports a single ad hoc question. `POST /runs`
validates the selected immutable corpus and backend-supported configuration,
then stores the trimmed question and resolved configuration snapshot in
`pipeline_run`. It does not execute pipeline stages yet. The question is stored
on the run—not on the corpus or chunk set—because it is a query-specific input.

The effective configuration includes ordered `retrieval_metrics` and
`answer_metrics` lists. Both lists may be empty to skip evaluation. A
single-question run may select groundedness and answer relevance together, but
cannot select retrieval metrics or answer correctness because it has no labelled
relevant documents or reference answer.

Evaluation datasets remain separate and are not represented in the current run
schema. When they are implemented, they will use stable evaluation-example
records rather than overloading the single `question` field.

### Fingerprint-based reuse

The backend should calculate a deterministic fingerprint for every reusable stage:

```text
parse fingerprint
  = corpus/document content + parser configuration + parser version

chunk fingerprint
  = parsed artifact(s) + chunking configuration + chunker version

index fingerprint
  = chunk set + embedding configuration + dimensions + metric + index configuration

retrieval fingerprint
  = index + query + retrieval parameters + filters + reranker settings

generation fingerprint
  = retrieval output + prompt version + LLM model + generation settings
```

Fingerprints can be generated by canonicalizing configuration JSON and hashing it with stable upstream IDs/hashes.

For each stage the backend:

```text
1. Calculates the expected fingerprint.
2. Looks for a ready artifact with that fingerprint.
3. Reuses it if found.
4. Otherwise creates it.
5. Stores status, provenance, timing, and errors.
```

For example, changing only `top_k` does not change parse, chunk, or index fingerprints. It creates a new retrieval and generation result while reusing upstream work.

## 6. Suggested MVP schema

Use explicit domain tables rather than a generic catch-all `artifacts` table at first:

```text
corpora
documents
parsed_documents
chunk_sets
chunks
vector_indexes
experiments or pipeline_configs
runs
retrieval_results
retrieved_chunks
generation_results
evaluation_results
```

Core relationships:

```text
corpora
  -> documents
    -> parsed_documents

corpora + parsed_documents + chunking config
  -> chunk_sets
    -> chunks

chunk_sets + embedding/index config
  -> vector_indexes

runs
  -> selected corpus
  -> effective config snapshot
  -> reused/built vector index
  -> retrieval result
  -> generation result
  -> evaluation result
```

Because a corpus is immutable and a chunk set processes all of its documents,
`chunk_set.corpus_id` identifies the complete input set. Each `chunk` retains
its own `source_document_id` for per-chunk provenance.

Artifact-like tables should generally contain:

- stable ID
- fingerprint
- configuration snapshot or configuration reference
- upstream artifact references
- status: `pending`, `running`, `ready`, or `failed`
- timestamps
- timings where meaningful
- structured error code/details when failed

Enforce uniqueness for reusable identities, especially parser output per document/parser fingerprint, chunk sets per corpus snapshot/chunk fingerprint, and indexes per chunk set/index fingerprint.

## 7. Pattern name

This design combines several established patterns:

- **Artifact-based pipeline architecture**: each stage materializes a reusable output.
- **Content-addressable storage / artifact caching**: artifacts are identified by hashes of inputs and configuration.
- **Incremental computation**: only affected stages are recomputed.
- **DAG-based workflow execution**: stages depend on upstream outputs.
- **Data lineage / provenance tracking**: results can be traced to exact inputs, settings, and artifacts.

The concise name for this product architecture is:

> A versioned, artifact-based RAG pipeline with fingerprint-based stage caching.

It is analogous to a build system: source-code changes rebuild only affected files; corpus or configuration changes rebuild only affected RAG artifacts. This provides reproducibility, efficient reuse of expensive work, and clear provenance for experiments and comparisons.
