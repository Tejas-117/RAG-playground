# Embedding and Vector Indexing

The embedding stage converts every persisted chunk into a dense vector and
stores the resulting searchable index in local Chroma. It runs after chunking
inside the persisted background-run workflow.

## Architecture

```text
POST /runs
  -> persist pending pipeline_run and return 202
  -> local worker claims the oldest pending run
  -> build or reuse chunk_set
  -> call the configured embedding provider over HTTP
  -> validate and store explicit vectors in Chroma
  -> persist ready vector_index metadata in SQLite
  -> attach vector_index and advance pipeline_run to retrieval
  -> retrieve ranked chunks before completing pipeline_run

GET /runs/{run_id}
  -> return current stage, artifact summaries, timings, or safe failure
```

`PipelineExecutor` coordinates ordering but does not contain Ollama or Chroma
logic. `EmbeddingProvider` defines the provider-neutral model interface, and
`VectorStore` defines the provider-neutral index interface. Provider SDK or
HTTP details remain behind adapters.

The first provider adapter is `OllamaHttpEmbeddingProvider`. It calls
`POST /api/embed` over HTTP and does not invoke the Ollama CLI, inspect local
models, pull a model, or start the Ollama service. The backend therefore does
not need to know where the model is hosted or installed. A missing model,
unreachable endpoint, timeout, rejected request, or malformed vector response
fails that run with a structured error while the worker remains available.

This boundary is not limited to Ollama. A future provider implements the same
`EmbeddingProvider` contract, registers its configuration identifier, and
maps its transport errors into the shared failure types. Pipeline execution,
vector indexing, and the frontend polling contract do not change.

## Configuration

The immutable run snapshot contains:

- embedding provider;
- provider model identifier; and
- distance metric: `cosine`, `dot_product`, or `euclidean`.

Ollama uses this base URL by default:

```text
http://localhost:11434
```

Override it for a remote or differently exposed Ollama service:

```bash
OLLAMA_BASE_URL=http://embedding-host:11434 uv run backend
```

The backend sends batches of 32 chunks and sets `truncate: false`. A chunk that
the selected model cannot accept is reported as `embedding_input_too_large`
instead of silently embedding truncated content. Batch size and the 120-second
per-request timeout are backend guardrails, not experiment parameters.

For `nomic-embed-text`, document inputs receive the `search_document:` prefix
and retrieval questions receive `search_query:`. The versioned input policy is
part of index compatibility, while persisted chunk and question text remain
unchanged.

## Validation and Atomic Visibility

Every provider response is treated as untrusted. Before indexing, the adapter
requires:

- exactly one vector for every submitted text;
- non-empty vectors with one consistent positive dimension;
- numeric, finite coordinates; and
- valid optional provider model provenance.

Each build owns a uniquely named Chroma collection. Chunks are stored using
their stable SQLite `chunk.id` values and scalar source metadata. The service
verifies that Chroma's vector count equals the chunk count before inserting the
ready `vector_index` row in SQLite. A failed build deletes its private
collection, so no partial index becomes reusable or visible to a completed run.

## Reuse Compatibility

A ready vector index is reusable only when its fingerprint matches all
compatibility-defining inputs:

- exact chunk-set fingerprint;
- resolved embedding configuration;
- provider adapter identifier and version;
- provider-reported model, revision, and vector dimensions;
- versioned input-prefix policy;
- vector-store adapter identifier and version; and
- distance metric.

Changing the chunking configuration, embedding model, dimensions, input policy,
distance metric, or index implementation therefore creates a different index.
The current service makes one live first-batch provider request before reuse so
the provider's current vector dimensions are confirmed rather than assumed.

## Persistence

SQLite stores the `vector_index` identity, compatibility fingerprint, provider
and model provenance, dimensions, metric, Chroma collection name, vector count,
timestamps, and build duration. Vector coordinates remain in Chroma.

By default Chroma persists under `backend/chroma_data/`, which is ignored by
Git. Override the location with:

```bash
CHROMA_DATA_PATH=/absolute/runtime/path uv run backend
```

`GET /testing/reset` deletes only Chroma collections whose names begin with
`rag_idx_`, then clears relational data and uploaded development files. It does
not delete unrelated collections in the same Chroma database.

## Run Failures

Embedding failures are persisted on the immutable run. Common error codes
include:

- `embedding_provider_unavailable`;
- `embedding_request_timeout`;
- `embedding_authentication_failed`;
- `embedding_rate_limited`;
- `embedding_request_rejected`;
- `embedding_input_too_large`;
- `invalid_embedding_response`; and
- `vector_store_unavailable`.

If embedding fails after chunking succeeds, the run retains its ready
`chunk_set_id`. It has no `vector_index_id`, so a partial or incompatible index
cannot be mistaken for a completed artifact.

After embedding succeeds, the executor attaches the ready vector index and
advances the run to retrieval. A later retrieval failure therefore retains both
ready upstream artifact IDs for provenance.
