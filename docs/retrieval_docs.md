# Vector Retrieval

The retrieval stage embeds one run's saved question, searches the exact ready
vector index attached to that run, hydrates the ranked chunk IDs from SQLite,
and persists the immutable result before completing the run.

## Pipeline Flow

```text
embedding completes
  -> attach ready vector_index to pipeline_run
  -> set current_stage to retrieval
  -> embed the normalized question with query purpose
  -> verify query/index provider, model, dimensions, policy, store, and metric
  -> query the exact vector collection with configured top_k
  -> hydrate matching chunks from the index's exact chunk_set
  -> atomically persist retrieval_result + retrieved_chunk rows
  -> record retrieval duration and complete pipeline_run
```

`PipelineExecutor` owns this ordering and timing. `retrieve_chunks` owns search
and hydration coordination, while provider HTTP and Chroma behavior remain
behind `EmbeddingProvider` and `VectorStore` adapters.

## Configuration and Score Semantics

The first retrieval configuration exposes only `top_k`, with a resolved default
of 10. The vector request is bounded by the index's persisted vector count, so a
small index may return fewer than requested. An empty result is valid and still
completes the run.

Every stored score is the unmodified `raw_distance` returned by the configured
vector store. It is persisted together with `cosine`, `dot_product`, or
`euclidean` metric identity. Retrieval does not relabel distance as similarity
or compare values across different metrics.

## Compatibility and Provenance

Before querying, the backend verifies the ready index against the run's
embedding provider, model, distance metric, input-policy version, vector-store
identity/version, vector dimensions, provider model provenance, and collection
identity. Query and document embeddings therefore remain in the same vector
space.

Chroma returns lightweight stable chunk IDs and distances. SQLite hydration is
scoped to the exact `chunk_set_id` used to build the index and restores text,
document ID, offsets, pages, section path, and source metadata in rank order.
Missing, duplicate, or foreign chunk IDs fail retrieval instead of weakening
provenance.

## Persistence and Failure Behavior

One run owns at most one `retrieval_result`. Its `retrieved_chunk` children
store contiguous one-based ranks, stable chunk references, and finite raw
distances. Chunk text is not duplicated in result rows.

Result insertion and the run's terminal transition share one SQLite
transaction. If validation, a child insert, or completion fails, no partial
result remains. Failures after embedding retain the run's ready `chunk_set_id`
and `vector_index_id` and expose a safe `retrieval` stage error.

The runs API currently exposes retrieval as an active or failed
`current_stage`. Returning the persisted retrieval summary and hydrated chunks
through an API response is intentionally deferred to the next phase.
