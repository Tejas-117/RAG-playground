# Vector Retrieval

The retrieval stage embeds each benchmark example's question, searches the exact
ready vector index selected by the benchmark, hydrates ranked chunks from SQLite,
and persists the immutable child result before generation.

## Pipeline Flow

```text
benchmark example starts
  -> use the prepared index's exact ready vector_index
  -> set the child current_stage to retrieval
  -> embed the normalized question with query purpose
  -> verify query/index provider, model, dimensions, policy, store, and metric
  -> query the exact vector collection with configured top_k
  -> hydrate matching chunks from the index's exact chunk_set
  -> atomically persist benchmark retrieval result + ranked chunk rows
  -> record retrieval duration and advance the child to generation
```

`BenchmarkExecutor` owns this ordering and timing. `retrieve_chunks` owns search
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

One benchmark child owns at most one retrieval result. Its ranked children
store contiguous one-based ranks, stable chunk references, and finite raw
distances. Chunk text is not duplicated in result rows.

Result insertion and the child execution's transition to generation share one SQLite
transaction. If validation, a child insert, or completion fails, no partial
result remains. Failures retain the parent benchmark's prepared-index provenance
and expose a safe `retrieval` stage error.

The runs API exposes retrieval lifecycle, result summary, labelled raw-distance
semantics, and hydrated ranked chunks with document, page, offset, and parser
provenance. Generation records which of these ranks were actually included in
its bounded prompt.
