import type { BenchmarkRunSummary } from "@/validation/benchmark-runs";

/** Map a validated summary and local clock to ledger labels; retain the API source for export. */
export function presentRun(source: BenchmarkRunSummary, now: number) {
  const config = source.configuration;
  const milliseconds = source.status === "running" && source.started_at
    ? Math.max(0, now - Date.parse(source.started_at)) : source.duration_ms;
  return {
    source,
    id: source.id,
    created: source.created_at,
    index: source.prepared_index_name,
    indexId: source.prepared_index_id,
    dataset: source.dataset_name,
    datasetId: source.dataset_id,
    provider: config.generation.provider,
    model: config.generation.model,
    status: source.status === "pending" ? "queued" : source.status,
    completed: source.completed_examples,
    total: source.total_examples,
    topK: config.retrieval.top_k,
    temperature: config.generation.temperature,
    chunkSize: config.chunking.chunk_size_tokens,
    seconds: milliseconds === null ? null : Math.floor(milliseconds / 1000),
    retrievalMs: source.metrics.average_retrieval_duration_ms,
    generationMs: source.metrics.average_generation_duration_ms,
    inputTokens: source.metrics.prompt_tokens,
    outputTokens: source.metrics.completion_tokens,
  };
}

/** Presentation state derived from one immutable API summary. */
export type RunPresentation = ReturnType<typeof presentRun>;
