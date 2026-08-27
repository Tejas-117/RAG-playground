import { z } from "zod";

/** The effective pipeline configuration accepted and returned by the runs API. */
const pipelineConfigurationSchema = z.object({
  chunking: z.object({
    strategy: z.string().min(1),
    chunk_size_tokens: z.number().int().positive(),
    chunk_overlap_tokens: z.number().int().nonnegative(),
  }),
  embedding: z.object({
    provider: z.string().min(1),
    model: z.string().min(1),
    distance_metric: z.enum(["cosine", "dot_product", "euclidean"]),
  }),
  retrieval: z.object({
    top_k: z.number().int().positive(),
  }),
  generation: z.object({
    provider: z.string().min(1),
    model: z.string().min(1),
    temperature: z.number().nonnegative(),
    max_output_tokens: z.number().int().positive(),
  }),
  evaluation: z.object({
    retrieval_metrics: z.array(z.string().min(1)),
    answer_metrics: z.array(z.string().min(1)),
  }),
});

/** Runtime contract for the payload sent to POST /runs. */
export const runCreateRequestSchema = z.object({
  corpus_id: z.string().min(1),
  question: z.string().trim().min(1),
  configuration: pipelineConfigurationSchema,
});

/** Lifecycle values returned for individual asynchronous stages. */
const runStageStatusSchema = z.enum([
  "pending",
  "running",
  "completed",
  "failed",
]);

/** Runtime contract for chunking state and its optional ready artifact. */
const runChunkingResponseSchema = z.object({
  status: runStageStatusSchema,
  chunk_set_id: z.string().min(1).nullable(),
  chunk_count: z.number().int().nonnegative().nullable(),
  reused: z.boolean().nullable(),
  duration_ms: z.number().int().nonnegative().nullable(),
});

/** Runtime contract for embedding state and its optional ready vector index. */
const runEmbeddingResponseSchema = z.object({
  status: runStageStatusSchema,
  vector_index_id: z.string().min(1).nullable(),
  vector_count: z.number().int().nonnegative().nullable(),
  dimensions: z.number().int().positive().nullable(),
  provider: z.string().min(1),
  model: z.string().min(1),
  distance_metric: z.enum(["cosine", "dot_product", "euclidean"]),
  reused: z.boolean().nullable(),
  duration_ms: z.number().int().nonnegative().nullable(),
});

/** Runtime contract for a persisted execution failure returned while polling. */
const runFailureSchema = z.object({
  code: z.string().min(1),
  message: z.string().min(1),
  stage: z.enum(["chunking", "embedding"]).nullable(),
  details: z.record(z.string(), z.unknown()),
});

/** Runtime contract shared by POST /runs and GET /runs/{id}. */
const runResponseSchema = runCreateRequestSchema.extend({
  id: z.string().min(1),
  status: z.enum(["pending", "running", "completed", "failed"]),
  current_stage: z.enum(["chunking", "embedding"]).nullable(),
  created_at: z.string().min(1),
  started_at: z.string().min(1).nullable(),
  completed_at: z.string().min(1).nullable(),
  duration_ms: z.number().int().nonnegative().nullable(),
  chunking: runChunkingResponseSchema,
  embedding: runEmbeddingResponseSchema,
  error: runFailureSchema.nullable(),
});

/** Structured error detail returned by request-level FastAPI failures. */
const runApiErrorSchema = z.object({
  detail: z.object({
    code: z.string().min(1),
    message: z.string().min(1),
    field: z.string().min(1).optional(),
  }),
});

/** Safe request-level failure details that the run UI may display. */
export type RunApiFailure = {
  message: string;
};

/** A validated request payload inferred from the public runtime contract. */
export type RunCreateRequest = z.infer<typeof runCreateRequestSchema>;

/** A validated persisted run inferred from the public runtime contract. */
export type RunResponse = z.infer<typeof runResponseSchema>;

/**
 * Validates an untrusted run returned by enqueueing or polling.
 *
 * @param value - Response body received from the backend.
 * @returns The validated persisted run and effective configuration snapshot.
 * @throws Error when the backend response does not match the runs API contract.
 */
export function parseRunResponse(value: unknown): RunResponse {
  const result = runResponseSchema.safeParse(value);

  // Hide schema internals while making a backend contract mismatch actionable.
  if (!result.success) {
    throw new Error("The backend returned a run that does not match the API contract.");
  }

  return result.data;
}

/**
 * Extracts a safe message from an application-level FastAPI error response.
 *
 * @param value - Untrusted error response body received from the backend.
 * @returns The backend message when the structured error contract is valid, otherwise null.
 */
export function parseRunApiError(value: unknown): string | null {
  const result = runApiErrorSchema.safeParse(value);

  // Only display messages that conform to the public error contract.
  if (!result.success) {
    return null;
  }

  return result.data.detail.message;
}

/**
 * Extracts safe request-level details before a run has been persisted.
 *
 * @param value - Untrusted error response body received from the backend.
 * @returns Display-safe failure details when valid, otherwise null.
 */
export function parseRunApiFailure(value: unknown): RunApiFailure | null {
  const message = parseRunApiError(value);

  // Keep malformed server responses out of presentation state.
  if (message === null) {
    return null;
  }

  return { message };
}
