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
    distance_metric: z.string().min(1),
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

/** Runtime contract for the ready chunk artifact returned by the current pipeline. */
const runChunkingResponseSchema = z.object({
  chunk_set_id: z.string().min(1),
  status: z.literal("ready"),
  chunk_count: z.number().int().nonnegative(),
  reused: z.boolean(),
});

/** Runtime contract for a completed run returned by POST /runs. */
const runResponseSchema = runCreateRequestSchema.extend({
  id: z.string().min(1),
  status: z.literal("completed"),
  created_at: z.string().min(1),
  started_at: z.string().min(1),
  completed_at: z.string().min(1),
  duration_ms: z.number().int().nonnegative(),
  chunking: runChunkingResponseSchema,
});

/** Structured error detail returned by application-level FastAPI failures. */
const runApiErrorSchema = z.object({
  detail: z.object({
    code: z.string().min(1),
    message: z.string().min(1),
    field: z.string().min(1).optional(),
    run_id: z.string().min(1).optional(),
    stage: z.string().min(1).optional(),
  }),
});

/** Safe application-level failure details that the run UI may display. */
export type RunApiFailure = {
  message: string;
  runId?: string;
  stage?: string;
};

/** A validated request payload inferred from the public runtime contract. */
export type RunCreateRequest = z.infer<typeof runCreateRequestSchema>;

/** A validated persisted run inferred from the public runtime contract. */
export type RunResponse = z.infer<typeof runResponseSchema>;

/**
 * Validates an untrusted successful response returned by POST /runs.
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
 * Extracts safe execution details from an application-level run failure.
 *
 * @param value - Untrusted error response body received from the backend.
 * @returns Display-safe failure details when valid, otherwise null.
 */
export function parseRunApiFailure(value: unknown): RunApiFailure | null {
  const result = runApiErrorSchema.safeParse(value);

  // Reject malformed failures instead of exposing unvalidated response values.
  if (!result.success) {
    return null;
  }

  return {
    message: result.data.detail.message,
    runId: result.data.detail.run_id,
    stage: result.data.detail.stage,
  };
}
