import { z } from "zod";

/** Runtime contract for the preparation configuration accepted by FastAPI. */
const preparationConfigurationSchema = z.object({
  chunking: z.object({
    strategy: z.enum(["recursive", "fixed_size", "paragraph"]),
    chunk_size_tokens: z.number().int().positive(),
    chunk_overlap_tokens: z.number().int().nonnegative(),
  }),
  embedding: z.object({
    provider: z.string().min(1),
    model: z.string().min(1),
    distance_metric: z.enum(["cosine", "dot_product", "euclidean"]),
  }),
});

/** Runtime contract for a named prepared-index creation request. */
export const preparedIndexCreateRequestSchema = z.object({
  name: z.string().trim().min(1).max(100),
  corpus_id: z.string().min(1),
  configuration: preparationConfigurationSchema,
});

/** Lifecycle values persisted for the complete preparation request. */
const preparedIndexStatusSchema = z.enum([
  "pending",
  "running",
  "ready",
  "failed",
]);

/** Lifecycle values derived for each preparation stage. */
const preparedIndexStageStatusSchema = z.enum([
  "pending",
  "running",
  "completed",
  "failed",
]);

/** Runtime contract for the reusable chunk artifact summary. */
const preparedIndexChunkingSchema = z.object({
  status: preparedIndexStageStatusSchema,
  chunk_set_id: z.string().min(1).nullable(),
  chunk_count: z.number().int().nonnegative().nullable(),
  reused: z.boolean().nullable(),
  duration_ms: z.number().int().nonnegative().nullable(),
});

/** Runtime contract for the reusable vector artifact summary. */
const preparedIndexEmbeddingSchema = z.object({
  status: preparedIndexStageStatusSchema,
  vector_index_id: z.string().min(1).nullable(),
  vector_count: z.number().int().nonnegative().nullable(),
  dimensions: z.number().int().positive().nullable(),
  provider: z.string().min(1),
  model: z.string().min(1),
  distance_metric: z.enum(["cosine", "dot_product", "euclidean"]),
  reused: z.boolean().nullable(),
  duration_ms: z.number().int().nonnegative().nullable(),
});

/** Runtime contract for safe terminal preparation errors. */
const preparedIndexErrorSchema = z.object({
  code: z.string().min(1),
  message: z.string().min(1),
  stage: z.enum(["chunking", "embedding"]).nullable(),
  details: z.record(z.string(), z.unknown()),
});

/** Runtime contract shared by create, list, detail, and polling responses. */
const preparedIndexSchema = preparedIndexCreateRequestSchema.extend({
  id: z.string().min(1),
  status: preparedIndexStatusSchema,
  current_stage: z.enum(["chunking", "embedding"]).nullable(),
  created_at: z.string().min(1),
  started_at: z.string().min(1).nullable(),
  completed_at: z.string().min(1).nullable(),
  duration_ms: z.number().int().nonnegative().nullable(),
  chunking: preparedIndexChunkingSchema,
  embedding: preparedIndexEmbeddingSchema,
  error: preparedIndexErrorSchema.nullable(),
});

/** Structured request-level error returned before a prepared index is queued. */
const preparedIndexApiErrorSchema = z.object({
  detail: z.object({
    code: z.string().min(1),
    message: z.string().min(1),
    field: z.string().min(1).optional(),
  }),
});

/** Creation payload inferred from the runtime backend contract. */
export type PreparedIndexCreateRequest = z.infer<
  typeof preparedIndexCreateRequestSchema
>;

/** Prepared-index lifecycle record inferred from the runtime backend contract. */
export type PreparedIndex = z.infer<typeof preparedIndexSchema>;

/** Closed prepared-index lifecycle used by list filters and polling. */
export type PreparedIndexStatus = z.infer<typeof preparedIndexStatusSchema>;

/**
 * Validates a prepared index returned by creation or detail polling.
 *
 * @param value - Untrusted response body returned by FastAPI.
 * @returns A validated prepared-index lifecycle record.
 * @throws Error when the response does not match the API contract.
 */
export function parsePreparedIndex(value: unknown): PreparedIndex {
  const result = preparedIndexSchema.safeParse(value);

  // Hide schema internals while reporting an actionable contract mismatch.
  if (!result.success) {
    throw new Error(
      "The backend returned a prepared index that does not match the API contract.",
    );
  }

  return result.data;
}

/**
 * Validates the prepared-index inventory returned by the list endpoint.
 *
 * @param value - Untrusted response body returned by FastAPI.
 * @returns Validated prepared indexes in backend-provided order.
 * @throws Error when any record does not match the API contract.
 */
export function parsePreparedIndexes(value: unknown): PreparedIndex[] {
  const result = z.array(preparedIndexSchema).safeParse(value);

  // Reject the complete response so malformed records never enter selection state.
  if (!result.success) {
    throw new Error(
      "The backend returned prepared indexes that do not match the API contract.",
    );
  }

  return result.data;
}

/**
 * Extracts a safe message from a structured prepared-index API failure.
 *
 * @param value - Untrusted error response body returned by FastAPI.
 * @returns The safe backend message, or null for an unknown error shape.
 */
export function parsePreparedIndexApiError(value: unknown): string | null {
  const result = preparedIndexApiErrorSchema.safeParse(value);

  // Only public contract messages are safe to present to the user.
  if (!result.success) {
    return null;
  }

  return result.data.detail.message;
}
