import { z } from "zod";

/** Runtime contract for the persisted benchmark returned immediately after launch. */
const benchmarkRunLaunchSchema = z.object({
  id: z.string().min(1),
  prepared_index_id: z.string().min(1),
  prepared_index_name: z.string().min(1),
  dataset_id: z.string().min(1),
  dataset_name: z.string().min(1),
  corpus_id: z.string().min(1),
  vector_index_id: z.string().min(1),
  status: z.enum(["pending", "running", "completed", "failed"]),
  total_examples: z.number().int().positive(),
  completed_examples: z.number().int().nonnegative(),
  created_at: z.string().min(1),
});

/** Query-time configuration accepted when launching a saved-dataset benchmark. */
export type BenchmarkRunCreateRequest = {
  prepared_index_id: string;
  dataset_id: string;
  configuration: {
    retrieval: { top_k: number };
    generation: {
      provider: string;
      model: string;
      temperature: number;
      max_output_tokens: number;
    };
    evaluation: {
      retrieval_metrics: string[];
      answer_metrics: string[];
    };
  };
};

/** Validated launch response used by the experiments page. */
export type BenchmarkRunLaunch = z.infer<typeof benchmarkRunLaunchSchema>;

/** Structured error returned when a benchmark cannot be enqueued. */
const benchmarkRunApiErrorSchema = z.object({
  detail: z.object({
    code: z.string().min(1),
    message: z.string().min(1),
    field: z.string().min(1).optional(),
  }),
});

/**
 * Validate the untrusted benchmark launch response from FastAPI.
 *
 * @param value - Response body returned by POST /runs.
 * @returns The validated benchmark identity and initial lifecycle state.
 */
export function parseBenchmarkRunLaunch(value: unknown): BenchmarkRunLaunch {
  const result = benchmarkRunLaunchSchema.safeParse(value);

  // Prevent malformed lifecycle data from entering experiment presentation state.
  if (!result.success) {
    throw new Error("The backend returned an invalid benchmark run.");
  }

  return result.data;
}

/**
 * Extract a safe backend message from a benchmark request failure.
 *
 * @param value - Untrusted Axios response body.
 * @returns The public backend message, or null for an unfamiliar shape.
 */
export function parseBenchmarkRunApiError(value: unknown): string | null {
  const result = benchmarkRunApiErrorSchema.safeParse(value);

  // Unknown transport bodies use a caller-owned generic fallback.
  if (!result.success) {
    return null;
  }

  return result.data.detail.message;
}
