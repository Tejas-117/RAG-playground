import { z } from "zod";
import { pipelineConfigurationSchema } from "@/validation/runs";

/** Nonnegative measurements preserve zero while nullable fields preserve missing data. */
const countSchema = z.number().int().nonnegative();

/** Validate timestamps before they are used for date filters and elapsed time. */
const timestampSchema = z.string().datetime({ offset: true });

/** Retrieval metrics currently calculated by the deterministic backend evaluator. */
export const retrievalMetricSchema = z.enum([
  "hit_rate_at_k",
  "recall_at_k",
  "mrr",
]);

/** Answer metrics are normalized judgments produced by the configured LLM evaluator. */
export const answerMetricSchema = z.enum([
  "groundedness",
  "answer_relevance",
  "answer_correctness",
]);

/** Every supported metric may be selected in a combined evaluation attempt. */
export const evaluationMetricSchema = z.union([
  retrievalMetricSchema,
  answerMetricSchema,
]);

/** Lifecycle states shared by benchmarks and their independent evaluation attempts. */
const lifecycleStatusSchema = z.enum(["pending", "running", "completed", "failed"]);

/** Runtime contract for one combined evaluation configuration snapshot. */
const retrievalEvaluationConfigurationSchema = z.object({
  retrieval_metrics: z.array(retrievalMetricSchema),
  answer_metrics: z.array(answerMetricSchema),
  evaluator: z.object({
    provider: z.string().min(1),
    model: z.string().min(1),
    temperature: z.number(),
    reasoning_effort: z.string().min(1),
    structured_output: z.string().min(1),
    prompt_version: z.string().min(1),
    rubric_version: z.string().min(1),
  }).nullable(),
});

/** Scores stay within the normalized zero-to-one range or remain unavailable. */
const evaluationScoreSchema = z.number().min(0).max(1).nullable();

/** Selected metrics form a partial record because each attempt can choose a subset. */
const evaluationAggregatesSchema = z.partialRecord(
  evaluationMetricSchema,
  evaluationScoreSchema,
);

/** Coverage distinguishes ineligible inputs from judge failures and valid scores. */
const evaluationCoverageSchema = z.partialRecord(
  evaluationMetricSchema,
  z.object({
    total: countSchema,
    eligible: countSchema,
    scored: countSchema,
    skipped: countSchema,
    error: countSchema,
  }),
);

/** Runtime contract for a durable evaluation attempt without question-level evidence. */
export const retrievalEvaluationSummarySchema = z.object({
  id: z.string().min(1),
  benchmark_run_id: z.string().min(1),
  status: lifecycleStatusSchema,
  configuration: retrievalEvaluationConfigurationSchema,
  aggregates: evaluationAggregatesSchema,
  coverage: evaluationCoverageSchema,
  has_errors: z.boolean(),
  eligible_count: countSchema.nullable(),
  total_count: countSchema.nullable(),
  error: z.object({
    code: z.string().min(1),
    message: z.string().min(1),
  }).nullable(),
  created_at: timestampSchema,
  started_at: timestampSchema.nullable(),
  completed_at: timestampSchema.nullable(),
});

/** Runtime contract for one question's scores and document-ranking evidence. */
const retrievalEvaluationQuestionSchema = z.object({
  example_id: z.string().min(1),
  ordinal: countSchema,
  skip_reason: z.literal("no_resolved_document_labels").nullable(),
  scores: z.partialRecord(
    retrievalMetricSchema,
    z.number().min(0).max(1),
  ).nullable(),
  matching_ranks: z.array(z.number().int().positive()),
  relevant_document_ids: z.array(z.string().min(1)),
  ranked_document_ids: z.array(z.string().min(1)),
  answer_scores: z.partialRecord(
    answerMetricSchema,
    z.object({
      rubric_score: z.number().int().min(0).max(4),
      score: z.number().min(0).max(1),
      rationale: z.string().min(1),
      evidence_ranks: z.array(z.number().int().positive()),
    }),
  ),
  answer_skips: z.partialRecord(
    answerMetricSchema,
    z.enum(["no_generated_answer", "no_reference_answer"]),
  ),
  answer_error: z.object({
    code: z.string().min(1),
    message: z.string().min(1),
    metrics: z.array(answerMetricSchema),
  }).nullable(),
  judge: z.object({
    duration_ms: countSchema.nullable(),
    prompt_tokens: countSchema.nullable(),
    completion_tokens: countSchema.nullable(),
    total_tokens: countSchema.nullable(),
    provider_request_id: z.string().nullable(),
    provider_model: z.string().nullable(),
  }).nullable(),
});

/** Runtime contract for one attempt including its ordered per-question evidence. */
export const retrievalEvaluationDetailSchema = retrievalEvaluationSummarySchema.extend({
  questions: z.array(retrievalEvaluationQuestionSchema),
});

/** Runtime contract for the persisted benchmark returned immediately after launch. */
const benchmarkRunLaunchSchema = z.object({
  id: z.string().min(1),
  prepared_index_id: z.string().min(1),
  prepared_index_name: z.string().min(1),
  dataset_id: z.string().min(1),
  dataset_name: z.string().min(1),
  corpus_id: z.string().min(1),
  vector_index_id: z.string().min(1),
  status: lifecycleStatusSchema,
  total_examples: z.number().int().positive(),
  completed_examples: z.number().int().nonnegative(),
  created_at: z.string().min(1),
});

/** Compact history contract; no question text or retrieved chunks are downloaded. */
const benchmarkRunSummarySchema = benchmarkRunLaunchSchema.extend({
  created_at: timestampSchema,
  started_at: timestampSchema.nullable(),
  completed_at: timestampSchema.nullable(),
  duration_ms: countSchema.nullable(),
  current_stage: z.enum(["retrieval", "generation"]).nullable(),
  current_example_id: z.string().nullable(),
  failed_examples: countSchema,
  pending_examples: countSchema,
  running_examples: countSchema,
  configuration: pipelineConfigurationSchema,
  error: z.object({
    code: z.string(),
    message: z.string(),
    stage: z.enum(["chunking", "embedding", "retrieval", "generation"]).nullable(),
    details: z.record(z.string(), z.unknown()),
  }).nullable(),
  metrics: z.object({
    retrieval_result_count: countSchema,
    generation_result_count: countSchema,
    average_retrieval_duration_ms: z.number().nonnegative().nullable(),
    average_generation_duration_ms: z.number().nonnegative().nullable(),
    prompt_tokens: countSchema.nullable(),
    completion_tokens: countSchema.nullable(),
    prompt_token_result_count: countSchema,
    completion_token_result_count: countSchema,
  }),
  latest_evaluation: retrievalEvaluationSummarySchema.nullable(),
});

/** Persisted ranked evidence, with raw score semantics supplied by its retrieval result. */
const retrievedChunkSchema = z.object({
  rank: countSchema.positive(),
  chunk_id: z.string().min(1),
  raw_distance: z.number().finite(),
  source_document_id: z.string().min(1),
  original_filename: z.string().min(1),
  ordinal: countSchema,
  text: z.string(),
  character_start_offset: countSchema.nullable(),
  character_end_offset: countSchema.nullable(),
  token_start_offset: countSchema.nullable(),
  token_end_offset: countSchema.nullable(),
  page_start: countSchema.positive().nullable(),
  page_end: countSchema.positive().nullable(),
  section_path: z.array(z.string()).nullable(),
  source_metadata: z.record(z.string(), z.unknown()),
});

/** Full detail contract extends the compact run with ordered example outcomes. */
const benchmarkRunDetailSchema = benchmarkRunSummarySchema.extend({
  examples: z.array(z.object({
    id: z.string().min(1),
    example_id: z.string().min(1),
    ordinal: countSchema,
    question: z.string().min(1),
    reference_answer: z.string().nullable(),
    status: z.enum(["pending", "running", "completed", "failed"]),
    current_stage: z.enum(["retrieval", "generation"]).nullable(),
    started_at: timestampSchema.nullable(),
    completed_at: timestampSchema.nullable(),
    duration_ms: countSchema.nullable(),
    retrieval: z.object({
      status: z.enum(["pending", "running", "completed", "failed"]),
      result_id: z.string().nullable(),
      requested_top_k: countSchema.positive(),
      returned_count: countSchema.nullable(),
      distance_metric: z.enum(["cosine", "dot_product", "euclidean"]),
      duration_ms: countSchema.nullable(),
      chunks: z.array(retrievedChunkSchema),
    }).nullable(),
    generation: z.object({
      status: z.enum(["pending", "running", "completed", "failed"]),
      result_id: z.string().nullable(),
      retrieval_result_id: z.string().nullable(),
      provider: z.string().min(1),
      model: z.string().min(1),
      provider_model: z.string().nullable(),
      answer: z.string().nullable(),
      finish_reason: z.string().nullable(),
      prompt_template_version: z.string().nullable(),
      provider_policy_version: z.string().nullable(),
      prompt_tokens: countSchema.nullable(),
      completion_tokens: countSchema.nullable(),
      total_tokens: countSchema.nullable(),
      provider_called: z.boolean().nullable(),
      context_chunks: z.array(z.object({
        ordinal: countSchema.positive(),
        retrieval_rank: countSchema.positive(),
        chunk_id: z.string().min(1),
      })),
      duration_ms: countSchema.nullable(),
    }).nullable(),
    error: benchmarkRunSummarySchema.shape.error,
  })),
});

/** Validated full run including its question-level evidence and answers. */
export type BenchmarkRunDetail = z.infer<typeof benchmarkRunDetailSchema>;

/** Validate an unknown GET /runs/{id} body before rendering it. */
export function parseBenchmarkRunDetail(value: unknown): BenchmarkRunDetail {
  const result = benchmarkRunDetailSchema.safeParse(value);
  // Malformed results must not be mistaken for incomplete execution.
  if (!result.success) throw new Error("The backend returned invalid run details.");
  return result.data;
}

/** Validated summary used by history, filters, and export. */
export type BenchmarkRunSummary = z.infer<typeof benchmarkRunSummarySchema>;

/** Validated compact evaluation attempt used by history and run summaries. */
export type RetrievalEvaluationSummary = z.infer<typeof retrievalEvaluationSummarySchema>;

/** Validated evaluation attempt containing ordered question-level evidence. */
export type RetrievalEvaluationDetail = z.infer<typeof retrievalEvaluationDetailSchema>;

/** Stable metric identifiers accepted by the retrieval evaluation endpoint. */
export type RetrievalMetric = z.infer<typeof retrievalMetricSchema>;

/** Stable answer metric identifiers accepted by the evaluation endpoint. */
export type AnswerMetric = z.infer<typeof answerMetricSchema>;

/** Stable identifiers shared by aggregate, coverage, and question results. */
export type EvaluationMetric = z.infer<typeof evaluationMetricSchema>;

/** Validate an unknown GET /runs body; returns summaries or throws a safe error. */
export function parseBenchmarkRuns(value: unknown): BenchmarkRunSummary[] {
  const result = z.array(benchmarkRunSummarySchema).safeParse(value);
  // Reject the entire snapshot instead of displaying partially validated history.
  if (!result.success) throw new Error("The backend returned invalid run history.");
  return result.data;
}

/**
 * Validate evaluation history returned by the backend.
 *
 * @param value - Untrusted response body from the evaluation history endpoint.
 * @returns Evaluation attempts ordered as supplied by the backend.
 */
export function parseRetrievalEvaluations(value: unknown): RetrievalEvaluationSummary[] {
  const result = z.array(retrievalEvaluationSummarySchema).safeParse(value);

  // Reject an incomplete history snapshot rather than mixing valid and invalid attempts.
  if (!result.success) {
    throw new Error("The backend returned invalid evaluation history.");
  }

  return result.data;
}

/**
 * Validate one evaluation attempt and its question-level evidence.
 *
 * @param value - Untrusted response body from an evaluation detail or create request.
 * @returns The validated durable evaluation attempt.
 */
export function parseRetrievalEvaluationDetail(value: unknown): RetrievalEvaluationDetail {
  const result = retrievalEvaluationDetailSchema.safeParse(value);

  // Malformed evidence must never be presented as a valid score explanation.
  if (!result.success) {
    throw new Error("The backend returned invalid evaluation details.");
  }

  return result.data;
}

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
