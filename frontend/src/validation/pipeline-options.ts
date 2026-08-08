import { z } from "zod";

/** A selectable value returned by the backend-owned pipeline catalog. */
const pipelineOptionSchema = z.object({
  value: z.string().min(1),
  label: z.string().min(1),
  description: z.string().nullable().optional(),
});

/** A numeric configuration default and its accepted bounds. */
const numericSettingSchema = z.object({
  default: z.number(),
  minimum: z.number(),
  maximum: z.number().nullable().optional(),
});

/** A chunking strategy and whether overlap is meaningful for it. */
const chunkingStrategySchema = pipelineOptionSchema.extend({
  supports_overlap: z.boolean(),
});

/** An embedding provider with at least one available model. */
const providerSchema = z.object({
  value: z.string().min(1),
  label: z.string().min(1),
  models: z.array(pipelineOptionSchema).min(1),
});

/** Token limits advertised for one generation model. */
const generationModelCapabilitiesSchema = z.object({
  context_window_tokens: z.number().int().positive(),
  max_output_tokens: z.number().int().positive().nullable(),
});

/** A generation model paired with its advertised token limits. */
const generationModelSchema = pipelineOptionSchema.extend({
  capabilities: generationModelCapabilitiesSchema,
});

/** A generation provider with at least one capability-aware model. */
const generationProviderSchema = z.object({
  value: z.string().min(1),
  label: z.string().min(1),
  models: z.array(generationModelSchema).min(1),
});

/** An evaluation metric and any additional data it requires. */
const evaluationMetricSchema = pipelineOptionSchema.extend({
  requires_reference_answer: z.boolean(),
  selected_by_default: z.boolean(),
});

/**
 * Runtime schema for the complete GET /pipeline/options response.
 *
 * The schema mirrors the FastAPI response model while keeping validation declarative
 * and reusable. Frontend types below are inferred from this single definition.
 */
export const pipelineOptionsSchema = z.object({
  chunking: z.object({
    strategies: z.array(chunkingStrategySchema).min(1),
    chunk_size_tokens: numericSettingSchema,
    chunk_overlap_tokens: numericSettingSchema,
  }),
  embedding: z.object({
    providers: z.array(providerSchema).min(1),
    distance_metrics: z.array(pipelineOptionSchema).min(1),
  }),
  retrieval: z.object({
    top_k: numericSettingSchema,
  }),
  generation: z.object({
    providers: z.array(generationProviderSchema).min(1),
    temperature: numericSettingSchema,
    max_output_tokens: numericSettingSchema,
  }),
  evaluation: z.object({
    retrieval_metrics: z.array(evaluationMetricSchema).min(1),
    answer_metrics: z.array(evaluationMetricSchema).min(1),
  }),
});

/** A selectable value inferred from the runtime option schema. */
export type PipelineOption = z.infer<typeof pipelineOptionSchema>;

/** An evaluation metric with its input and default-selection metadata. */
export type EvaluationMetricOption = z.infer<typeof evaluationMetricSchema>;

/** Numeric defaults and bounds inferred from the runtime setting schema. */
export type NumericSettingOption = z.infer<typeof numericSettingSchema>;

/** The complete validated pipeline catalog inferred from its runtime schema. */
export type PipelineOptions = z.infer<typeof pipelineOptionsSchema>;

/**
 * Validates the pipeline options response before the UI consumes it.
 *
 * @param value - Untrusted response body returned by the backend.
 * @returns A validated catalog with its TypeScript type inferred from Zod.
 * @throws Error when the response does not match the pipeline options API contract.
 */
export function parsePipelineOptions(value: unknown): PipelineOptions {
  const result = pipelineOptionsSchema.safeParse(value);

  // Keep detailed validation internals out of the UI while reporting a contract failure.
  if (!result.success) {
    throw new Error("The backend returned pipeline options that do not match the API contract.");
  }

  return result.data;
}
