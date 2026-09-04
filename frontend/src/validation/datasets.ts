import { z } from "zod";

/** Maximum dataset name length enforced by the backend import contract. */
export const DATASET_NAME_MAX_LENGTH = 100;

/** Maximum dataset upload size enforced by the backend in bytes. */
export const DATASET_FILE_MAX_SIZE_BYTES = 30 * 1024 * 1024;

/** Runtime contract for one successfully resolved relevant document. */
const relevantDocumentSchema = z.object({
  id: z.string().min(1),
  filename: z.string().min(1),
});

/** Runtime contract for one non-fatal document-name resolution warning. */
const datasetImportWarningSchema = z.object({
  example_id: z.string().min(1),
  example_ordinal: z.number().int().nonnegative(),
  document_name: z.string(),
  code: z.string().min(1),
  message: z.string().min(1),
});

/** Runtime contract for one ordered evaluation example. */
const datasetExampleSchema = z.object({
  id: z.string().min(1),
  ordinal: z.number().int().nonnegative(),
  question: z.string().min(1),
  reference_answer: z.string().nullable(),
  relevant_documents: z.array(relevantDocumentSchema),
});

/** Runtime contract shared by dataset inventory and detail responses. */
const datasetSummarySchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1).max(DATASET_NAME_MAX_LENGTH),
  corpus_id: z.string().min(1),
  corpus_name: z.string().min(1),
  source_filename: z.string().min(1),
  source_sha256: z.string().length(64),
  example_count: z.number().int().nonnegative(),
  resolved_document_count: z.number().int().nonnegative(),
  warning_count: z.number().int().nonnegative(),
  created_at: z.string().min(1),
});

/** Runtime contract for a fully hydrated immutable dataset. */
const datasetDetailSchema = datasetSummarySchema.extend({
  warnings: z.array(datasetImportWarningSchema),
  examples: z.array(datasetExampleSchema),
});

/** Runtime contract for safe structured errors returned by dataset routes. */
const datasetApiErrorSchema = z.object({
  detail: z.object({
    code: z.string().min(1),
    message: z.string().min(1),
  }),
});

/** One resolved relevant document inferred from the backend contract. */
export type RelevantDocument = z.infer<typeof relevantDocumentSchema>;

/** One persisted import warning inferred from the backend contract. */
export type DatasetImportWarning = z.infer<typeof datasetImportWarningSchema>;

/** One ordered dataset example inferred from the backend contract. */
export type DatasetExample = z.infer<typeof datasetExampleSchema>;

/** One dataset inventory record inferred from the backend contract. */
export type DatasetSummary = z.infer<typeof datasetSummarySchema>;

/** One complete dataset detail record inferred from the backend contract. */
export type DatasetDetail = z.infer<typeof datasetDetailSchema>;

/** Values required by the multipart dataset import endpoint. */
export type DatasetImportInput = {
  name: string;
  corpusId: string;
  file: File;
};

/** Optional server-side filters supported by the dataset inventory endpoint. */
export type DatasetListFilters = {
  corpusId?: string;
  search?: string;
};

/**
 * Validates the dataset inventory returned by FastAPI.
 *
 * @param value - Untrusted response body returned by the dataset list endpoint.
 * @returns Validated newest-first dataset summaries.
 * @throws Error when any inventory record violates the API contract.
 */
export function parseDatasets(value: unknown): DatasetSummary[] {
  const result = z.array(datasetSummarySchema).safeParse(value);

  // Reject the complete response so malformed records never enter UI state.
  if (!result.success) {
    throw new Error("The backend returned datasets that do not match the API contract.");
  }

  return result.data;
}

/**
 * Validates one fully hydrated dataset returned by FastAPI.
 *
 * @param value - Untrusted response body returned by import or detail endpoints.
 * @returns A validated immutable dataset and all of its examples.
 * @throws Error when the response violates the API contract.
 */
export function parseDatasetDetail(value: unknown): DatasetDetail {
  const result = datasetDetailSchema.safeParse(value);

  // Report a concise boundary error without exposing Zod internals to the user.
  if (!result.success) {
    throw new Error("The backend returned a dataset that does not match the API contract.");
  }

  return result.data;
}

/**
 * Extracts a safe backend message from a structured dataset API failure.
 *
 * @param value - Untrusted error response body returned by FastAPI.
 * @returns The safe backend message, or null when the shape is unknown.
 */
export function parseDatasetApiError(value: unknown): string | null {
  const result = datasetApiErrorSchema.safeParse(value);

  // Unknown failures use a caller-owned fallback instead of leaking response details.
  if (!result.success) {
    return null;
  }

  return result.data.detail.message;
}
