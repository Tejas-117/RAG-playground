import { z } from "zod";

/** Persisted parser provenance and extraction statistics for one document. */
const parseSummarySchema = z.object({
  id: z.string().min(1),
  parser_name: z.string().min(1),
  parser_version: z.string().min(1),
  warnings: z.array(z.string()),
  page_count: z.number().int().nonnegative(),
  block_count: z.number().int().nonnegative(),
  utf8_size_bytes: z.number().int().nonnegative(),
  character_count: z.number().int().nonnegative(),
  duration_ms: z.number().int().nonnegative(),
  created_at: z.string().min(1),
});

/** Persisted source document and its optional canonical parse summary. */
const ingestionDocumentSchema = z.object({
  id: z.string().min(1),
  original_filename: z.string().min(1),
  mime_type: z.string().nullable(),
  size_bytes: z.number().int().nonnegative(),
  uploaded_at: z.string().min(1),
  parse: parseSummarySchema.nullable(),
});

/** Corpus record displayed in the ingestion inventory. */
const ingestionCorpusSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  created_at: z.string().min(1),
  updated_at: z.string().min(1),
  documents: z.array(ingestionDocumentSchema),
});

/** Runtime contract for the detailed GET /corpora/ response. */
const ingestionCorporaResponseSchema = z.object({
  corpora: z.array(ingestionCorpusSchema),
});

/** Runtime contract for POST /uploads after parsing and persistence complete. */
const uploadResponseSchema = z.object({
  message: z.string().min(1),
  filenames: z.array(z.string().min(1)),
  corpus: ingestionCorpusSchema.extend({
    documents: z.array(
      ingestionDocumentSchema.extend({
        parse: parseSummarySchema,
      }),
    ),
  }),
});

/** Detailed corpus payload inferred from the backend response contract. */
export type IngestionCorpusPayload = z.infer<typeof ingestionCorpusSchema>;

/** Completed upload payload inferred from the backend response contract. */
export type UploadResponse = z.infer<typeof uploadResponseSchema>;

/**
 * Validate the detailed corpus inventory returned by the backend.
 *
 * @param value - Untrusted response body returned by GET /corpora/.
 * @returns Validated corpus records including document parse summaries.
 * @throws Error when the response does not match the ingestion API contract.
 */
export function parseIngestionCorpora(value: unknown): IngestionCorpusPayload[] {
  const result = ingestionCorporaResponseSchema.safeParse(value);

  // Keep schema implementation details out of the visible inventory error.
  if (!result.success) {
    throw new Error("The backend returned invalid corpus parsing details.");
  }

  return result.data.corpora;
}

/**
 * Validate a completed upload response before reporting success.
 *
 * @param value - Untrusted response body returned by POST /uploads.
 * @returns Validated upload data containing the persisted parse summaries.
 * @throws Error when the response does not match the upload API contract.
 */
export function parseUploadResponse(value: unknown): UploadResponse {
  const result = uploadResponseSchema.safeParse(value);

  // Do not report success when parsing metadata is missing or malformed.
  if (!result.success) {
    throw new Error("The backend returned an invalid upload result.");
  }

  return result.data;
}
