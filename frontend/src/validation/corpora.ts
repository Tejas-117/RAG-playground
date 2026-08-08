import { z } from "zod";

/** A persisted corpus that can be selected as an experiment source. */
const corpusOptionSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
});

/** Runtime schema for the GET /corpora/ response used by the experiments page. */
export const corporaResponseSchema = z.object({
  corpora: z.array(corpusOptionSchema),
});

/** A selectable corpus inferred from the runtime corpus schema. */
export type CorpusOption = z.infer<typeof corpusOptionSchema>;

/**
 * Validates and extracts selectable corpora from the backend response.
 *
 * @param value - Untrusted response body returned by the backend.
 * @returns Validated corpus identifiers and display names.
 * @throws Error when the response does not match the corpora API contract.
 */
export function parseCorpora(value: unknown): CorpusOption[] {
  const result = corporaResponseSchema.safeParse(value);

  // Report a concise API contract failure instead of exposing Zod internals in the UI.
  if (!result.success) {
    throw new Error("The backend returned corpora that do not match the API contract.");
  }

  return result.data.corpora;
}
