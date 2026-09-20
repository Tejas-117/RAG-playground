/** Presentation-only fixtures, deliberately separate from the benchmark API contract. */
export type RunPreview = {
  id: string;
  created: string;
  index: string;
  dataset: string;
  provider: string;
  model: string;
  status: "running" | "queued" | "completed" | "failed";
  completed: number;
  total: number;
  topK: number;
  temperature: number;
  chunkSize: number;
  seconds: number | null;
  retrievalMs: number | null;
  generationMs: number | null;
  inputTokens: number | null;
  outputTokens: number | null;
};

// A fixed reference date makes preview date filters deterministic across visits.
export const PREVIEW_DATE = "2026-09-20";

// Fictional runs demonstrate every supported lifecycle state without provider calls.
export const RUN_PREVIEWS: RunPreview[] = [
  {
    id: "8f29a1b0-74fd-4201-9d14-7f416ec6ad21", created: "2026-09-20T14:22:00Z",
    index: "nimbus_forge_index", dataset: "Test Dataset 1", provider: "Groq",
    model: "qwen/qwen3-32b", status: "running", completed: 8, total: 25,
    topK: 5, temperature: 0.2, chunkSize: 800, seconds: 81,
    retrievalMs: 142, generationMs: 8500, inputTokens: 12400, outputTokens: 1800,
  },
  {
    id: "c44b917e-4d17-4817-9c34-bd51c9dc25e4", created: "2026-09-20T14:20:00Z",
    index: "product_docs_800", dataset: "Customer QA Bench", provider: "Ollama",
    model: "llama3.2:3b", status: "queued", completed: 0, total: 25,
    topK: 10, temperature: 0, chunkSize: 800, seconds: null,
    retrievalMs: null, generationMs: null, inputTokens: null, outputTokens: null,
  },
  {
    id: "17e902df-fbaa-4763-a175-56cb22d381ea", created: "2026-09-20T11:05:00Z",
    index: "nimbus_forge_index", dataset: "Test Dataset 1", provider: "Groq",
    model: "openai/gpt-oss-120b", status: "completed", completed: 25, total: 25,
    topK: 5, temperature: 0.1, chunkSize: 800, seconds: 134,
    retrievalMs: 110, generationMs: 4900, inputTokens: 38500, outputTokens: 5200,
  },
  {
    id: "33d7b881-bdf2-43e5-8df4-1f3d1e479e94", created: "2026-09-19T17:40:00Z",
    index: "support_knowledge", dataset: "Support Tickets v3", provider: "Ollama",
    model: "llama3.2:3b", status: "failed", completed: 3, total: 25,
    topK: 10, temperature: 0.3, chunkSize: 512, seconds: 238,
    retrievalMs: 135, generationMs: 38000, inputTokens: 4100, outputTokens: 600,
  },
  {
    id: "920bac61-8e18-482f-9462-a98b66d13204", created: "2026-09-10T09:15:00Z",
    index: "product_docs_800", dataset: "Customer QA Bench", provider: "Ollama",
    model: "llama3.2:1b", status: "completed", completed: 25, total: 25,
    topK: 3, temperature: 0.2, chunkSize: 800, seconds: 421,
    retrievalMs: 125, generationMs: 16000, inputTokens: 21000, outputTokens: 3200,
  },
];
