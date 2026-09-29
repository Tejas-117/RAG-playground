/** Static, explicitly fictional data for designing the run inspection UI. */
export type PreviewChunk = {
  id: string;
  rank: number;
  filename: string;
  documentId: string;
  distance: number;
  text: string;
  page: number | null;
  characterStart: number;
  characterEnd: number;
  contextOrder: number | null;
};

/** One dataset example with optional saved retrieval and generation output. */
export type PreviewQuestion = {
  id: string;
  ordinal: number;
  question: string;
  reference: string | null;
  status: "completed" | "running" | "pending" | "failed";
  stage: "retrieval" | "generation" | null;
  durationMs: number | null;
  retrievalMs: number | null;
  answer: string | null;
  generationMs: number | null;
  promptTokens: number | null;
  completionTokens: number | null;
  chunks: PreviewChunk[];
  error: string | null;
};

// Five questions cover completed, active, empty, and waiting presentation states.
export const PREVIEW_QUESTIONS: PreviewQuestion[] = [
  {
    id: "example-company-profile",
    ordinal: 0,
    question: "What does NimbusForge Systems build?",
    reference: "NimbusForge builds workflow automation and operational analytics software.",
    status: "completed",
    stage: null,
    durationMs: 982,
    retrievalMs: 112,
    answer: "NimbusForge Systems builds workflow automation and operational analytics " +
      "software. Its primary product, OrbitOps, helps teams design approval flows and " +
      "automate operational workflows.",
    generationMs: 870,
    promptTokens: 412,
    completionTokens: 59,
    chunks: [
      {
        id: "chunk-company-001",
        rank: 1,
        filename: "01_company_profile.docx",
        documentId: "document-company-profile",
        distance: 0.142,
        text: "NimbusForge Systems Pvt. Ltd. is a fictional B2B software company. " +
          "It builds workflow automation and operational analytics software for " +
          "mid-market and enterprise customers. Its primary product is OrbitOps.",
        page: 1,
        characterStart: 148,
        characterEnd: 377,
        contextOrder: 1,
      },
      {
        id: "chunk-product-001",
        rank: 2,
        filename: "02_orbitops_product_brief_current.docx",
        documentId: "document-product-brief",
        distance: 0.188,
        text: "OrbitOps lets operations teams design approval flows, trigger automations " +
          "from events, call external APIs, and monitor workflow performance.",
        page: 1,
        characterStart: 251,
        characterEnd: 399,
        contextOrder: 2,
      },
      {
        id: "chunk-faq-001",
        rank: 3,
        filename: "12_sales_solutions_faq.docx",
        documentId: "document-sales-faq",
        distance: 0.296,
        text: "The Sales and Solutions FAQ answers recurring questions about " +
          "NimbusForge and OrbitOps for customer-facing teams.",
        page: 1,
        characterStart: 110,
        characterEnd: 224,
        contextOrder: null,
      },
    ],
    error: null,
  },
  {
    id: "example-sla",
    ordinal: 1,
    question: "What is the current Enterprise uptime commitment?",
    reference: "The current Enterprise SLA targets 99.95% monthly uptime.",
    status: "completed",
    stage: null,
    durationMs: 1210,
    retrievalMs: 126,
    answer: "The current Enterprise SLA targets 99.95% monthly uptime for the " +
      "OrbitOps Enterprise production service, subject to its exclusions.",
    generationMs: 1084,
    promptTokens: 529,
    completionTokens: 34,
    chunks: [
      {
        id: "chunk-sla-current-001",
        rank: 1,
        filename: "05_enterprise_sla_current.docx",
        documentId: "document-sla-current",
        distance: 0.121,
        text: "NimbusForge targets 99.95% monthly uptime for the OrbitOps Enterprise " +
          "production service, excluding scheduled maintenance and defined exclusions.",
        page: 1,
        characterStart: 220,
        characterEnd: 364,
        contextOrder: 1,
      },
      {
        id: "chunk-sla-archived-001",
        rank: 2,
        filename: "06_enterprise_sla_archived.docx",
        documentId: "document-sla-archived",
        distance: 0.192,
        text: "Archived 2024 terms targeted 99.9% monthly uptime. For new contracts, " +
          "use SLA version 2.2 unless Legal approves an exception.",
        page: 1,
        characterStart: 174,
        characterEnd: 299,
        contextOrder: null,
      },
    ],
    error: null,
  },
  {
    id: "example-pricing",
    ordinal: 2,
    question: "Which source has the approved Business edition list price?",
    reference: "The approved Commercial Pricing Guide contains the current list price.",
    status: "running",
    stage: "generation",
    durationMs: null,
    retrievalMs: 141,
    answer: null,
    generationMs: null,
    promptTokens: null,
    completionTokens: null,
    chunks: [
      {
        id: "chunk-pricing-001",
        rank: 1,
        filename: "04_pricing_guide.docx",
        documentId: "document-pricing-guide",
        distance: 0.154,
        text: "Business edition: INR 1,35,000 base fee per month. The approved guide " +
          "reflects list prices effective August 1, 2026.",
        page: 1,
        characterStart: 302,
        characterEnd: 430,
        contextOrder: null,
      },
    ],
    error: null,
  },
  {
    id: "example-unknown",
    ordinal: 3,
    question: "What is the CEO’s home address?",
    reference: null,
    status: "pending",
    stage: null,
    durationMs: null,
    retrievalMs: null,
    answer: null,
    generationMs: null,
    promptTokens: null,
    completionTokens: null,
    chunks: [],
    error: null,
  },
  {
    id: "example-incident",
    ordinal: 4,
    question: "What were the corrective actions after the August incident?",
    reference: "The incident postmortem records corrective actions and lessons learned.",
    status: "pending",
    stage: null,
    durationMs: null,
    retrievalMs: null,
    answer: null,
    generationMs: null,
    promptTokens: null,
    completionTokens: null,
    chunks: [],
    error: null,
  },
];
