import { type RunApiFailure, type RunResponse } from "@/validation/runs";

/** Stable execution stages presented in pipeline order after source selection. */
export type RunStageId = "chunking" | "embedding" | "retrieval" | "generation" | "evaluation";

/** Visual states supported by every execution stage. */
export type RunStageStatus = "running" | "completed" | "failed" | "unavailable";

/** One backend-neutral stage shown by the execution timeline. */
export type RunStageView = {
  id: RunStageId;
  label: string;
  description: string;
  status: RunStageStatus;
};

/** Backend-neutral state consumed by the inline experiment execution panel. */
export type RunExecutionView = {
  status: "running" | "completed" | "failed";
  corpusName: string;
  runId?: string;
  durationMs?: number;
  errorMessage?: string;
  chunking?: {
    chunkSetId: string;
    chunkCount: number;
    reused: boolean;
  };
  stages: RunStageView[];
};

/** Future stages stay visible without implying that the backend executed them. */
const unavailableStages: RunStageView[] = [
  {
    id: "embedding",
    label: "Embed",
    description: "Not available yet",
    status: "unavailable",
  },
  {
    id: "retrieval",
    label: "Retrieve",
    description: "Not available yet",
    status: "unavailable",
  },
  {
    id: "generation",
    label: "Generate",
    description: "Not available yet",
    status: "unavailable",
  },
  {
    id: "evaluation",
    label: "Evaluate",
    description: "Not available yet",
    status: "unavailable",
  },
];

/**
 * Creates the honest client state displayed while the synchronous run request executes.
 *
 * @param corpusName - Human-readable name of the corpus snapshot being processed.
 * @returns A running view with chunking active and later stages unavailable.
 */
export function createRunningExecution(corpusName: string): RunExecutionView {
  return {
    status: "running",
    corpusName,
    stages: [
      {
        id: "chunking",
        label: "Chunk",
        description: "Preparing document chunks",
        status: "running",
      },
      ...unavailableStages,
    ],
  };
}

/**
 * Maps the current completed backend contract into the stable UI presentation model.
 *
 * @param run - Validated completed run returned by POST /runs.
 * @param corpusName - Human-readable name captured when the run started.
 * @returns A completed view containing real chunking results.
 */
export function createCompletedExecution(
  run: RunResponse,
  corpusName: string,
): RunExecutionView {
  return {
    status: "completed",
    corpusName,
    runId: run.id,
    durationMs: run.duration_ms,
    chunking: {
      chunkSetId: run.chunking.chunk_set_id,
      chunkCount: run.chunking.chunk_count,
      reused: run.chunking.reused,
    },
    stages: [
      {
        id: "chunking",
        label: "Chunk",
        description: run.chunking.reused ? "Reused saved chunks" : "Created document chunks",
        status: "completed",
      },
      ...unavailableStages,
    ],
  };
}

/**
 * Creates a retryable failed view without exposing unvalidated transport details.
 *
 * @param failure - Safe backend failure details or a local fallback message.
 * @param corpusName - Human-readable name captured when the run started.
 * @returns A failed view that identifies chunking when it is the known failed stage.
 */
export function createFailedExecution(
  failure: RunApiFailure,
  corpusName: string,
): RunExecutionView {
  // A persisted run ID identifies an execution failure even before stage is returned publicly.
  const failedStage = failure.stage === "chunking" || failure.runId !== undefined;

  return {
    status: "failed",
    corpusName,
    runId: failure.runId,
    errorMessage: failure.message,
    stages: [
      {
        id: "chunking",
        label: "Chunk",
        description: failedStage ? "Chunking did not complete" : "Execution stopped",
        status: failedStage ? "failed" : "unavailable",
      },
      ...unavailableStages,
    ],
  };
}
