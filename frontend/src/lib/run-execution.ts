import { type RunApiFailure, type RunResponse } from "@/validation/runs";

/** Stable execution stages presented in pipeline order after source selection. */
export type RunStageId =
  | "chunking"
  | "embedding"
  | "retrieval"
  | "generation"
  | "evaluation";

/** Visual states supported by every execution stage. */
export type RunStageStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "unavailable";

/** One backend-neutral stage shown by the execution timeline. */
export type RunStageView = {
  id: RunStageId;
  label: string;
  description: string;
  status: RunStageStatus;
};

/** Backend-neutral state consumed by the experiment execution modal. */
export type RunExecutionView = {
  status: "pending" | "running" | "completed" | "failed";
  corpusName: string;
  runId?: string;
  durationMs?: number;
  errorMessage?: string;
  chunking?: {
    chunkSetId: string;
    chunkCount: number;
    reused: boolean;
    durationMs: number;
  };
  embedding?: {
    vectorIndexId: string;
    vectorCount: number;
    dimensions: number;
    provider: string;
    model: string;
    distanceMetric: string;
    reused: boolean;
    durationMs: number;
  };
  stages: RunStageView[];
};

/** Future stages stay visible without implying that the backend executed them. */
const unavailableStages: RunStageView[] = [
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
 * Describes one persisted stage without inventing fractional progress.
 *
 * @param run - Validated run containing the latest persisted stage state.
 * @param stage - Chunking or embedding stage being presented.
 * @returns A concise description derived only from persisted facts.
 */
function describeStage(
  run: RunResponse,
  stage: "chunking" | "embedding",
): string {
  const stageResult = run[stage];

  // Active stages name the real work currently performed by the backend.
  if (stageResult.status === "running") {
    return stage === "chunking"
      ? "Creating reusable document chunks"
      : `Requesting vectors from ${run.embedding.provider}`;
  }

  // Completed stages distinguish cache reuse from newly materialized artifacts.
  if (stageResult.status === "completed") {
    return stageResult.reused
      ? `Reused compatible ${stage === "chunking" ? "chunks" : "vector index"}`
      : `Created ${stage === "chunking" ? "document chunks" : "vector index"}`;
  }

  // Failed stages stay explicit while the detailed safe message appears in the ledger.
  if (stageResult.status === "failed") {
    return `${stage === "chunking" ? "Chunking" : "Embedding"} did not complete`;
  }

  // A wholly pending run has not yet been claimed by the local worker.
  if (run.status === "pending" && stage === "chunking") {
    return "Waiting for the pipeline worker";
  }

  return "Waiting for the previous stage";
}

/**
 * Describes retrieval using the persisted run lifecycle without exposing hits.
 *
 * @param run - Validated run containing the active stage and safe failure data.
 * @returns A concise retrieval-stage description derived from persisted state.
 */
function describeRetrievalStage(run: RunResponse): string {
  // The backend exposes retrieval directly while the vector index is searched.
  if (run.current_stage === "retrieval") {
    return "Searching the vector index and saving ranked chunks";
  }

  // A completed run now guarantees that retrieval was persisted successfully.
  if (run.status === "completed") {
    return "Saved the ranked retrieval result";
  }

  // Safe failure provenance distinguishes retrieval from upstream failures.
  if (run.status === "failed" && run.error?.stage === "retrieval") {
    return "Retrieval did not complete";
  }

  return "Waiting for the vector index";
}

/**
 * Maps any persisted backend run state into the stable UI presentation model.
 *
 * @param run - Validated run returned by enqueueing or polling.
 * @param corpusName - Human-readable name captured when the run started.
 * @returns A presentation view containing only real persisted stage facts.
 */
export function createExecutionView(
  run: RunResponse,
  corpusName: string,
): RunExecutionView {
  const chunking =
    run.chunking.chunk_set_id !== null &&
    run.chunking.chunk_count !== null &&
    run.chunking.reused !== null &&
    run.chunking.duration_ms !== null
      ? {
          chunkSetId: run.chunking.chunk_set_id,
          chunkCount: run.chunking.chunk_count,
          reused: run.chunking.reused,
          durationMs: run.chunking.duration_ms,
        }
      : undefined;
  const embedding =
    run.embedding.vector_index_id !== null &&
    run.embedding.vector_count !== null &&
    run.embedding.dimensions !== null &&
    run.embedding.reused !== null &&
    run.embedding.duration_ms !== null
      ? {
          vectorIndexId: run.embedding.vector_index_id,
          vectorCount: run.embedding.vector_count,
          dimensions: run.embedding.dimensions,
          provider: run.embedding.provider,
          model: run.embedding.model,
          distanceMetric: run.embedding.distance_metric,
          reused: run.embedding.reused,
          durationMs: run.embedding.duration_ms,
        }
      : undefined;

  // Retrieval has lifecycle visibility even though result details remain internal.
  const retrievalStatus: RunStageStatus =
    run.current_stage === "retrieval"
      ? "running"
      : run.status === "completed"
        ? "completed"
        : run.status === "failed" && run.error?.stage === "retrieval"
          ? "failed"
          : "pending";

  return {
    status: run.status,
    corpusName,
    runId: run.id,
    durationMs: run.duration_ms ?? undefined,
    errorMessage: run.error?.message,
    chunking,
    embedding,
    stages: [
      {
        id: "chunking",
        label: "Chunk",
        description: describeStage(run, "chunking"),
        status: run.chunking.status,
      },
      {
        id: "embedding",
        label: "Embed",
        description: describeStage(run, "embedding"),
        status: run.embedding.status,
      },
      {
        id: "retrieval",
        label: "Retrieve",
        description: describeRetrievalStage(run),
        status: retrievalStatus,
      },
      ...unavailableStages,
    ],
  };
}

/**
 * Creates a retryable failure for requests rejected before a run was enqueued.
 *
 * @param failure - Safe backend failure details or a local fallback message.
 * @param corpusName - Human-readable name captured when submission started.
 * @returns A failed view that does not imply any pipeline stage began.
 */
export function createFailedExecution(
  failure: RunApiFailure,
  corpusName: string,
): RunExecutionView {
  return {
    status: "failed",
    corpusName,
    errorMessage: failure.message,
    stages: [
      {
        id: "chunking",
        label: "Chunk",
        description: "The run could not be enqueued",
        status: "unavailable",
      },
      {
        id: "embedding",
        label: "Embed",
        description: "Not started",
        status: "unavailable",
      },
      {
        id: "retrieval",
        label: "Retrieve",
        description: "Not started",
        status: "unavailable",
      },
      ...unavailableStages,
    ],
  };
}
