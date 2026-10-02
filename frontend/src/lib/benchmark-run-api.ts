import apiClient from "@/lib/axios";
import {
  type BenchmarkRunCreateRequest,
  type BenchmarkRunLaunch,
  parseBenchmarkRunLaunch,
  parseBenchmarkRunDetail,
  parseBenchmarkRuns,
  parseRetrievalEvaluationDetail,
  parseRetrievalEvaluations,
  type BenchmarkRunDetail,
  type BenchmarkRunSummary,
  type RetrievalEvaluationDetail,
  type RetrievalEvaluationSummary,
  type RetrievalMetric,
} from "@/validation/benchmark-runs";

/** Fetch validated history; optional signal cancels the request on hide or unmount. */
export async function listBenchmarkRuns(signal?: AbortSignal): Promise<BenchmarkRunSummary[]> {
  const response = await apiClient.get<unknown>("/runs", { signal });
  return parseBenchmarkRuns(response.data);
}

/** Fetch one validated run with all saved example results; signal cancels stale requests. */
export async function getBenchmarkRun(
  runId: string,
  signal?: AbortSignal,
): Promise<BenchmarkRunDetail> {
  const response = await apiClient.get<unknown>(`/runs/${encodeURIComponent(runId)}`, { signal });
  return parseBenchmarkRunDetail(response.data);
}

/**
 * Enqueue one dataset-wide benchmark against a ready prepared index.
 *
 * @param payload - Stable resource IDs and query-time configuration.
 * @param signal - Optional request cancellation signal.
 * @returns The validated pending benchmark returned by FastAPI.
 */
export async function createBenchmarkRun(
  payload: BenchmarkRunCreateRequest,
  signal?: AbortSignal,
): Promise<BenchmarkRunLaunch> {
  const response = await apiClient.post<unknown>("/runs", payload, { signal });
  return parseBenchmarkRunLaunch(response.data);
}

/**
 * List durable retrieval evaluation attempts for one benchmark.
 *
 * @param runId - Stable completed or active benchmark identifier.
 * @param signal - Optional request cancellation signal.
 * @returns Validated evaluation summaries ordered newest first.
 */
export async function listRetrievalEvaluations(
  runId: string,
  signal?: AbortSignal,
): Promise<RetrievalEvaluationSummary[]> {
  const encodedRunId = encodeURIComponent(runId);
  const response = await apiClient.get<unknown>(`/runs/${encodedRunId}/evaluations`, {
    signal,
  });
  return parseRetrievalEvaluations(response.data);
}

/**
 * Read one retrieval evaluation with its saved question-level evidence.
 *
 * @param runId - Stable parent benchmark identifier.
 * @param evaluationId - Stable evaluation attempt identifier.
 * @param signal - Optional request cancellation signal.
 * @returns The validated evaluation detail.
 */
export async function getRetrievalEvaluation(
  runId: string,
  evaluationId: string,
  signal?: AbortSignal,
): Promise<RetrievalEvaluationDetail> {
  const encodedRunId = encodeURIComponent(runId);
  const encodedEvaluationId = encodeURIComponent(evaluationId);
  const response = await apiClient.get<unknown>(
    `/runs/${encodedRunId}/evaluations/${encodedEvaluationId}`,
    { signal },
  );
  return parseRetrievalEvaluationDetail(response.data);
}

/**
 * Queue independent scoring of a completed benchmark's saved retrieval results.
 *
 * @param runId - Stable completed benchmark identifier.
 * @param metrics - Retrieval metrics selected for the new attempt.
 * @param signal - Optional request cancellation signal.
 * @returns The validated pending evaluation attempt.
 */
export async function createRetrievalEvaluation(
  runId: string,
  metrics: RetrievalMetric[],
  signal?: AbortSignal,
): Promise<RetrievalEvaluationDetail> {
  const encodedRunId = encodeURIComponent(runId);
  const response = await apiClient.post<unknown>(
    `/runs/${encodedRunId}/evaluations`,
    { retrieval_metrics: metrics },
    { signal },
  );
  return parseRetrievalEvaluationDetail(response.data);
}
