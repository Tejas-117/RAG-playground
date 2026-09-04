import apiClient from "@/lib/axios";
import {
  type BenchmarkRunCreateRequest,
  type BenchmarkRunLaunch,
  parseBenchmarkRunLaunch,
} from "@/validation/benchmark-runs";

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
