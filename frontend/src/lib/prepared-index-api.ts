import apiClient from "@/lib/axios";
import {
  type PreparedIndex,
  type PreparedIndexCreateRequest,
  type PreparedIndexStatus,
  parsePreparedIndex,
  parsePreparedIndexes,
} from "@/validation/prepared-indexes";

/**
 * Creates one durable named index-preparation request.
 *
 * @param payload - Validated name, corpus, chunking, and embedding inputs.
 * @param signal - Optional cancellation signal owned by the calling component.
 * @returns The validated pending prepared index returned by FastAPI.
 */
export async function createPreparedIndex(
  payload: PreparedIndexCreateRequest,
  signal?: AbortSignal,
): Promise<PreparedIndex> {
  const response = await apiClient.post<unknown>("/indexes", payload, { signal });
  return parsePreparedIndex(response.data);
}

/**
 * Lists prepared indexes with an optional exact lifecycle filter.
 *
 * @param status - Optional pending, running, ready, or failed filter.
 * @param signal - Optional cancellation signal owned by the calling component.
 * @returns Validated prepared indexes in newest-first backend order.
 */
export async function listPreparedIndexes(
  status?: PreparedIndexStatus,
  signal?: AbortSignal,
): Promise<PreparedIndex[]> {
  const response = await apiClient.get<unknown>("/indexes", {
    params: status ? { status } : undefined,
    signal,
  });
  return parsePreparedIndexes(response.data);
}

/**
 * Loads the latest lifecycle state for one prepared index.
 *
 * @param preparedIndexId - Stable identifier returned by the creation endpoint.
 * @param signal - Optional cancellation signal owned by the calling component.
 * @returns The validated current prepared-index state.
 */
export async function getPreparedIndex(
  preparedIndexId: string,
  signal?: AbortSignal,
): Promise<PreparedIndex> {
  const response = await apiClient.get<unknown>(`/indexes/${preparedIndexId}`, {
    signal,
  });
  return parsePreparedIndex(response.data);
}
