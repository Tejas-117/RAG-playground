import apiClient from "@/lib/axios";
import {
  type DatasetDetail,
  type DatasetImportInput,
  type DatasetListFilters,
  type DatasetSummary,
  parseDatasetDetail,
  parseDatasets,
} from "@/validation/datasets";

/**
 * Lists evaluation datasets using backend-owned corpus and name filtering.
 *
 * @param filters - Optional corpus identity and dataset-name query.
 * @param signal - Optional cancellation signal owned by the calling component.
 * @returns Validated dataset summaries in backend-provided order.
 */
export async function listDatasets(
  filters: DatasetListFilters = {},
  signal?: AbortSignal,
): Promise<DatasetSummary[]> {
  const response = await apiClient.get<unknown>("/datasets", {
    params: {
      corpus_id: filters.corpusId || undefined,
      search: filters.search || undefined,
    },
    signal,
  });
  return parseDatasets(response.data);
}

/**
 * Imports one immutable evaluation dataset as multipart form data.
 *
 * @param input - Dataset name, corpus identity, and JSON source file.
 * @param signal - Optional cancellation signal owned by the calling component.
 * @returns The validated persisted dataset including warnings and examples.
 */
export async function importDataset(
  input: DatasetImportInput,
  signal?: AbortSignal,
): Promise<DatasetDetail> {
  const formData = new FormData();
  formData.append("name", input.name);
  formData.append("corpus_id", input.corpusId);
  formData.append("file", input.file);

  const response = await apiClient.post<unknown>("/datasets", formData, {
    signal,
  });
  return parseDatasetDetail(response.data);
}

/**
 * Loads one dataset with ordered examples and import warnings.
 *
 * @param datasetId - Stable dataset identity selected from the inventory.
 * @param signal - Optional cancellation signal owned by the calling component.
 * @returns The validated fully hydrated dataset.
 */
export async function getDataset(
  datasetId: string,
  signal?: AbortSignal,
): Promise<DatasetDetail> {
  const response = await apiClient.get<unknown>(`/datasets/${datasetId}`, {
    signal,
  });
  return parseDatasetDetail(response.data);
}

/**
 * Deletes one dataset when benchmark history does not protect it.
 *
 * @param datasetId - Stable dataset identity selected for removal.
 * @param signal - Optional cancellation signal owned by the calling component.
 * @returns Nothing after FastAPI confirms deletion with an empty response.
 */
export async function deleteDataset(
  datasetId: string,
  signal?: AbortSignal,
): Promise<void> {
  await apiClient.delete(`/datasets/${datasetId}`, { signal });
}
