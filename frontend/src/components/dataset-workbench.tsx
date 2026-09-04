"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  FiAlertCircle,
  FiChevronRight,
  FiFileText,
  FiPlus,
  FiRefreshCw,
  FiSearch,
} from "react-icons/fi";
import DatasetDetailPanel from "@/components/dataset-detail-panel";
import DatasetImportDialog from "@/components/dataset-import-dialog";
import Toast, { type ToastType } from "@/components/toast";
import WorkbenchGridCanvas from "@/components/workbench-grid-canvas";
import WorkbenchSidebar from "@/components/workbench-sidebar";
import apiClient, { isCancel } from "@/lib/axios";
import { listDatasets } from "@/lib/dataset-api";
import { type CorpusOption, parseCorpora } from "@/validation/corpora";
import { type DatasetDetail, type DatasetSummary } from "@/validation/datasets";

/** Delay applied to name searches before requesting filtered backend results. */
const DATASET_SEARCH_DEBOUNCE_MS = 300;

/** Eight characters keep stable IDs recognizable without overwhelming table rows. */
const DATASET_ID_DISPLAY_LENGTH = 8;

type Notice = {
  message: string;
  type: ToastType;
};

/**
 * Formats an import timestamp for compact local inventory display.
 *
 * @param timestamp - UTC timestamp returned by FastAPI.
 * @returns Localized date and time, or the original value when parsing fails.
 */
function formatDatasetTimestamp(timestamp: string): string {
  const date = new Date(timestamp);

  // Preserve unfamiliar backend values instead of showing an invalid date label.
  if (Number.isNaN(date.getTime())) {
    return timestamp;
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

/**
 * Converts a complete import response into the inventory summary contract.
 *
 * @param dataset - Fully hydrated dataset returned after a successful import.
 * @returns The summary fields needed by the inventory and inspector launcher.
 */
function datasetDetailToSummary(dataset: DatasetDetail): DatasetSummary {
  return {
    id: dataset.id,
    name: dataset.name,
    corpus_id: dataset.corpus_id,
    corpus_name: dataset.corpus_name,
    source_filename: dataset.source_filename,
    source_sha256: dataset.source_sha256,
    example_count: dataset.example_count,
    resolved_document_count: dataset.resolved_document_count,
    warning_count: dataset.warning_count,
    created_at: dataset.created_at,
  };
}

/**
 * Renders the connected evaluation-dataset management workspace.
 *
 * @returns The searchable inventory, import dialog, inspector, and notices.
 */
export default function DatasetWorkbench() {
  // Stores persisted corpora available for import and inventory filtering.
  const [corpora, setCorpora] = useState<CorpusOption[]>([]);

  // Stores validated dataset summaries returned by the active filters.
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);

  // Stores the immediate dataset-name query visible in the search field.
  const [search, setSearch] = useState("");

  // Stores the delayed query sent to the backend after typing settles.
  const [debouncedSearch, setDebouncedSearch] = useState("");

  // Stores the exact corpus identity used for optional server-side filtering.
  const [corpusFilter, setCorpusFilter] = useState("");

  // Indicates that corpus options are being loaded for the first time.
  const [isLoadingCorpora, setIsLoadingCorpora] = useState(true);

  // Indicates that the current dataset inventory request is active.
  const [isLoadingDatasets, setIsLoadingDatasets] = useState(true);

  // Stores a safe corpus-loading failure that blocks importing.
  const [corporaError, setCorporaError] = useState<string | null>(null);

  // Stores a safe inventory loading or filtering failure.
  const [datasetsError, setDatasetsError] = useState<string | null>(null);

  // Increments when the user explicitly retries loading page data.
  const [loadAttempt, setLoadAttempt] = useState(0);

  // Controls whether the focused multipart import dialog is mounted.
  const [isImportOpen, setIsImportOpen] = useState(false);

  // Stores the dataset whose examples should be inspected.
  const [selectedDataset, setSelectedDataset] = useState<DatasetSummary | null>(null);

  // Reuses a complete import response to avoid an immediate redundant detail request.
  const [selectedDatasetDetail, setSelectedDatasetDetail] =
    useState<DatasetDetail | null>(null);

  // Stores the latest successful management action or request failure notice.
  const [notice, setNotice] = useState<Notice | null>(null);

  // Loads and validates corpus options when the page opens or explicitly retries.
  useEffect(() => {
    const abortController = new AbortController();

    /**
     * Loads corpus identities used by both filtering and document-name resolution.
     *
     * @returns A promise resolved after corpus or safe failure state is updated.
     */
    async function loadCorpora(): Promise<void> {
      setIsLoadingCorpora(true);
      setCorporaError(null);

      try {
        const response = await apiClient.get<unknown>("/corpora/", {
          signal: abortController.signal,
        });
        setCorpora(parseCorpora(response.data));
      } catch (error) {
        // Navigation and retry cleanup intentionally cancel obsolete requests.
        if (isCancel(error)) {
          return;
        }

        setCorporaError(
          error instanceof Error
            ? error.message
            : "Corpora could not be loaded for dataset management.",
        );
      } finally {
        // Avoid updating loading state after the page or retry effect has unmounted.
        if (!abortController.signal.aborted) {
          setIsLoadingCorpora(false);
        }
      }
    }

    void loadCorpora();

    return () => {
      abortController.abort();
    };
  }, [loadAttempt]);

  // Debounces user typing so name filtering does not issue one request per keystroke.
  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      setDebouncedSearch(search.trim());
    }, DATASET_SEARCH_DEBOUNCE_MS);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [search]);

  // Loads the dataset inventory whenever filters or an explicit retry change.
  useEffect(() => {
    const abortController = new AbortController();

    /**
     * Loads dataset summaries using the current backend-owned filters.
     *
     * @returns A promise resolved after inventory or safe failure state is updated.
     */
    async function loadDatasetInventory(): Promise<void> {
      setIsLoadingDatasets(true);
      setDatasetsError(null);

      try {
        const loadedDatasets = await listDatasets(
          {
            corpusId: corpusFilter || undefined,
            search: debouncedSearch || undefined,
          },
          abortController.signal,
        );
        setDatasets(loadedDatasets);
      } catch (error) {
        // Filter changes and navigation intentionally cancel stale inventory requests.
        if (isCancel(error)) {
          return;
        }

        setDatasetsError(
          error instanceof Error
            ? error.message
            : "Evaluation datasets could not be loaded.",
        );
      } finally {
        // Avoid updating loading state after this filter request becomes obsolete.
        if (!abortController.signal.aborted) {
          setIsLoadingDatasets(false);
        }
      }
    }

    void loadDatasetInventory();

    return () => {
      abortController.abort();
    };
  }, [corpusFilter, debouncedSearch, loadAttempt]);

  // Import is available only when corpus identities are valid and fully loaded.
  const canImport = corpora.length > 0 && !isLoadingCorpora && !corporaError;

  // Distinguish an empty database from a filter that simply has no matches.
  const hasActiveFilters = Boolean(corpusFilter || debouncedSearch);

  /**
   * Opens an existing dataset and requests fresh detail from FastAPI.
   *
   * @param dataset - Inventory record selected for inspection.
   * @returns Nothing. Inspector selection state is updated in place.
   */
  function inspectDataset(dataset: DatasetSummary): void {
    setSelectedDatasetDetail(null);
    setSelectedDataset(dataset);
  }

  /**
   * Integrates a successful import into the inventory and opens its warnings.
   *
   * @param dataset - Complete immutable dataset returned by the import endpoint.
   * @returns Nothing. Filters, inventory, dialog, selection, and notice are updated.
   */
  function handleDatasetImported(dataset: DatasetDetail): void {
    const summary = datasetDetailToSummary(dataset);
    setSearch("");
    setDebouncedSearch("");
    setCorpusFilter("");
    setDatasets((current) => [
      summary,
      ...current.filter((item) => item.id !== summary.id),
    ]);
    setIsImportOpen(false);
    setNotice({
      message:
        dataset.warning_count > 0
          ? `“${dataset.name}” was imported with ${dataset.warning_count} warning(s).`
          : `“${dataset.name}” was imported.`,
      type: "success",
    });

    // Warnings need immediate inspection; clean imports remain in the inventory view.
    if (dataset.warning_count > 0) {
      setSelectedDatasetDetail(dataset);
      setSelectedDataset(summary);
    }
  }

  /**
   * Removes a confirmed dataset from the current inventory after backend deletion.
   *
   * @param dataset - Summary of the dataset confirmed as deleted by FastAPI.
   * @returns Nothing. The inspector closes and a success notice is displayed.
   */
  function handleDatasetDeleted(dataset: DatasetSummary): void {
    setDatasets((current) => current.filter((item) => item.id !== dataset.id));
    setSelectedDataset(null);
    setSelectedDatasetDetail(null);
    setNotice({
      message: `“${dataset.name}” was deleted.`,
      type: "success",
    });
  }

  return (
    <main className="grid min-h-screen lg:grid-cols-[17.5rem_minmax(0,1fr)]">
      <WorkbenchSidebar activeLabel="Datasets" />

      {/* The grid canvas contains the complete dataset-management workbench. */}
      <WorkbenchGridCanvas className="min-h-screen px-5 py-8 sm:px-8 lg:px-12 lg:py-10">
        {/* The page header frames datasets as reusable benchmark inputs. */}
        <header
          className="mx-auto flex max-w-[74rem] flex-col gap-6 border-b-2
            border-[var(--charcoal)] pb-7 sm:flex-row sm:items-end sm:justify-between"
        >
          <div>
            <p
              className="font-mono text-[10px] font-bold uppercase tracking-[0.16em]
                text-[var(--tone-black)]"
            >
              Evaluation datasets
            </p>
            <h1
              className="mt-3 max-w-3xl text-3xl font-semibold tracking-[-0.035em]
                text-[var(--charcoal)] sm:text-4xl"
            >
              Register the questions that test your pipeline.
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-[var(--tone-black)]">
              Import stable question sets, resolve document labels, and inspect exactly
              what each future benchmark will measure.
            </p>
          </div>

          {/* The primary action opens import only when a target corpus exists. */}
          <button
            className="flex w-fit items-center gap-2 rounded-sm bg-[var(--charcoal)]
              px-4 py-3 text-sm font-semibold text-white hover:bg-[var(--primary-hover)]
              focus-visible:outline-2 focus-visible:outline-offset-2
              focus-visible:outline-[var(--charcoal)] disabled:cursor-not-allowed
              disabled:opacity-50"
            disabled={!canImport}
            onClick={() => setIsImportOpen(true)}
            type="button"
          >
            <FiPlus aria-hidden="true" className="size-5" />
            Import dataset
          </button>
        </header>

        {/* Missing corpora explain why importing is unavailable and provide the next step. */}
        {!isLoadingCorpora && !corporaError && corpora.length === 0 ? (
          <section
            className="mx-auto mt-8 flex max-w-[74rem] items-start gap-4 border
              border-[var(--border-strong)] bg-white p-5 sm:p-6"
          >
            <FiFileText aria-hidden="true" className="mt-0.5 size-5 shrink-0" />
            <div>
              <h2 className="font-semibold text-[var(--charcoal)]">
                Upload documents before importing a dataset.
              </h2>
              <p className="mt-1 text-sm leading-6 text-[var(--tone-black)]">
                A corpus is required to turn relevant document filenames into stable IDs.
              </p>
              <Link
                className="mt-3 inline-flex items-center gap-1 text-sm font-semibold
                  underline underline-offset-4"
                href="/ingestion"
              >
                Go to documents
                <FiChevronRight aria-hidden="true" className="size-4" />
              </Link>
            </div>
          </section>
        ) : null}

        {/* Loading failures retain a retry path without exposing partial inventory state. */}
        {corporaError ? (
          <section
            className="mx-auto mt-8 flex max-w-[74rem] items-start justify-between gap-5
              border border-[var(--toast-error-border)] bg-[var(--toast-error-surface)]
              p-5 text-[var(--toast-error-text)]"
            role="alert"
          >
            <div className="flex items-start gap-3">
              <FiAlertCircle aria-hidden="true" className="mt-0.5 size-5 shrink-0" />
              <p className="text-sm">{corporaError}</p>
            </div>
            <button
              className="shrink-0 text-sm font-semibold underline underline-offset-4"
              onClick={() => setLoadAttempt((current) => current + 1)}
              type="button"
            >
              Retry
            </button>
          </section>
        ) : null}

        {/* The inventory surface combines server filters and dataset provenance. */}
        <section
          className="mx-auto mt-8 max-w-[74rem] overflow-hidden border
            border-[var(--border-strong)] bg-white"
        >
          {/* Search and corpus filters control the backend inventory query. */}
          <div
            className="flex flex-col gap-4 border-b border-[var(--border-strong)]
              p-5 sm:flex-row sm:items-end sm:justify-between sm:p-6"
          >
            <div>
              <p
                className="font-mono text-[10px] font-bold uppercase tracking-[0.12em]
                  text-[var(--tone-black)]"
              >
                Dataset inventory
              </p>
              <h2 className="mt-2 text-xl font-semibold text-[var(--charcoal)]">
                Imported question sets
              </h2>
            </div>

            <div className="flex w-full flex-col gap-3 sm:w-auto sm:flex-row">
              <label className="relative block sm:w-64" htmlFor="dataset-search">
                <span className="sr-only">Search datasets by name</span>
                <FiSearch
                  aria-hidden="true"
                  className="absolute left-3 top-1/2 size-4 -translate-y-1/2
                    text-[var(--muted-text)]"
                />
                <input
                  className="w-full rounded-sm border border-[var(--border-strong)]
                    bg-white py-2.5 pl-9 pr-3 font-mono text-xs outline-none
                    focus:border-[var(--charcoal)] focus:ring-1 focus:ring-[var(--charcoal)]"
                  id="dataset-search"
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Search by name"
                  type="search"
                  value={search}
                />
              </label>

              <label className="block sm:w-52" htmlFor="dataset-corpus-filter">
                <span className="sr-only">Filter datasets by corpus</span>
                <select
                  className="w-full rounded-sm border border-[var(--border-strong)]
                    bg-white px-3 py-2.5 font-mono text-xs outline-none
                    focus:border-[var(--charcoal)] focus:ring-1 focus:ring-[var(--charcoal)]"
                  id="dataset-corpus-filter"
                  onChange={(event) => setCorpusFilter(event.target.value)}
                  value={corpusFilter}
                >
                  <option value="">All corpora</option>
                  {/* Corpus options map readable names to exact backend filter identities. */}
                  {corpora.map((corpus) => (
                    <option key={corpus.id} value={corpus.id}>
                      {corpus.name}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </div>

          {/* Loading, failure, and empty states replace the unavailable table body. */}
          {isLoadingDatasets ? (
            <div className="flex items-center gap-3 p-8" aria-live="polite">
              <FiRefreshCw aria-hidden="true" className="size-5 animate-spin" />
              <p className="text-sm text-[var(--tone-black)]">
                Loading evaluation datasets…
              </p>
            </div>
          ) : datasetsError ? (
            <div className="flex items-start justify-between gap-5 p-8" role="alert">
              <div className="flex items-start gap-3 text-[var(--toast-error-text)]">
                <FiAlertCircle aria-hidden="true" className="mt-0.5 size-5 shrink-0" />
                <p className="text-sm">{datasetsError}</p>
              </div>
              <button
                className="text-sm font-semibold underline underline-offset-4"
                onClick={() => setLoadAttempt((current) => current + 1)}
                type="button"
              >
                Retry
              </button>
            </div>
          ) : datasets.length === 0 ? (
            <div className="p-8 text-center sm:p-12">
              <FiFileText
                aria-hidden="true"
                className="mx-auto size-6 text-[var(--muted-text)]"
              />
              <h3 className="mt-4 font-semibold text-[var(--charcoal)]">
                {hasActiveFilters ? "No datasets match these filters." : "No datasets yet."}
              </h3>
              <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-[var(--tone-black)]">
                {hasActiveFilters
                  ? "Clear the search or choose another corpus to widen the inventory."
                  : "Import a JSON question set to create a reusable benchmark input."}
              </p>
              {hasActiveFilters ? (
                <button
                  className="mt-4 text-sm font-semibold underline underline-offset-4"
                  onClick={() => {
                    setSearch("");
                    setDebouncedSearch("");
                    setCorpusFilter("");
                  }}
                  type="button"
                >
                  Clear filters
                </button>
              ) : null}
            </div>
          ) : (
            /* The compact ledger table exposes enough provenance to choose a dataset. */
            <div className="overflow-x-auto">
              <table className="w-full min-w-[58rem] border-collapse text-left">
                <thead className="bg-[var(--panel-surface)]">
                  <tr
                    className="border-b border-[var(--border-strong)] font-mono text-[10px]
                      font-bold uppercase tracking-[0.1em] text-[var(--tone-black)]"
                  >
                    <th className="px-5 py-3 font-inherit sm:px-6">Dataset</th>
                    <th className="px-5 py-3 font-inherit sm:px-6">Corpus</th>
                    <th className="px-5 py-3 text-right font-inherit sm:px-6">Questions</th>
                    <th className="px-5 py-3 text-right font-inherit sm:px-6">Labels</th>
                    <th className="px-5 py-3 text-right font-inherit sm:px-6">Warnings</th>
                    <th className="px-5 py-3 font-inherit sm:px-6">Imported</th>
                    <th className="px-5 py-3 font-inherit sm:px-6">
                      <span className="sr-only">Inspect</span>
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border-subtle)]">
                  {/* Every row represents one immutable corpus-scoped dataset. */}
                  {datasets.map((dataset) => (
                    <tr
                      className="transition-colors hover:bg-[var(--page-surface)]"
                      key={dataset.id}
                    >
                      <td className="px-5 py-4 align-top sm:px-6">
                        <button
                          className="max-w-64 text-left focus-visible:outline-2
                            focus-visible:outline-offset-2
                            focus-visible:outline-[var(--charcoal)]"
                          onClick={() => inspectDataset(dataset)}
                          type="button"
                        >
                          <span
                            className="block truncate font-mono text-xs font-semibold
                              text-[var(--charcoal)]"
                          >
                            {dataset.name}
                          </span>
                          <span
                            className="mt-1 block truncate font-mono text-[9px]
                              text-[var(--muted-text)]"
                          >
                            {dataset.source_filename} · ID{" "}
                            {dataset.id.slice(0, DATASET_ID_DISPLAY_LENGTH)}
                          </span>
                        </button>
                      </td>
                      <td className="px-5 py-4 text-xs text-[var(--tone-black)] sm:px-6">
                        {dataset.corpus_name}
                      </td>
                      <td
                        className="px-5 py-4 text-right font-mono text-xs
                          text-[var(--charcoal)] sm:px-6"
                      >
                        {dataset.example_count}
                      </td>
                      <td
                        className="px-5 py-4 text-right font-mono text-xs
                          text-[var(--charcoal)] sm:px-6"
                      >
                        {dataset.resolved_document_count}
                      </td>
                      <td className="px-5 py-4 text-right sm:px-6">
                        <span
                          className={
                            dataset.warning_count > 0
                              ? "font-mono text-xs font-semibold text-[var(--toast-error-text)]"
                              : "font-mono text-xs text-[var(--muted-text)]"
                          }
                        >
                          {dataset.warning_count}
                        </span>
                      </td>
                      <td
                        className="whitespace-nowrap px-5 py-4 font-mono text-[10px]
                          text-[var(--muted-text)] sm:px-6"
                      >
                        {formatDatasetTimestamp(dataset.created_at)}
                      </td>
                      <td className="px-5 py-4 text-right sm:px-6">
                        <button
                          aria-label={`Inspect ${dataset.name}`}
                          className="rounded-sm p-1.5 hover:bg-[var(--hover-surface)]
                            focus-visible:outline-2 focus-visible:outline-offset-2
                            focus-visible:outline-[var(--charcoal)]"
                          onClick={() => inspectDataset(dataset)}
                          type="button"
                        >
                          <FiChevronRight aria-hidden="true" className="size-4" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </WorkbenchGridCanvas>

      {/* The import dialog owns multipart values until success or explicit cancellation. */}
      {isImportOpen ? (
        <DatasetImportDialog
          corpora={corpora}
          onClose={() => setIsImportOpen(false)}
          onImported={handleDatasetImported}
        />
      ) : null}

      {/* The inspector hydrates examples without navigating away from the inventory. */}
      {selectedDataset ? (
        <DatasetDetailPanel
          dataset={selectedDataset}
          initialDetail={selectedDatasetDetail}
          onClose={() => {
            setSelectedDataset(null);
            setSelectedDatasetDetail(null);
          }}
          onDeleted={handleDatasetDeleted}
        />
      ) : null}

      {/* Management outcomes remain visible after dialogs close. */}
      {notice ? (
        <Toast
          message={notice.message}
          onDismiss={() => setNotice(null)}
          type={notice.type}
        />
      ) : null}
    </main>
  );
}
