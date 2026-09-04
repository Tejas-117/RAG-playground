"use client";

import { useEffect, useRef, useState } from "react";
import {
  FiAlertTriangle,
  FiFilter,
  FiFileText,
  FiRefreshCw,
  FiTrash2,
  FiX,
} from "react-icons/fi";
import { deleteDataset, getDataset } from "@/lib/dataset-api";
import { isAxiosError, isCancel } from "@/lib/axios";
import {
  type DatasetDetail,
  type DatasetImportWarning,
  type DatasetSummary,
  parseDatasetApiError,
} from "@/validation/datasets";

/** Twelve digest characters provide recognizable provenance without visual noise. */
const DATASET_DIGEST_DISPLAY_LENGTH = 12;

/** Supported question views keep warning inspection local to the loaded dataset. */
const QUESTION_FILTER_OPTIONS = [
  { label: "All questions", value: "all" },
  { label: "With warnings", value: "warnings" },
  { label: "Without warnings", value: "clean" },
] as const;

/** Active filters use the dark ledger treatment to communicate selection. */
const ACTIVE_FILTER_CLASS =
  "border-[var(--charcoal)] bg-[var(--charcoal)] text-white";

/** Inactive filters remain quiet while retaining a visible interactive boundary. */
const INACTIVE_FILTER_CLASS =
  "border-[var(--border-strong)] bg-white text-[var(--tone-black)] " +
  "hover:bg-[var(--badge-surface)]";

/** One finite question view selected in the dataset inspector. */
type QuestionFilter = (typeof QUESTION_FILTER_OPTIONS)[number]["value"];

type DatasetDetailPanelProps = {
  dataset: DatasetSummary;
  initialDetail?: DatasetDetail | null;
  onClose: () => void;
  onDeleted: (dataset: DatasetSummary) => void;
};

/**
 * Formats a backend timestamp for local dataset provenance display.
 *
 * @param timestamp - UTC timestamp returned by the dataset API.
 * @returns Localized date and time, or the original value when parsing fails.
 */
function formatDatasetTimestamp(timestamp: string): string {
  const date = new Date(timestamp);

  // Preserve an unfamiliar backend timestamp instead of displaying an invalid date.
  if (Number.isNaN(date.getTime())) {
    return timestamp;
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

/**
 * Groups import warnings by the stable example they describe.
 *
 * @param warnings - Persisted non-fatal warnings returned with dataset detail.
 * @returns A lookup from example identity to its ordered import warnings.
 */
function groupWarningsByExample(
  warnings: DatasetImportWarning[],
): Map<string, DatasetImportWarning[]> {
  const groupedWarnings = new Map<string, DatasetImportWarning[]>();

  // Each warning belongs beside the question whose document label was skipped.
  for (const warning of warnings) {
    const exampleWarnings = groupedWarnings.get(warning.example_id) ?? [];
    exampleWarnings.push(warning);
    groupedWarnings.set(warning.example_id, exampleWarnings);
  }

  return groupedWarnings;
}

/**
 * Inspects one immutable dataset and supports guarded deletion.
 *
 * @param props - Selected summary, optional imported detail, and lifecycle callbacks.
 * @returns A native right-side dialog with provenance, questions, labels, and warnings.
 */
export default function DatasetDetailPanel({
  dataset,
  initialDetail = null,
  onClose,
  onDeleted,
}: DatasetDetailPanelProps) {
  // Stores the native dialog used for top-layer focus and backdrop behavior.
  const dialogRef = useRef<HTMLDialogElement | null>(null);

  // Stores the complete dataset returned by import or the detail endpoint.
  const [detail, setDetail] = useState<DatasetDetail | null>(initialDetail ?? null);

  // Indicates that examples and warnings are being hydrated from FastAPI.
  const [isLoading, setIsLoading] = useState(initialDetail === null);

  // Stores a safe detail-loading or deletion failure.
  const [error, setError] = useState<string | null>(null);

  // Reveals the destructive confirmation controls only after explicit intent.
  const [isConfirmingDelete, setIsConfirmingDelete] = useState(false);

  // Prevents repeated deletion requests while the backend is responding.
  const [isDeleting, setIsDeleting] = useState(false);

  // Controls whether the ledger shows every question, warnings, or clean examples.
  const [questionFilter, setQuestionFilter] = useState<QuestionFilter>("all");

  // Opens the side panel modally and restores focus after it unmounts.
  useEffect(() => {
    const dialog = dialogRef.current;
    const previousActiveElement = document.activeElement;

    // Native modal presentation traps focus while preserving the inventory behind it.
    if (dialog && !dialog.open) {
      dialog.showModal();
    }

    return () => {
      // Return focus to the inventory row that launched the inspector.
      if (previousActiveElement instanceof HTMLElement) {
        previousActiveElement.focus();
      }
    };
  }, []);

  // Loads detail only when the parent did not provide the complete import response.
  useEffect(() => {
    // Freshly imported datasets already contain all examples and warnings.
    if (initialDetail) {
      return;
    }

    const abortController = new AbortController();

    /**
     * Hydrates the selected dataset from its stable identifier.
     *
     * @returns A promise resolved after detail or safe failure state is stored.
     */
    async function loadDatasetDetail(): Promise<void> {
      setIsLoading(true);
      setError(null);

      try {
        const loadedDetail = await getDataset(dataset.id, abortController.signal);
        setDetail(loadedDetail);
      } catch (requestError) {
        // Effect cleanup intentionally cancels detail work after selection changes.
        if (isCancel(requestError)) {
          return;
        }

        const apiMessage = isAxiosError(requestError)
          ? parseDatasetApiError(requestError.response?.data)
          : null;
        setError(
          apiMessage ??
            (requestError instanceof Error
              ? requestError.message
              : "The evaluation dataset could not be loaded."),
        );
      } finally {
        // Avoid state updates after the selected inspector has unmounted.
        if (!abortController.signal.aborted) {
          setIsLoading(false);
        }
      }
    }

    void loadDatasetDetail();

    return () => {
      abortController.abort();
    };
  }, [dataset.id, initialDetail]);

  // Place each warning directly beside the evaluation example it affected.
  const warningsByExample = groupWarningsByExample(detail?.warnings ?? []);

  // Derive the visible ledger without mutating the immutable backend response order.
  const visibleExamples = (detail?.examples ?? []).filter((example) => {
    const hasWarnings = warningsByExample.has(example.id);

    // The warning view retains only questions whose document labels were skipped.
    if (questionFilter === "warnings") {
      return hasWarnings;
    }

    // The clean view excludes every question carrying an import warning.
    if (questionFilter === "clean") {
      return !hasWarnings;
    }

    return true;
  });

  /**
   * Requests panel closure when no destructive request is active.
   *
   * @returns Nothing. The parent removes the inspector from the page.
   */
  function closePanel(): void {
    // Keep an active deletion visible until the backend confirms its outcome.
    if (!isDeleting) {
      onClose();
    }
  }

  /**
   * Handles native Escape requests before the dialog closes itself.
   *
   * @param event - Cancel event emitted by the browser for the native dialog.
   * @returns Nothing. The event is redirected through the guarded close action.
   */
  function handleCancel(event: React.SyntheticEvent<HTMLDialogElement>): void {
    event.preventDefault();
    closePanel();
  }

  /**
   * Closes the panel when the user clicks the empty area beside it.
   *
   * @param event - Pointer event received by the full-viewport dialog.
   * @returns Nothing. Only the outer backdrop surface requests closure.
   */
  function handleBackdropClick(
    event: React.MouseEvent<HTMLDialogElement>,
  ): void {
    // Child interactions must never be mistaken for backdrop dismissal.
    if (event.target === event.currentTarget) {
      closePanel();
    }
  }

  /**
   * Deletes the selected dataset after the confirmation step.
   *
   * @returns A promise resolved after the inventory callback or safe error is shown.
   */
  async function handleDelete(): Promise<void> {
    setIsDeleting(true);
    setError(null);

    try {
      await deleteDataset(dataset.id);
      onDeleted(dataset);
    } catch (requestError) {
      // A protected dataset remains available and keeps its confirmation context.
      const apiMessage = isAxiosError(requestError)
        ? parseDatasetApiError(requestError.response?.data)
        : null;
      setError(
        apiMessage ??
          (requestError instanceof Error
            ? requestError.message
            : "The evaluation dataset could not be deleted."),
      );
    } finally {
      setIsDeleting(false);
    }
  }

  return (
    /* The viewport dialog reserves the right edge for a focused dataset inspector. */
    <dialog
      aria-labelledby="dataset-detail-title"
      className="dataset-inspector-dialog"
      onCancel={handleCancel}
      onClick={handleBackdropClick}
      ref={dialogRef}
    >
      {/* The inspector resembles a data ledger rather than a generic detail card. */}
      <aside
        className="dataset-inspector-shell ml-auto flex h-full w-full max-w-2xl
          flex-col border-l-2 border-[var(--charcoal)] bg-white"
      >
        {/* The dark ledger header separates identity from question-level evidence. */}
        <header className="relative shrink-0 bg-[var(--charcoal)] p-5 text-white sm:p-6">
          <button
            aria-label="Close dataset inspector"
            className="absolute right-4 top-4 rounded-sm p-1 text-white/70 hover:bg-white/10
              hover:text-white focus-visible:outline-2 focus-visible:outline-offset-2
              focus-visible:outline-white disabled:opacity-50"
            disabled={isDeleting}
            onClick={closePanel}
            type="button"
          >
            <FiX aria-hidden="true" className="size-5" />
          </button>
          <p
            className="font-mono text-[10px] font-bold uppercase tracking-[0.14em]
              text-white/60"
          >
            Evaluation dataset
          </p>
          <h2
            className="mt-2 max-w-[calc(100%-3rem)] text-2xl font-semibold
              tracking-[-0.03em]"
            id="dataset-detail-title"
          >
            {dataset.name}
          </h2>
          <p className="mt-2 font-mono text-xs text-white/65">
            {dataset.corpus_name} · {dataset.source_filename}
          </p>
        </header>

        {/* The scroll region contains immutable provenance and ordered examples. */}
        <div className="min-h-0 flex-1 overflow-y-auto">
          {/* The provenance strip exposes counts and source identity at a glance. */}
          <section
            aria-label="Dataset provenance"
            className="grid grid-cols-2 border-b border-[var(--border-strong)]
              bg-[var(--panel-surface)] sm:grid-cols-4"
          >
            {[
              ["Questions", String(dataset.example_count)],
              ["Labels", String(dataset.resolved_document_count)],
              ["Warnings", String(dataset.warning_count)],
              [
                "Digest",
                dataset.source_sha256.slice(0, DATASET_DIGEST_DISPLAY_LENGTH),
              ],
            ].map(([label, value]) => (
              <div
                className="border-b border-r border-[var(--border-subtle)] p-4
                  last:border-r-0 sm:border-b-0"
                key={label}
              >
                <p
                  className="font-mono text-[9px] font-bold uppercase
                    tracking-[0.12em] text-[var(--muted-text)]"
                >
                  {label}
                </p>
                <p className="mt-1 truncate font-mono text-sm text-[var(--charcoal)]">
                  {value}
                </p>
              </div>
            ))}
          </section>

          {/* Loading and failure states replace unavailable example content. */}
          {isLoading ? (
            <section className="flex items-center gap-3 p-6" aria-live="polite">
              <FiRefreshCw aria-hidden="true" className="size-4 animate-spin" />
              <p className="text-sm text-[var(--tone-black)]">
                Loading questions and document labels…
              </p>
            </section>
          ) : error && !detail ? (
            <section className="p-6" role="alert">
              <p className="text-sm text-[var(--toast-error-text)]">{error}</p>
            </section>
          ) : detail ? (
            <section aria-labelledby="dataset-examples-title" className="p-5 sm:p-6">
              <div className="flex items-end justify-between gap-4">
                <div>
                  <p
                    className="font-mono text-[10px] font-bold uppercase
                      tracking-[0.14em] text-[var(--tone-black)]"
                  >
                    Question ledger
                  </p>
                  <h3 className="mt-2 text-lg font-semibold" id="dataset-examples-title">
                    Retrieval evidence
                  </h3>
                </div>
                <p className="font-mono text-[10px] text-[var(--muted-text)]">
                  Imported {formatDatasetTimestamp(detail.created_at)}
                </p>
              </div>

              {/* The local filter narrows questions by their persisted warning state. */}
              <div
                aria-label="Filter dataset questions"
                className="mt-5 flex flex-wrap items-center gap-2 border-y
                  border-[var(--border-subtle)] bg-[var(--panel-surface)] p-2"
                role="group"
              >
                <span
                  className="mr-1 flex items-center gap-1.5 px-2 font-mono text-[9px]
                    font-bold uppercase tracking-[0.12em] text-[var(--muted-text)]"
                >
                  <FiFilter aria-hidden="true" className="size-3" />
                  View
                </span>

                {/* Each option applies one mutually exclusive warning-state filter. */}
                {QUESTION_FILTER_OPTIONS.map((option) => {
                  const isActive = questionFilter === option.value;

                  return (
                    <button
                      aria-pressed={isActive}
                      className={`rounded-sm border px-2.5 py-1.5 font-mono text-[10px]
                        font-semibold transition-colors ${
                          isActive
                            ? ACTIVE_FILTER_CLASS
                            : INACTIVE_FILTER_CLASS
                        }`}
                      key={option.value}
                      onClick={() => setQuestionFilter(option.value)}
                      type="button"
                    >
                      {option.label}
                    </button>
                  );
                })}

                <span className="ml-auto px-2 font-mono text-[10px] text-[var(--muted-text)]">
                  {visibleExamples.length} shown
                </span>
              </div>

              {/* Every ledger row preserves source order and its resolved evidence. */}
              <ol className="border-t border-[var(--border-strong)]">
                {visibleExamples.map((example) => {
                  const exampleWarnings = warningsByExample.get(example.id) ?? [];

                  return (
                    <li
                      className="grid gap-4 border-b border-[var(--border-strong)] py-5
                        sm:grid-cols-[3rem_minmax(0,1fr)]"
                      key={example.id}
                    >
                      {/* The ordinal maintains the exact order of the imported source. */}
                      <span
                        className="font-mono text-xs font-bold text-[var(--muted-text)]"
                      >
                        {String(example.ordinal + 1).padStart(2, "0")}
                      </span>

                      {/* The evidence column links each question to stable documents. */}
                      <div className="min-w-0">
                        <p className="text-sm font-semibold leading-6 text-[var(--charcoal)]">
                          {example.question}
                        </p>

                        {example.reference_answer ? (
                          <div className="mt-3 border-l-2 border-[var(--border-strong)] pl-3">
                            <p
                              className="font-mono text-[9px] font-bold uppercase
                                tracking-[0.1em] text-[var(--muted-text)]"
                            >
                              Reference answer
                            </p>
                            <p className="mt-1 text-xs leading-5 text-[var(--tone-black)]">
                              {example.reference_answer}
                            </p>
                          </div>
                        ) : null}

                        <div className="mt-4 flex flex-wrap gap-2">
                          {/* Resolved filenames expose the labels evaluation can trust. */}
                          {example.relevant_documents.length > 0 ? (
                            example.relevant_documents.map((document) => (
                              <span
                                className="flex items-center gap-1.5 rounded-sm
                                  bg-[var(--badge-surface)] px-2 py-1 font-mono text-[10px]"
                                key={document.id}
                              >
                                <FiFileText aria-hidden="true" className="size-3" />
                                {document.filename}
                              </span>
                            ))
                          ) : (
                            <span
                              className="rounded-sm border border-dashed
                                border-[var(--border-strong)] px-2 py-1 font-mono
                                text-[10px] text-[var(--muted-text)]"
                            >
                              Unlabelled for retrieval
                            </span>
                          )}
                        </div>

                        {/* Skipped labels remain attached to their affected question. */}
                        {exampleWarnings.length > 0 ? (
                          <ul className="mt-3 space-y-2">
                            {exampleWarnings.map((warning) => (
                              <li
                                className="flex items-start gap-2 text-xs leading-5
                                  text-[var(--toast-error-text)]"
                                key={`${warning.code}-${warning.document_name}`}
                              >
                                <FiAlertTriangle
                                  aria-hidden="true"
                                  className="mt-0.5 size-3.5 shrink-0"
                                />
                                <span>{warning.message}</span>
                              </li>
                            ))}
                          </ul>
                        ) : null}
                      </div>
                    </li>
                  );
                })}
              </ol>

              {/* An empty filter result explains that the dataset itself still has data. */}
              {visibleExamples.length === 0 ? (
                <div
                  className="border-b border-[var(--border-strong)] px-4 py-10 text-center"
                >
                  <p className="text-sm font-semibold text-[var(--charcoal)]">
                    No questions match this view.
                  </p>
                  <button
                    className="mt-3 font-mono text-[10px] font-bold uppercase
                      tracking-[0.1em] underline underline-offset-4"
                    onClick={() => setQuestionFilter("all")}
                    type="button"
                  >
                    Show all questions
                  </button>
                </div>
              ) : null}
            </section>
          ) : null}
        </div>

        {/* The footer isolates destructive management from immutable inspection. */}
        <footer
          className="shrink-0 border-t border-[var(--border-strong)]
            bg-[var(--page-surface)] p-4 sm:p-5"
        >
          {error && detail ? (
            <p className="mb-3 text-sm text-[var(--toast-error-text)]" role="alert">
              {error}
            </p>
          ) : null}

          {isConfirmingDelete ? (
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-xs leading-5 text-[var(--tone-black)]">
                Delete this immutable dataset? This cannot be undone.
              </p>
              <div className="flex shrink-0 gap-2">
                <button
                  className="rounded-sm border border-[var(--border-strong)] bg-white
                    px-3 py-2 text-xs font-semibold hover:bg-[var(--panel-surface)]
                    disabled:opacity-50"
                  disabled={isDeleting}
                  onClick={() => setIsConfirmingDelete(false)}
                  type="button"
                >
                  Cancel
                </button>
                <button
                  className="rounded-sm bg-[var(--toast-error-text)] px-3 py-2 text-xs
                    font-semibold text-white disabled:opacity-50"
                  disabled={isDeleting}
                  onClick={() => void handleDelete()}
                  type="button"
                >
                  {isDeleting ? "Deleting…" : "Delete dataset"}
                </button>
              </div>
            </div>
          ) : (
            <button
              className="flex items-center gap-2 text-xs font-semibold
                text-[var(--toast-error-text)] hover:underline
                focus-visible:outline-2 focus-visible:outline-offset-2
                focus-visible:outline-[var(--toast-error-text)]"
              onClick={() => setIsConfirmingDelete(true)}
              type="button"
            >
              <FiTrash2 aria-hidden="true" className="size-4" />
              Delete dataset
            </button>
          )}
        </footer>
      </aside>
    </dialog>
  );
}
