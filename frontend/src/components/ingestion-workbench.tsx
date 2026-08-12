"use client";

import { useEffect, useRef, useState } from "react";
import {
  FiAlertTriangle,
  FiCheck,
  FiChevronDown,
  FiFileText,
  FiFolder,
  FiPlus,
  FiRefreshCw,
  FiX,
} from "react-icons/fi";
import Toast, { ToastType } from "@/components/toast";
import UploadForm from "@/components/upload-form";
import WorkbenchGridCanvas from "@/components/workbench-grid-canvas";
import WorkbenchSidebar from "@/components/workbench-sidebar";
import apiClient, { isAxiosError } from "@/lib/axios";
import {
  type IngestionCorpusPayload,
  parseIngestionCorpora,
} from "@/validation/ingestion";

type Corpus = {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  documents: DocumentRecord[];
  isNew?: boolean;
};

type DocumentRecord = {
  id: string;
  originalFilename: string;
  mimeType: string | null;
  sizeBytes: number;
  uploadedAt: string;
  parse: ParseSummary | null;
};

type ParseSummary = {
  parserName: string;
  parserVersion: string;
  warnings: string[];
  pageCount: number;
  blockCount: number;
  utf8SizeBytes: number;
  characterCount: number;
  durationMs: number;
};

type Notice = {
  type: ToastType;
  message: string;
};

/**
 * Converts the backend response into the camelCase model used by the UI.
 *
 * @param response - The response payload returned by GET /corpora/.
 * @returns Corpus records sorted newest first with documents sorted by upload date.
 */
function normalizeCorpora(corpora: IngestionCorpusPayload[]): Corpus[] {
  return corpora
    .map((corpus) => ({
      id: corpus.id,
      name: corpus.name,
      createdAt: corpus.created_at,
      updatedAt: corpus.updated_at,
      documents: corpus.documents
        .map((document) => ({
          id: document.id,
          originalFilename: document.original_filename,
          mimeType: document.mime_type,
          sizeBytes: document.size_bytes,
          uploadedAt: document.uploaded_at,
          parse: document.parse
            ? {
                parserName: document.parse.parser_name,
                parserVersion: document.parse.parser_version,
                warnings: document.parse.warnings,
                pageCount: document.parse.page_count,
                blockCount: document.parse.block_count,
                utf8SizeBytes: document.parse.utf8_size_bytes,
                characterCount: document.parse.character_count,
                durationMs: document.parse.duration_ms,
              }
            : null,
        }))
        .sort((left, right) => right.uploadedAt.localeCompare(left.uploadedAt)),
    }))
    .sort((left, right) => right.createdAt.localeCompare(left.createdAt));
}

/**
 * Formats an ISO timestamp for the user's local timezone.
 *
 * @param timestamp - ISO timestamp returned by the backend.
 * @returns A compact local date and time string.
 */
function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);

  if (Number.isNaN(date.getTime())) {
    return "Unknown date";
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

/**
 * Formats a document byte count for its metadata label.
 *
 * @param sizeBytes - Document size in bytes.
 * @returns A human-readable file size.
 */
function formatFileSize(sizeBytes: number): string {
  if (sizeBytes < 1024) {
    return `${sizeBytes} B`;
  }

  if (sizeBytes < 1024 * 1024) {
    return `${(sizeBytes / 1024).toFixed(1)} KB`;
  }

  return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Format parser latency for compact document telemetry.
 *
 * @param durationMs - Completed parsing duration in milliseconds.
 * @returns Milliseconds below one second, otherwise seconds with one decimal place.
 */
function formatDuration(durationMs: number): string {
  // Preserve precise millisecond information for quick parser executions.
  if (durationMs < 1000) {
    return `${durationMs} ms`;
  }

  return `${(durationMs / 1000).toFixed(1)} s`;
}

/**
 * Render canonical parsing provenance and warnings for one persisted document.
 *
 * @param props - Optional parse summary returned for the document.
 * @returns Compact parser telemetry or a legacy-document fallback.
 */
function DocumentParseDetails({ parse }: { parse: ParseSummary | null }) {
  // Legacy rows can exist without a canonical parse artifact.
  if (!parse) {
    return (
      <span className="mt-2 block font-mono text-[9px] uppercase tracking-wide
        text-[var(--muted-text)]">
        Parse details unavailable
      </span>
    );
  }

  return (
    /* Parsing trace groups the document's primary extraction measurements into one strip. */
    <div className="mt-2 min-w-0">
      <div
        className="inline-flex max-w-full flex-wrap items-center gap-x-2 gap-y-1 rounded-sm
          bg-[var(--badge-surface)] px-3 py-1.5 font-mono text-[11px] font-medium
          uppercase tracking-[0.04em] text-[var(--charcoal)] sm:gap-x-3 sm:text-xs"
      >
        <span className="inline-flex items-center gap-1.5">
          <FiCheck aria-hidden="true" className="size-3.5" />
          Parsed
        </span>
        <span aria-hidden="true" className="text-[var(--muted-text)]">
          |
        </span>
        <span>{parse.pageCount} pages</span>
        <span aria-hidden="true" className="text-[var(--muted-text)]">
          |
        </span>
        <span>{parse.blockCount} blocks</span>
        <span aria-hidden="true" className="text-[var(--muted-text)]">
          |
        </span>
        <span>{formatFileSize(parse.utf8SizeBytes)} text</span>
        <span aria-hidden="true" className="text-[var(--muted-text)]">
          |
        </span>
        <span>{formatDuration(parse.durationMs)}</span>
      </div>

      {/* Parser warnings remain visible without treating a successful parse as a failure. */}
      {parse.warnings.length > 0 ? (
        <ul className="mt-2 space-y-1.5" role="note">
          {/* Each parser warning receives its own readable alert line. */}
          {parse.warnings.map((warning, warningIndex) => (
            <li
              className="flex items-start gap-2 text-[13px] leading-5
                text-[var(--toast-error-text)]"
              key={`${warning}-${warningIndex}`}
            >
              <FiAlertTriangle aria-hidden="true" className="mt-0.5 size-4 shrink-0" />
              <span className="break-words">{warning}</span>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

/**
 * Renders the corpus workspace, upload modal, and upload notices.
 *
 * @returns The interactive ingestion workspace.
 */
export default function IngestionWorkbench() {
  // Stores the corpora currently displayed in the workbench table.
  const [corpora, setCorpora] = useState<Corpus[]>([]);

  // Indicates that the persisted corpus inventory is being loaded or refreshed.
  const [isLoadingCorpora, setIsLoadingCorpora] = useState(true);

  // Stores a readable error when the persisted corpus inventory cannot be loaded.
  const [corporaError, setCorporaError] = useState<string | null>(null);

  // Tracks which corpus document list is currently expanded.
  const [expandedCorpusId, setExpandedCorpusId] = useState<string | null>(null);

  // Tracks the corpora whose document rows expose parsing details and warnings.
  const [parseSummaryCorpusIds, setParseSummaryCorpusIds] = useState<Set<string>>(
    () => new Set(),
  );

  // Controls whether the upload dialog is visible.
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Prevents the synchronous parsing request from being hidden or submitted twice.
  const [isUploadPending, setIsUploadPending] = useState(false);

  // Holds the latest success or failure message shown to the user.
  const [notice, setNotice] = useState<Notice | null>(null);

  // Keeps the upload dialog's close control available for focus management.
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  /**
   * Fetches the latest corpus and document inventory from the backend.
   *
   * @returns A promise resolved after the inventory state is updated.
   */
  async function fetchCorpora(): Promise<void> {
    setIsLoadingCorpora(true);
    setCorporaError(null);

    try {
      const response = await apiClient.get<unknown>("/corpora/");
      const validatedCorpora = parseIngestionCorpora(response.data);
      setCorpora(normalizeCorpora(validatedCorpora));
    } catch (error) {
      const message = isAxiosError(error)
        ? "The uploaded corpus details could not be loaded."
        : error instanceof Error
          ? error.message
          : "The uploaded corpus details could not be loaded.";
      setCorporaError(message);
    } finally {
      setIsLoadingCorpora(false);
    }
  }

  // Loads persisted documents when the ingestion page first becomes interactive.
  useEffect(() => {
    const loadTimeoutId = window.setTimeout(() => void fetchCorpora(), 0);

    return () => window.clearTimeout(loadTimeoutId);
  }, []);

  // Locks page scrolling, focuses the dialog, and handles Escape while the modal is open.
  useEffect(() => {
    if (!isModalOpen) {
      return;
    }

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButtonRef.current?.focus();

    /**
     * Closes the modal when the Escape key is pressed.
     *
     * @param event - The keyboard event dispatched by the browser.
     * @returns Nothing. The modal state is updated for Escape only.
     */
    function handleEscape(event: KeyboardEvent): void {
      if (event.key === "Escape" && !isUploadPending) {
        setIsModalOpen(false);
      }
    }

    window.addEventListener("keydown", handleEscape);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleEscape);
    };
  }, [isModalOpen, isUploadPending]);

  // Automatically clears upload notices after they have been visible briefly.
  useEffect(() => {
    if (!notice) {
      return;
    }

    const timeoutId = window.setTimeout(() => setNotice(null), 6000);
    return () => window.clearTimeout(timeoutId);
  }, [notice]);

  /**
   * Opens the upload dialog and clears any previous result notice.
   *
   * @returns Nothing. The modal and notice state are updated.
   */
  function openUploadModal(): void {
    setNotice(null);
    setIsModalOpen(true);
  }

  /**
   * Closes the upload dialog without changing the corpus list.
   *
   * @returns Nothing. The modal state is updated.
   */
  function closeUploadModal(): void {
    // Keep the request visible until the backend finishes parsing and persistence.
    if (isUploadPending) {
      return;
    }

    setIsModalOpen(false);
  }

  /**
   * Refreshes the persisted inventory after a successful upload.
   *
   * @param filenames - Filenames accepted by the backend upload endpoint.
   * @param requestedName - Corpus name entered before upload.
   * @returns A promise resolved after the inventory refresh is requested.
   */
  async function handleUploadSuccess(filenames: string[], requestedName: string): Promise<void> {
    setIsModalOpen(false);
    setNotice({
      type: "success",
      message: `${filenames.length} ${
        filenames.length === 1 ? "document" : "documents"
      } uploaded and parsed for ${requestedName}.`,
    });
    await fetchCorpora();
  }

  /**
   * Closes the upload modal and exposes a useful failure notice.
   *
   * @param message - The upload error returned by the form.
   * @returns Nothing. The modal and notice states are updated.
   */
  function handleUploadFailure(message: string): void {
    setIsModalOpen(false);
    setNotice({ type: "error", message });
  }

  /**
   * Expands or collapses one corpus document list.
   *
   * @param corpusId - Stable identifier of the selected corpus.
   * @returns Nothing. The expanded corpus state is updated.
   */
  function toggleCorpus(corpusId: string): void {
    setExpandedCorpusId((currentId) => (currentId === corpusId ? null : corpusId));
  }

  /**
   * Shows or hides parsing details for every document in one corpus.
   *
   * @param corpusId - Stable identifier of the corpus whose summaries are toggled.
   * @param shouldShow - Whether parsing details should become visible.
   * @returns Nothing. The set of visible corpus summaries is updated.
   */
  function toggleParseSummary(corpusId: string, shouldShow: boolean): void {
    setParseSummaryCorpusIds((currentIds) => {
      const nextIds = new Set(currentIds);

      // Add or remove the corpus without mutating the previous React state value.
      if (shouldShow) {
        nextIds.add(corpusId);
      } else {
        nextIds.delete(corpusId);
      }

      return nextIds;
    });

    // Expand a collapsed corpus so newly enabled summaries are immediately visible.
    if (shouldShow) {
      setExpandedCorpusId(corpusId);
    }
  }

  return (
    <main
      className="min-h-screen bg-[var(--page-surface)] text-[var(--charcoal)]
        lg:grid lg:grid-cols-[240px_minmax(0,1fr)]"
    >
      <WorkbenchSidebar />

      {/* Main grid canvas contains the corpus inventory and its primary action. */}
      <WorkbenchGridCanvas className="px-5 py-8 sm:px-8 lg:px-12">
        <div className="mx-auto w-full max-w-5xl">
          <header className="mb-6 flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p
                className="mb-2 font-mono text-[10px] font-bold uppercase
                  tracking-[0.16em] text-[var(--muted-text)] lg:hidden"
              >
                RAG Playground / Documents
              </p>
              <h1 className="text-2xl font-bold tracking-[-0.03em]">Uploaded Corpora</h1>
              <p className="mt-1 text-sm text-[var(--tone-black)]">
                Manage knowledge bases and their uploaded documents.
              </p>
            </div>

            <div className="flex items-center gap-4">
              <span
                className="font-mono text-[10px] font-bold uppercase tracking-[0.14em]
                  text-[var(--muted-text)]"
              >
                {corpora.length} {corpora.length === 1 ? "Collection" : "Collections"}
              </span>
              <button
                className="inline-flex items-center gap-2 rounded bg-[var(--charcoal)]
                  px-4 py-2.5 text-sm font-semibold text-[var(--white)]
                  transition-colors hover:bg-[var(--primary-hover)]
                  focus-visible:outline-none focus-visible:ring-2
                  focus-visible:ring-[var(--charcoal)] focus-visible:ring-offset-2"
                onClick={openUploadModal}
                type="button"
              >
                <FiPlus aria-hidden="true" className="size-4" />
                Add New Corpus
              </button>
              <button
                aria-label="Refresh corpus list"
                className="rounded p-2 text-[var(--tone-black)] transition-colors
                  hover:bg-[var(--white)] hover:text-[var(--charcoal)]
                  focus-visible:outline-none focus-visible:ring-2
                  focus-visible:ring-[var(--charcoal)] disabled:cursor-not-allowed
                  disabled:opacity-50"
                disabled={isLoadingCorpora}
                onClick={() => void fetchCorpora()}
                title="Refresh corpus list"
                type="button"
              >
                <FiRefreshCw
                  aria-hidden="true"
                  className={`size-4 ${isLoadingCorpora ? "animate-spin" : ""}`}
                />
              </button>
            </div>
          </header>

          {/* Compact data table presents corpus metadata and expandable documents. */}
          <div
            className="overflow-hidden rounded border border-[var(--border-subtle)]
              bg-[var(--white)]"
          >
            <div
              className="grid grid-cols-[minmax(0,1fr)_9rem] border-b
                border-[var(--border-subtle)] bg-[var(--panel-surface)]
                sm:grid-cols-[minmax(0,1fr)_13rem]"
            >
              <span
                className="px-4 py-3 text-[10px] font-bold uppercase tracking-[0.12em]
                  text-[var(--tone-black)] sm:px-6"
              >
                Corpus / Document
              </span>
              <span
                className="px-4 py-3 text-[10px] font-bold uppercase tracking-[0.12em]
                  text-[var(--tone-black)] sm:px-6"
              >
                Uploaded
              </span>
            </div>

            {isLoadingCorpora && corpora.length === 0 ? (
              /* Loading state reserves the inventory space while persisted records arrive. */
              <div
                className="flex min-h-64 items-center justify-center px-6 py-12 text-sm
                  text-[var(--muted-text)]"
              >
                Loading uploaded documents…
              </div>
            ) : corporaError && corpora.length === 0 ? (
              /* Error state gives the user a direct way to retry the inventory request. */
              <div
                className="flex min-h-64 flex-col items-center justify-center px-6 py-12
                  text-center"
              >
                <h2 className="text-sm font-semibold">Couldn’t load corpora</h2>
                <p className="mt-1 max-w-sm text-sm leading-6 text-[var(--muted-text)]">
                  {corporaError}
                </p>
                <button
                  className="mt-5 text-sm font-semibold text-[var(--charcoal)] underline
                    decoration-[var(--muted-text)] underline-offset-4
                    hover:decoration-[var(--charcoal)] focus-visible:outline-none
                    focus-visible:ring-2 focus-visible:ring-[var(--charcoal)]
                    focus-visible:ring-offset-2"
                  onClick={() => void fetchCorpora()}
                  type="button"
                >
                  Try again
                </button>
              </div>
            ) : corpora.length === 0 ? (
              /* Empty state directs first-time users toward the upload workflow. */
              <div
                className="flex min-h-64 flex-col items-center justify-center px-6 py-12
                  text-center"
              >
                <span
                  className="flex size-12 items-center justify-center rounded border
                    border-[var(--border-subtle)] bg-[var(--panel-surface)]"
                >
                  <FiFolder aria-hidden="true" className="size-5 text-[var(--muted-text)]" />
                </span>
                <h2 className="mt-4 text-sm font-semibold">No corpora uploaded</h2>
                <p className="mt-1 max-w-sm text-sm leading-6 text-[var(--muted-text)]">
                  Add a corpus to create a knowledge base from your documents.
                </p>
                <button
                  className="mt-5 text-sm font-semibold text-[var(--charcoal)] underline
                    decoration-[var(--muted-text)] underline-offset-4
                    hover:decoration-[var(--charcoal)] focus-visible:outline-none
                    focus-visible:ring-2 focus-visible:ring-[var(--charcoal)]
                    focus-visible:ring-offset-2"
                  onClick={openUploadModal}
                  type="button"
                >
                  Add your first corpus
                </button>
              </div>
            ) : null}

            {corpora.map((corpus) => {
              const isExpanded = expandedCorpusId === corpus.id;
              const isParseSummaryVisible = parseSummaryCorpusIds.has(corpus.id);

              return (
                <article
                  className="border-b border-[var(--border-subtle)] last:border-b-0"
                  key={corpus.id}
                >
                  {/* Corpus controls keep expansion and parse visibility independent. */}
                  <div
                    className="grid grid-cols-1 transition-colors
                      hover:bg-[var(--landing-soft)]
                      sm:grid-cols-[minmax(0,1fr)_13rem]"
                  >
                    <div
                      className="flex min-w-0 flex-col gap-3 px-4 py-4
                        md:flex-row md:items-center sm:px-6"
                    >
                      <button
                        aria-expanded={isExpanded}
                        className="flex min-w-0 flex-1 items-center gap-3 text-left
                          focus-visible:outline-none focus-visible:ring-2
                          focus-visible:ring-[var(--charcoal)] focus-visible:ring-offset-2"
                        onClick={() => toggleCorpus(corpus.id)}
                        type="button"
                      >
                        <FiFolder
                          aria-hidden="true"
                          className="size-5 shrink-0 text-[var(--muted-text)]"
                        />
                        <span className="min-w-0">
                          <span className="flex items-center gap-2">
                            <span className="truncate text-sm font-bold">
                              {corpus.name}
                            </span>
                            {corpus.isNew ? (
                              <span
                                className="rounded-sm bg-[var(--badge-surface)] px-1.5
                                  py-0.5 font-mono text-[9px] font-bold uppercase
                                  tracking-wide text-[var(--subtle-text)]"
                              >
                                New
                              </span>
                            ) : null}
                          </span>
                          <span
                            className="mt-0.5 block text-[10px] font-medium uppercase
                              tracking-wide text-[var(--muted-text)]"
                          >
                            {corpus.documents.length}{" "}
                            {corpus.documents.length === 1 ? "document" : "documents"}
                          </span>
                        </span>
                        <FiChevronDown
                          aria-hidden="true"
                          className={`ml-auto size-4 shrink-0 text-[var(--muted-text)]
                            transition-transform ${isExpanded ? "rotate-180" : ""}`}
                        />
                      </button>

                      {/* Toggle reveals parsing telemetry without changing the expanded state. */}
                      <div className="flex shrink-0 items-center gap-2 md:ml-4">
                        <span
                          className="font-mono text-[9px] font-bold uppercase tracking-[0.1em]
                            text-[var(--muted-text)]"
                        >
                          Show parse summary
                        </span>
                        <button
                          aria-checked={isParseSummaryVisible}
                          aria-label={`Show parse summary for ${corpus.name}`}
                          className={`relative h-5 w-9 shrink-0 rounded-full border
                            transition-colors focus-visible:outline-none focus-visible:ring-2
                            focus-visible:ring-[var(--charcoal)] focus-visible:ring-offset-2
                            ${
                              isParseSummaryVisible
                                ? "border-[var(--charcoal)] bg-[var(--charcoal)]"
                                : `border-[var(--border-subtle)]
                                  bg-[var(--badge-surface)]`
                            }`}
                          onClick={() => {
                            toggleParseSummary(corpus.id, !isParseSummaryVisible);
                          }}
                          role="switch"
                          type="button"
                        >
                          <span
                            aria-hidden="true"
                            className={`absolute top-0.5 size-3.5 rounded-full
                              transition-[left,background-color]
                              ${
                                isParseSummaryVisible
                                  ? "left-[17px] bg-[var(--white)]"
                                  : "left-0.5 bg-[var(--muted-text)]"
                              }`}
                          />
                        </button>
                      </div>
                    </div>
                    <span
                      className="border-t border-[var(--border-soft)] px-4 pb-4 font-mono
                        text-[10px] text-[var(--tone-black)] sm:self-center sm:border-t-0
                        sm:px-6 sm:pb-0 sm:text-xs"
                    >
                      {formatTimestamp(corpus.createdAt)}
                    </span>
                  </div>

                  {isExpanded ? (
                    /* Expanded corpus details form a bounded, scrollable document ledger. */
                    <div
                      className="border-t border-[var(--border-soft)]
                        bg-[var(--page-surface)]"
                    >
                      {/* Sticky labels preserve column meaning while long corpora scroll. */}
                      <div
                        className="grid grid-cols-[2.25rem_minmax(0,1fr)_9rem] border-b
                          border-[var(--border-soft)] bg-[var(--panel-surface)]
                          sm:grid-cols-[2.75rem_minmax(0,1fr)_13rem]"
                      >
                        <span aria-hidden="true" />
                        <span
                          className="py-2 font-mono text-[9px] font-bold uppercase
                            tracking-[0.12em] text-[var(--muted-text)]"
                        >
                          Filename
                        </span>
                        <span
                          className="px-4 py-2 font-mono text-[9px] font-bold uppercase
                            tracking-[0.12em] text-[var(--muted-text)] sm:px-6"
                        >
                          Last modified
                        </span>
                      </div>

                      {/* Scroll region keeps every document accessible within the corpus row. */}
                      <div className="max-h-72 overflow-y-auto overscroll-contain">
                        {corpus.documents.map((document) => (
                          <div
                            className="grid grid-cols-[2.25rem_minmax(0,1fr)_9rem]
                              border-b border-[var(--border-soft)] bg-[var(--white)]
                              transition-colors last:border-b-0
                              hover:bg-[var(--landing-soft)]
                              sm:grid-cols-[2.75rem_minmax(0,1fr)_13rem]"
                            key={document.id}
                          >
                            <FiFileText
                              aria-hidden="true"
                              className="ml-4 mt-3.5 size-3.5 text-[var(--muted-text)]
                                sm:ml-6"
                            />
                            <div
                              className="min-w-0 py-3 pr-4 text-xs
                                text-[var(--tone-black)]"
                            >
                              <span className="block truncate" title={document.originalFilename}>
                                {document.originalFilename}
                              </span>
                              <span
                                className="block font-mono text-[9px] uppercase tracking-wide
                                  text-[var(--muted-text)]"
                              >
                                {document.mimeType ?? "Document"} ·{" "}
                                {formatFileSize(document.sizeBytes)}
                              </span>
                              {isParseSummaryVisible ? (
                                <DocumentParseDetails parse={document.parse} />
                              ) : null}
                            </div>
                            <span
                              className="self-center px-4 font-mono text-[10px] uppercase
                                text-[var(--muted-text)] sm:px-6"
                            >
                              {formatTimestamp(document.uploadedAt)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </article>
              );
            })}
          </div>
        </div>
      </WorkbenchGridCanvas>

      {/* Top-centered toast confirms the final upload outcome outside the modal. */}
      {notice ? (
        <Toast
          message={notice.message}
          onDismiss={() => setNotice(null)}
          type={notice.type}
        />
      ) : null}

      {/* Focused upload modal retains the existing form over a blurred workspace. */}
      {isModalOpen ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center
            bg-[var(--modal-backdrop)] p-4 backdrop-blur-sm sm:p-6"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              closeUploadModal();
            }
          }}
        >
          <section
            aria-labelledby="upload-modal-title"
            aria-modal="true"
            className="flex max-h-[calc(100dvh-2rem)] w-full max-w-4xl flex-col
              overflow-hidden rounded-lg border-2 border-[var(--charcoal)]
              bg-[var(--white)] sm:max-h-[calc(100dvh-3rem)]"
            role="dialog"
          >
            <header
              className="flex shrink-0 items-start justify-between gap-6 border-b
                border-[var(--border-soft)] p-5 sm:p-6"
            >
              <div>
                <h2
                  className="text-2xl font-bold tracking-[-0.03em] sm:text-3xl"
                  id="upload-modal-title"
                >
                  Upload Knowledge Assets
                </h2>
                <p className="mt-2 text-sm leading-6 text-[var(--tone-black)]">
                  Upload source files and create their reusable parsed text artifacts.
                </p>
              </div>
              <button
                aria-label="Close upload dialog"
                className="shrink-0 rounded-sm p-1 text-[var(--muted-text)]
                  transition-colors hover:bg-[var(--panel-surface)]
                  hover:text-[var(--charcoal)] focus-visible:outline-none
                  focus-visible:ring-2 focus-visible:ring-[var(--charcoal)]
                  disabled:cursor-not-allowed disabled:opacity-50"
                disabled={isUploadPending}
                onClick={closeUploadModal}
                ref={closeButtonRef}
                type="button"
              >
                <FiX aria-hidden="true" className="size-6" />
              </button>
            </header>

            <UploadForm
              onCancel={closeUploadModal}
              onFailure={handleUploadFailure}
              onSuccess={handleUploadSuccess}
              onUploadStateChange={setIsUploadPending}
            />
          </section>
        </div>
      ) : null}
    </main>
  );
}
