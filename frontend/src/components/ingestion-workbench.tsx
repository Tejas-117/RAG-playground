"use client";

import axios from "axios";
import { useEffect, useRef, useState } from "react";
import {
  FiChevronDown,
  FiFileText,
  FiFolder,
  FiPlus,
  FiRefreshCw,
  FiX,
} from "react-icons/fi";
import Toast, { ToastType } from "@/components/toast";
import UploadForm from "@/components/upload-form";
import WorkbenchSidebar from "@/components/workbench-sidebar";

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
};

type CorporaResponse = {
  corpora: Array<{
    id: string;
    name: string;
    created_at: string;
    updated_at: string;
    documents: Array<{
      id: string;
      original_filename: string;
      mime_type: string | null;
      size_bytes: number;
      uploaded_at: string;
    }>;
  }>;
};

type Notice = {
  type: ToastType;
  message: string;
};

const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

/**
 * Resolves the API origin configured for the browser bundle.
 *
 * @returns The configured API base URL.
 */
function getApiBaseUrl(): string {
  if (!apiBaseUrl) {
    throw new Error("NEXT_PUBLIC_API_BASE_URL is not configured.");
  }

  return apiBaseUrl;
}

/**
 * Converts the backend response into the camelCase model used by the UI.
 *
 * @param response - The response payload returned by GET /corpora/.
 * @returns Corpus records sorted newest first with documents sorted by upload date.
 */
function normalizeCorpora(response: CorporaResponse): Corpus[] {
  return response.corpora
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

  // Controls whether the upload dialog is visible.
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Holds the latest success or failure message shown to the user.
  const [notice, setNotice] = useState<Notice | null>(null);

  // Keeps the grid canvas available to the pointer-tracking effect without causing renders.
  const gridBackgroundRef = useRef<HTMLElement>(null);

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
      const response = await axios.get<CorporaResponse>(
        `${getApiBaseUrl()}/corpora/`,
      );
      setCorpora(normalizeCorpora(response.data));
    } catch (error) {
      const message = axios.isAxiosError(error)
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

  // Subscribes to pointer movement so the grid glow follows the cursor.
  useEffect(() => {
    const gridBackground = gridBackgroundRef.current;
    const hasFinePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;

    if (gridBackground === null || !hasFinePointer) {
      return;
    }

    // Capturing the narrowed element in a dedicated constant keeps it non-null in callbacks.
    const gridElement: HTMLElement = gridBackground;

    let animationFrameId: number | null = null;
    let nextPointerPosition: { x: number; y: number } | null = null;

    /**
     * Applies the latest pointer position to the grid glow on the next animation frame.
     *
     * @returns Nothing. The grid CSS variables are updated directly on the canvas.
     */
    function paintGridGlow(): void {
      animationFrameId = null;

      if (!nextPointerPosition) {
        return;
      }

      gridElement.style.setProperty(
        "--grid-pointer-x",
        `${nextPointerPosition.x}px`,
      );
      gridElement.style.setProperty(
        "--grid-pointer-y",
        `${nextPointerPosition.y}px`,
      );
      nextPointerPosition = null;
    }

    /**
     * Tracks the cursor relative to the grid canvas for the local highlight.
     *
     * @param event - The pointer movement reported by the browser.
     * @returns Nothing. The latest pointer coordinates are queued for painting.
     */
    function handleGridPointerMove(event: PointerEvent): void {
      const bounds = gridElement.getBoundingClientRect();
      nextPointerPosition = {
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
      };
      gridElement.style.setProperty("--grid-glow-opacity", "1");

      if (animationFrameId === null) {
        animationFrameId = window.requestAnimationFrame(paintGridGlow);
      }
    }

    /**
     * Fades the grid highlight after the pointer leaves the canvas.
     *
     * @returns Nothing. The glow opacity is reset through a CSS variable.
     */
    function handleGridPointerLeave(): void {
      gridElement.style.setProperty("--grid-glow-opacity", "0");
    }

    gridElement.addEventListener("pointermove", handleGridPointerMove);
    gridElement.addEventListener("pointerleave", handleGridPointerLeave);

    return () => {
      gridElement.removeEventListener("pointermove", handleGridPointerMove);
      gridElement.removeEventListener("pointerleave", handleGridPointerLeave);

      if (animationFrameId !== null) {
        window.cancelAnimationFrame(animationFrameId);
      }
    };
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
      if (event.key === "Escape") {
        setIsModalOpen(false);
      }
    }

    window.addEventListener("keydown", handleEscape);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleEscape);
    };
  }, [isModalOpen]);

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
    setIsModalOpen(false);
  }

  /**
   * Refreshes the persisted inventory after a successful upload.
   *
   * @param filenames - Filenames accepted by the backend upload endpoint.
   * @param requestedName - Corpus name entered before upload.
   * @returns A promise resolved after the inventory refresh is requested.
   */
  async function handleUploadSuccess(
    filenames: string[],
    requestedName: string,
  ): Promise<void> {
    setIsModalOpen(false);
    setNotice({
      type: "success",
      message: `${filenames.length} ${filenames.length === 1 ? "document" : "documents"} uploaded for ${requestedName}.`,
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
    setExpandedCorpusId((currentId) =>
      currentId === corpusId ? null : corpusId,
    );
  }

  return (
    <main className="min-h-screen bg-[var(--page-surface)] text-[var(--charcoal)] lg:grid lg:grid-cols-[240px_minmax(0,1fr)]">
      <WorkbenchSidebar />

      {/* Main grid canvas contains the corpus inventory and its primary action. */}
      <section
        className="ingestion-grid-background min-w-0 px-5 py-8 sm:px-8 lg:px-12"
        ref={gridBackgroundRef}
      >
        <div className="mx-auto w-full max-w-5xl">
          <header className="mb-6 flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="mb-2 font-mono text-[10px] font-bold uppercase tracking-[0.16em] text-[var(--muted-text)] lg:hidden">
                RAG Playground / Documents
              </p>
              <h1 className="text-2xl font-bold tracking-[-0.03em]">
                Uploaded Corpora
              </h1>
              <p className="mt-1 text-sm text-[var(--tone-black)]">
                Manage knowledge bases and their uploaded documents.
              </p>
            </div>

            <div className="flex items-center gap-4">
              <span className="font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-[var(--muted-text)]">
                {corpora.length} {corpora.length === 1 ? "Collection" : "Collections"}
              </span>
              <button
                className="inline-flex items-center gap-2 rounded bg-[var(--charcoal)] px-4 py-2.5 text-sm font-semibold text-[var(--white)] transition-colors hover:bg-[var(--primary-hover)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--charcoal)] focus-visible:ring-offset-2"
                onClick={openUploadModal}
                type="button"
              >
                <FiPlus aria-hidden="true" className="size-4" />
                Add New Corpus
              </button>
              <button
                aria-label="Refresh corpus list"
                className="rounded p-2 text-[var(--tone-black)] transition-colors hover:bg-[var(--white)] hover:text-[var(--charcoal)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--charcoal)] disabled:cursor-not-allowed disabled:opacity-50"
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
          <div className="overflow-hidden rounded border border-[var(--border-subtle)] bg-[var(--white)]">
            <div className="grid grid-cols-[minmax(0,1fr)_9rem] border-b border-[var(--border-subtle)] bg-[var(--panel-surface)] sm:grid-cols-[minmax(0,1fr)_13rem]">
              <span className="px-4 py-3 text-[10px] font-bold uppercase tracking-[0.12em] text-[var(--tone-black)] sm:px-6">
                Corpus / Document
              </span>
              <span className="px-4 py-3 text-[10px] font-bold uppercase tracking-[0.12em] text-[var(--tone-black)] sm:px-6">
                Uploaded
              </span>
            </div>

            {isLoadingCorpora && corpora.length === 0 ? (
              /* Loading state reserves the inventory space while persisted records arrive. */
              <div className="flex min-h-64 items-center justify-center px-6 py-12 text-sm text-[var(--muted-text)]">
                Loading uploaded documents…
              </div>
            ) : corporaError && corpora.length === 0 ? (
              /* Error state gives the user a direct way to retry the inventory request. */
              <div className="flex min-h-64 flex-col items-center justify-center px-6 py-12 text-center">
                <h2 className="text-sm font-semibold">Couldn’t load corpora</h2>
                <p className="mt-1 max-w-sm text-sm leading-6 text-[var(--muted-text)]">
                  {corporaError}
                </p>
                <button
                  className="mt-5 text-sm font-semibold text-[var(--charcoal)] underline decoration-[var(--muted-text)] underline-offset-4 hover:decoration-[var(--charcoal)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--charcoal)] focus-visible:ring-offset-2"
                  onClick={() => void fetchCorpora()}
                  type="button"
                >
                  Try again
                </button>
              </div>
            ) : corpora.length === 0 ? (
              /* Empty state directs first-time users toward the upload workflow. */
              <div className="flex min-h-64 flex-col items-center justify-center px-6 py-12 text-center">
                <span className="flex size-12 items-center justify-center rounded border border-[var(--border-subtle)] bg-[var(--panel-surface)]">
                  <FiFolder aria-hidden="true" className="size-5 text-[var(--muted-text)]" />
                </span>
                <h2 className="mt-4 text-sm font-semibold">No corpora uploaded</h2>
                <p className="mt-1 max-w-sm text-sm leading-6 text-[var(--muted-text)]">
                  Add a corpus to create a knowledge base from your documents.
                </p>
                <button
                  className="mt-5 text-sm font-semibold text-[var(--charcoal)] underline decoration-[var(--muted-text)] underline-offset-4 hover:decoration-[var(--charcoal)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--charcoal)] focus-visible:ring-offset-2"
                  onClick={openUploadModal}
                  type="button"
                >
                  Add your first corpus
                </button>
              </div>
            ) : null}

            {corpora.map((corpus) => {
              const isExpanded = expandedCorpusId === corpus.id;
              return (
                <article
                  className="border-b border-[var(--border-subtle)] last:border-b-0"
                  key={corpus.id}
                >
                  <button
                    aria-expanded={isExpanded}
                    className="grid w-full grid-cols-[minmax(0,1fr)_9rem] text-left transition-colors hover:bg-[var(--landing-soft)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[var(--charcoal)] sm:grid-cols-[minmax(0,1fr)_13rem]"
                    onClick={() => toggleCorpus(corpus.id)}
                    type="button"
                  >
                    <span className="flex min-w-0 items-center gap-3 px-4 py-4 sm:px-6">
                      <FiFolder aria-hidden="true" className="size-5 shrink-0 text-[var(--muted-text)]" />
                      <span className="min-w-0">
                        <span className="flex items-center gap-2">
                          <span className="truncate text-sm font-bold">
                            {corpus.name}
                          </span>
                          {corpus.isNew ? (
                            <span className="rounded-sm bg-[var(--badge-surface)] px-1.5 py-0.5 font-mono text-[9px] font-bold uppercase tracking-wide text-[var(--subtle-text)]">
                              New
                            </span>
                          ) : null}
                        </span>
                        <span className="mt-0.5 block text-[10px] font-medium uppercase tracking-wide text-[var(--muted-text)]">
                          {corpus.documents.length} {corpus.documents.length === 1 ? "document" : "documents"}
                        </span>
                      </span>
                      <FiChevronDown
                        aria-hidden="true"
                        className={`ml-auto size-4 shrink-0 text-[var(--muted-text)] transition-transform ${isExpanded ? "rotate-180" : ""}`}
                      />
                    </span>
                    <span className="self-center px-4 font-mono text-[10px] text-[var(--tone-black)] sm:px-6 sm:text-xs">
                      {formatTimestamp(corpus.createdAt)}
                    </span>
                  </button>

                  {isExpanded ? (
                    /* Expanded corpus details list the documents returned by the upload API. */
                    <div className="border-t border-[var(--border-soft)] bg-[var(--page-surface)]">
                      {corpus.documents.map((document) => (
                        <div
                          className="grid grid-cols-[minmax(0,1fr)_9rem] border-b border-[var(--border-soft)] last:border-b-0 sm:grid-cols-[minmax(0,1fr)_13rem]"
                          key={document.id}
                        >
                          <span className="flex min-w-0 items-center gap-2 py-2 pl-12 pr-4 text-xs text-[var(--tone-black)] sm:pl-16">
                            <FiFileText aria-hidden="true" className="size-3.5 shrink-0 text-[var(--muted-text)]" />
                            <span className="min-w-0 truncate">
                              <span className="block truncate">{document.originalFilename}</span>
                              <span className="block font-mono text-[9px] uppercase tracking-wide text-[var(--muted-text)]">
                                {document.mimeType ?? "Document"} · {formatFileSize(document.sizeBytes)}
                              </span>
                            </span>
                          </span>
                          <span className="self-center px-4 font-mono text-[10px] uppercase text-[var(--muted-text)] sm:px-6">
                            {formatTimestamp(document.uploadedAt)}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : null}
                </article>
              );
            })}
          </div>
        </div>
      </section>

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
          className="fixed inset-0 z-50 flex items-center justify-center overflow-y-auto bg-[var(--modal-backdrop)] px-4 py-8 backdrop-blur-sm"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              closeUploadModal();
            }
          }}
        >
          <section
            aria-labelledby="upload-modal-title"
            aria-modal="true"
            className="w-full max-w-5xl overflow-hidden rounded-lg border-2 border-[var(--charcoal)] bg-[var(--white)]"
            role="dialog"
          >
            <header className="flex items-start justify-between gap-6 border-b border-[var(--border-soft)] p-6 sm:p-8">
              <div>
                <h2
                  className="text-2xl font-bold tracking-[-0.03em] sm:text-3xl"
                  id="upload-modal-title"
                >
                  Upload Knowledge Assets
                </h2>
                <p className="mt-2 text-sm leading-6 text-[var(--tone-black)]">
                  Select files to index for your RAG pipeline. Multiple files are supported.
                </p>
              </div>
              <button
                aria-label="Close upload dialog"
                className="shrink-0 rounded-sm p-1 text-[var(--muted-text)] transition-colors hover:bg-[var(--panel-surface)] hover:text-[var(--charcoal)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--charcoal)]"
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
            />
          </section>
        </div>
      ) : null}
    </main>
  );
}
