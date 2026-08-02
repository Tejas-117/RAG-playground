"use client";

import { useEffect, useRef, useState } from "react";
import {
  FiAlertCircle,
  FiCheckCircle,
  FiChevronDown,
  FiFileText,
  FiFolder,
  FiPlus,
  FiX,
} from "react-icons/fi";
import UploadForm from "@/components/upload-form";
import WorkbenchSidebar from "@/components/workbench-sidebar";

type Corpus = {
  id: string;
  name: string;
  versionCount: number;
  modifiedAt: string;
  documents: string[];
  isNew?: boolean;
};

type Notice = {
  type: "success" | "error";
  message: string;
};

/**
 * Resolves an explicit corpus name or derives one from the first filename.
 *
 * @param requestedName - Optional corpus name entered by the user.
 * @param filenames - Filenames returned by the upload endpoint.
 * @returns The requested name or a normalized filename-based fallback.
 */
function createCorpusName(
  requestedName: string,
  filenames: string[],
): string {
  const trimmedName = requestedName.trim();

  if (trimmedName) {
    return trimmedName;
  }

  const firstFilename = filenames[0] ?? "uploaded_corpus";
  const nameWithoutExtension = firstFilename.replace(/\.[^/.]+$/, "");
  const normalizedName = nameWithoutExtension
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");

  return normalizedName || "uploaded_corpus";
}

/**
 * Formats the current browser time for the corpus table.
 *
 * @returns A local timestamp in YYYY-MM-DD HH:mm format.
 */
function createModifiedTimestamp(): string {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  const hours = String(now.getHours()).padStart(2, "0");
  const minutes = String(now.getMinutes()).padStart(2, "0");

  return `${year}-${month}-${day} ${hours}:${minutes}`;
}

/**
 * Renders the corpus workspace, upload modal, and upload notices.
 *
 * @returns The interactive ingestion workspace.
 */
export default function IngestionWorkbench() {
  // Stores the corpora currently displayed in the workbench table.
  const [corpora, setCorpora] = useState<Corpus[]>([]);
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
   * Inserts a successfully uploaded corpus at the top of the table.
   *
   * @param filenames - Filenames accepted by the backend upload endpoint.
   * @param requestedName - Optional corpus name entered before upload.
   * @returns Nothing. The corpus, modal, and notice states are updated.
   */
  function handleUploadSuccess(
    filenames: string[],
    requestedName: string,
  ): void {
    const newCorpus: Corpus = {
      id: `uploaded-${Date.now()}`,
      name: createCorpusName(requestedName, filenames),
      versionCount: 1,
      modifiedAt: createModifiedTimestamp(),
      documents: filenames,
      isNew: true,
    };

    setCorpora((currentCorpora) => [newCorpus, ...currentCorpora]);
    setExpandedCorpusId(newCorpus.id);
    setIsModalOpen(false);
    setNotice({
      type: "success",
      message: `${filenames.length} ${filenames.length === 1 ? "document" : "documents"} uploaded to ${newCorpus.name}.`,
    });
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
    <main className="min-h-screen bg-[#f9f9f8] text-[#181616] lg:grid lg:grid-cols-[240px_minmax(0,1fr)]">
      <WorkbenchSidebar />

      {/* Main grid canvas contains the corpus inventory and its primary action. */}
      <section
        className="ingestion-grid-background min-w-0 px-5 py-8 sm:px-8 lg:px-12"
        ref={gridBackgroundRef}
      >
        <div className="mx-auto w-full max-w-5xl">
          <header className="mb-6 flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="mb-2 font-mono text-[10px] font-bold uppercase tracking-[0.16em] text-[#7f7575] lg:hidden">
                RAG Playground / Documents
              </p>
              <h1 className="text-2xl font-bold tracking-[-0.03em]">
                Uploaded Corpora
              </h1>
              <p className="mt-1 text-sm text-[#5d5c5c]">
                Manage knowledge bases and their uploaded documents.
              </p>
            </div>

            <div className="flex items-center gap-4">
              <span className="font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-[#7f7575]">
                {corpora.length} {corpora.length === 1 ? "Collection" : "Collections"}
              </span>
              <button
                className="inline-flex items-center gap-2 rounded bg-[#181616] px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-[#353232] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#181616] focus-visible:ring-offset-2"
                onClick={openUploadModal}
                type="button"
              >
                <FiPlus aria-hidden="true" className="size-4" />
                Add New Corpus
              </button>
            </div>
          </header>

          {/* Compact data table presents corpus metadata and expandable documents. */}
          <div className="overflow-hidden rounded border border-[#dadad9] bg-white">
            <div className="grid grid-cols-[minmax(0,1fr)_9rem] border-b border-[#dadad9] bg-[#f3f4f3] sm:grid-cols-[minmax(0,1fr)_13rem]">
              <span className="px-4 py-3 text-[10px] font-bold uppercase tracking-[0.12em] text-[#5d5c5c] sm:px-6">
                Corpus / Document
              </span>
              <span className="px-4 py-3 text-[10px] font-bold uppercase tracking-[0.12em] text-[#5d5c5c] sm:px-6">
                Last Modified
              </span>
            </div>

            {corpora.length === 0 ? (
              /* Empty state directs first-time users toward the upload workflow. */
              <div className="flex min-h-64 flex-col items-center justify-center px-6 py-12 text-center">
                <span className="flex size-12 items-center justify-center rounded border border-[#dadad9] bg-[#f3f4f3]">
                  <FiFolder aria-hidden="true" className="size-5 text-[#7f7575]" />
                </span>
                <h2 className="mt-4 text-sm font-semibold">No corpora uploaded</h2>
                <p className="mt-1 max-w-sm text-sm leading-6 text-[#7f7575]">
                  Add a corpus to create a knowledge base from your documents.
                </p>
                <button
                  className="mt-5 text-sm font-semibold text-[#181616] underline decoration-[#7f7575] underline-offset-4 hover:decoration-[#181616] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#181616] focus-visible:ring-offset-2"
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
                  className="border-b border-[#dadad9] last:border-b-0"
                  key={corpus.id}
                >
                  <button
                    aria-expanded={isExpanded}
                    className="grid w-full grid-cols-[minmax(0,1fr)_9rem] text-left transition-colors hover:bg-[#f7f7f6] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[#181616] sm:grid-cols-[minmax(0,1fr)_13rem]"
                    onClick={() => toggleCorpus(corpus.id)}
                    type="button"
                  >
                    <span className="flex min-w-0 items-center gap-3 px-4 py-4 sm:px-6">
                      <FiFolder aria-hidden="true" className="size-5 shrink-0 text-[#7f7575]" />
                      <span className="min-w-0">
                        <span className="flex items-center gap-2">
                          <span className="truncate text-sm font-bold">
                            {corpus.name}
                          </span>
                          {corpus.isNew ? (
                            <span className="rounded-sm bg-[#e2dfdf] px-1.5 py-0.5 font-mono text-[9px] font-bold uppercase tracking-wide text-[#474647]">
                              New
                            </span>
                          ) : null}
                        </span>
                        <span className="mt-0.5 block text-[10px] font-medium uppercase tracking-wide text-[#7f7575]">
                          {corpus.versionCount} {corpus.versionCount === 1 ? "version" : "versions"} · {corpus.documents.length} {corpus.documents.length === 1 ? "document" : "documents"}
                        </span>
                      </span>
                      <FiChevronDown
                        aria-hidden="true"
                        className={`ml-auto size-4 shrink-0 text-[#7f7575] transition-transform ${isExpanded ? "rotate-180" : ""}`}
                      />
                    </span>
                    <span className="self-center px-4 font-mono text-[10px] text-[#5d5c5c] sm:px-6 sm:text-xs">
                      {corpus.modifiedAt}
                    </span>
                  </button>

                  {isExpanded ? (
                    /* Expanded corpus details list the documents returned by the upload API. */
                    <div className="border-t border-[#eeeeed] bg-[#f9f9f8]">
                      {corpus.documents.map((documentName) => (
                        <div
                          className="grid grid-cols-[minmax(0,1fr)_9rem] border-b border-[#eeeeed] last:border-b-0 sm:grid-cols-[minmax(0,1fr)_13rem]"
                          key={documentName}
                        >
                          <span className="flex min-w-0 items-center gap-2 py-2 pl-12 pr-4 text-xs text-[#5d5c5c] sm:pl-16">
                            <FiFileText aria-hidden="true" className="size-3.5 shrink-0 text-[#7f7575]" />
                            <span className="truncate">{documentName}</span>
                          </span>
                          <span className="self-center px-4 font-mono text-[10px] uppercase text-[#7f7575] sm:px-6">
                            Indexed
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

      {/* Top-centered notice confirms the final upload outcome outside the modal. */}
      {notice ? (
        <div
          aria-live="polite"
          className={`fixed left-1/2 top-5 z-[70] flex w-[calc(100%-2rem)] max-w-xl -translate-x-1/2 items-start gap-3 rounded border px-4 py-3 text-sm shadow-lg ${
            notice.type === "success"
              ? "border-[#b8c8ba] bg-[#f3faf4] text-[#244b2a]"
              : "border-[#e2a9a4] bg-[#fff5f4] text-[#93000a]"
          }`}
          role={notice.type === "error" ? "alert" : "status"}
        >
          {notice.type === "success" ? (
            <FiCheckCircle aria-hidden="true" className="mt-0.5 size-4 shrink-0" />
          ) : (
            <FiAlertCircle aria-hidden="true" className="mt-0.5 size-4 shrink-0" />
          )}
          <span className="min-w-0 flex-1 font-medium">{notice.message}</span>
          <button
            aria-label="Dismiss notification"
            className="rounded-sm p-0.5 hover:bg-black/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-current"
            onClick={() => setNotice(null)}
            type="button"
          >
            <FiX aria-hidden="true" className="size-4" />
          </button>
        </div>
      ) : null}

      {/* Focused upload modal retains the existing form over a blurred workspace. */}
      {isModalOpen ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center overflow-y-auto bg-black/45 px-4 py-8 backdrop-blur-sm"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              closeUploadModal();
            }
          }}
        >
          <section
            aria-labelledby="upload-modal-title"
            aria-modal="true"
            className="w-full max-w-5xl overflow-hidden rounded-lg border-2 border-[#181616] bg-white"
            role="dialog"
          >
            <header className="flex items-start justify-between gap-6 border-b border-[#eeeeed] p-6 sm:p-8">
              <div>
                <h2
                  className="text-2xl font-bold tracking-[-0.03em] sm:text-3xl"
                  id="upload-modal-title"
                >
                  Upload Knowledge Assets
                </h2>
                <p className="mt-2 text-sm leading-6 text-[#5d5c5c]">
                  Select files to index for your RAG pipeline. Multiple files are supported.
                </p>
              </div>
              <button
                aria-label="Close upload dialog"
                className="shrink-0 rounded-sm p-1 text-[#7f7575] transition-colors hover:bg-[#f3f4f3] hover:text-[#181616] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#181616]"
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
