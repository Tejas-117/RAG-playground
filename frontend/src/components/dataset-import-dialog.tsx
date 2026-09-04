"use client";

import { useEffect, useRef, useState } from "react";
import { FiFileText, FiUploadCloud, FiX } from "react-icons/fi";
import { type CorpusOption } from "@/validation/corpora";
import { importDataset } from "@/lib/dataset-api";
import { isAxiosError } from "@/lib/axios";
import {
  DATASET_FILE_MAX_SIZE_BYTES,
  DATASET_NAME_MAX_LENGTH,
  type DatasetDetail,
  parseDatasetApiError,
} from "@/validation/datasets";

/** Human-readable file ceiling shown beside the JSON picker. */
const DATASET_FILE_MAX_SIZE_LABEL = "30 MB";

/** A compact valid payload that explains the import contract in place. */
const DATASET_FORMAT_EXAMPLE = `{
  "examples": [
    {
      "question": "How is authentication configured?",
      "reference_answer": "Authentication is configured through...",
      "relevant_documents": ["authentication.pdf"]
    }
  ]
}`;

type DatasetImportDialogProps = {
  corpora: CorpusOption[];
  onClose: () => void;
  onImported: (dataset: DatasetDetail) => void;
};

/**
 * Presents and submits the multipart evaluation-dataset import form.
 *
 * @param props - Available corpora plus close and successful-import callbacks.
 * @returns A native modal dialog containing dataset metadata and JSON selection.
 */
export default function DatasetImportDialog({
  corpora,
  onClose,
  onImported,
}: DatasetImportDialogProps) {
  // Stores the native dialog used for focus trapping and top-layer presentation.
  const dialogRef = useRef<HTMLDialogElement | null>(null);

  // Stores the user-facing immutable dataset name.
  const [name, setName] = useState("");

  // Stores the corpus used to resolve document filenames during import.
  const [corpusId, setCorpusId] = useState(corpora[0]?.id ?? "");

  // Stores the JSON source selected by the user.
  const [file, setFile] = useState<File | null>(null);

  // Stores a local validation or safe backend import failure.
  const [error, setError] = useState<string | null>(null);

  // Prevents closing or submitting the dialog twice while import is active.
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Opens the native modal once and restores focus when the component unmounts.
  useEffect(() => {
    const dialog = dialogRef.current;
    const previousActiveElement = document.activeElement;

    // Native modal presentation provides focus trapping and an accessible backdrop.
    if (dialog && !dialog.open) {
      dialog.showModal();
    }

    return () => {
      // Return keyboard users to the control that opened the import dialog.
      if (previousActiveElement instanceof HTMLElement) {
        previousActiveElement.focus();
      }
    };
  }, []);

  /**
   * Requests dialog closure when no import request is active.
   *
   * @returns Nothing. The parent removes the dialog from the page.
   */
  function closeDialog(): void {
    // An in-flight upload remains visible so its eventual outcome is not hidden.
    if (!isSubmitting) {
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
    closeDialog();
  }

  /**
   * Closes the dialog when the pointer lands on its empty backdrop.
   *
   * @param event - Pointer event received by the native dialog element.
   * @returns Nothing. Backdrop clicks request closure when submission is idle.
   */
  function handleBackdropClick(
    event: React.MouseEvent<HTMLDialogElement>,
  ): void {
    // Native dialog backdrop clicks target the outer dialog rather than its shell.
    if (event.target === event.currentTarget) {
      closeDialog();
    }
  }

  /**
   * Validates and stores a newly selected JSON file.
   *
   * @param event - Change event emitted by the hidden file input.
   * @returns Nothing. File and local error state are updated in place.
   */
  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>): void {
    const selectedFile = event.target.files?.[0] ?? null;
    setError(null);

    // Clearing the native picker should also clear the visible selected-file state.
    if (!selectedFile) {
      setFile(null);
      return;
    }

    // Mirror the backend extension rule so an invalid choice receives immediate feedback.
    if (!selectedFile.name.toLocaleLowerCase().endsWith(".json")) {
      setFile(null);
      setError("Choose a JSON file ending in .json.");
      event.target.value = "";
      return;
    }

    // Mirror the backend byte ceiling while preserving backend authority over validation.
    if (selectedFile.size > DATASET_FILE_MAX_SIZE_BYTES) {
      setFile(null);
      setError(`Choose a JSON file no larger than ${DATASET_FILE_MAX_SIZE_LABEL}.`);
      event.target.value = "";
      return;
    }

    setFile(selectedFile);
  }

  /**
   * Imports one locally validated dataset through the multipart API.
   *
   * @param event - Form submission event from the import action.
   * @returns A promise resolved after success or safe failure state is shown.
   */
  async function handleSubmit(
    event: React.FormEvent<HTMLFormElement>,
  ): Promise<void> {
    event.preventDefault();
    setError(null);
    const normalizedName = name.trim();

    // Required metadata and source input must exist before contacting FastAPI.
    if (!normalizedName || !corpusId || !file) {
      setError("Enter a name, select a corpus, and choose a JSON file.");
      return;
    }

    setIsSubmitting(true);

    try {
      const dataset = await importDataset({
        name: normalizedName,
        corpusId,
        file,
      });
      onImported(dataset);
    } catch (requestError) {
      // Prefer safe structured backend messages over generic transport wording.
      const apiMessage = isAxiosError(requestError)
        ? parseDatasetApiError(requestError.response?.data)
        : null;
      setError(
        apiMessage ??
          (requestError instanceof Error
            ? requestError.message
            : "The evaluation dataset could not be imported."),
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    /* The native dialog keeps import focused while retaining the workbench as context. */
    <dialog
      aria-labelledby="dataset-import-title"
      className="dataset-modal"
      onCancel={handleCancel}
      onClick={handleBackdropClick}
      ref={dialogRef}
    >
      {/* The bordered shell follows the workbench surface hierarchy without soft cards. */}
      <section
        className="dataset-modal-shell grid w-[calc(100vw-2rem)] max-w-5xl
          overflow-hidden border-2 border-[var(--charcoal)] bg-white lg:grid-cols-2"
      >
        {/* The form pane owns editable import metadata and the primary action. */}
        <form
          className="flex min-h-0 flex-col border-[var(--border-strong)] lg:border-r"
          onSubmit={handleSubmit}
        >
          {/* The dialog header names the immutable artifact being created. */}
          <header
            className="flex items-start justify-between gap-5 border-b
              border-[var(--border-subtle)] p-5 sm:p-6"
          >
            <div>
              <p
                className="font-mono text-[10px] font-bold uppercase tracking-[0.14em]
                  text-[var(--tone-black)]"
              >
                Dataset import
              </p>
              <h2
                className="mt-2 text-2xl font-semibold tracking-[-0.03em]"
                id="dataset-import-title"
              >
                Register evaluation questions.
              </h2>
            </div>

            <button
              aria-label="Close dataset import"
              className="rounded-sm p-1 text-[var(--muted-text)] hover:bg-[var(--panel-surface)]
                hover:text-[var(--charcoal)] focus-visible:outline-2
                focus-visible:outline-offset-2 focus-visible:outline-[var(--charcoal)]
                disabled:cursor-not-allowed disabled:opacity-50"
              disabled={isSubmitting}
              onClick={closeDialog}
              type="button"
            >
              <FiX aria-hidden="true" className="size-5" />
            </button>
          </header>

          {/* The input stack captures only values required by the backend contract. */}
          <div className="flex-1 space-y-5 overflow-y-auto p-5 sm:p-6">
            <label className="block" htmlFor="dataset-name">
              <span
                className="font-mono text-[10px] font-bold uppercase tracking-[0.12em]
                  text-[var(--tone-black)]"
              >
                Dataset name
              </span>
              <input
                autoFocus
                className="mt-2 w-full rounded-sm border border-[var(--border-strong)]
                  bg-white px-3 py-2.5 font-mono text-sm outline-none
                  focus:border-[var(--charcoal)] focus:ring-1 focus:ring-[var(--charcoal)]"
                disabled={isSubmitting}
                id="dataset-name"
                maxLength={DATASET_NAME_MAX_LENGTH}
                onChange={(event) => setName(event.target.value)}
                placeholder="Customer support benchmark"
                required
                type="text"
                value={name}
              />
            </label>

            <label className="block" htmlFor="dataset-corpus">
              <span
                className="font-mono text-[10px] font-bold uppercase tracking-[0.12em]
                  text-[var(--tone-black)]"
              >
                Corpus
              </span>
              <select
                className="mt-2 w-full rounded-sm border border-[var(--border-strong)]
                  bg-white px-3 py-2.5 font-mono text-sm outline-none
                  focus:border-[var(--charcoal)] focus:ring-1 focus:ring-[var(--charcoal)]"
                disabled={isSubmitting}
                id="dataset-corpus"
                onChange={(event) => setCorpusId(event.target.value)}
                required
                value={corpusId}
              >
                {/* Every option maps display names to stable corpus identities. */}
                {corpora.map((corpus) => (
                  <option key={corpus.id} value={corpus.id}>
                    {corpus.name}
                  </option>
                ))}
              </select>
            </label>

            <label
              className="block cursor-pointer border border-dashed
                border-[var(--border-strong)] bg-[var(--page-surface)] p-5
                transition-colors hover:border-[var(--charcoal)]"
              htmlFor="dataset-file"
            >
              {/* The file summary makes the selected immutable source unambiguous. */}
              <span className="flex items-start gap-3">
                {file ? (
                  <FiFileText
                    aria-hidden="true"
                    className="mt-0.5 size-5 shrink-0"
                  />
                ) : (
                  <FiUploadCloud
                    aria-hidden="true"
                    className="mt-0.5 size-5 shrink-0"
                  />
                )}
                <span>
                  <span className="block text-sm font-semibold">
                    {file?.name ?? "Choose evaluation JSON"}
                  </span>
                  <span className="mt-1 block text-xs text-[var(--muted-text)]">
                    UTF-8 JSON · Maximum {DATASET_FILE_MAX_SIZE_LABEL}
                  </span>
                </span>
              </span>
              <input
                accept=".json,application/json"
                className="sr-only"
                disabled={isSubmitting}
                id="dataset-file"
                onChange={handleFileChange}
                required
                type="file"
              />
            </label>

            {/* Validation and request failures remain next to the affected form. */}
            {error ? (
              <p
                className="border border-[var(--toast-error-border)]
                  bg-[var(--toast-error-surface)] px-3 py-2 text-sm
                  text-[var(--toast-error-text)]"
                role="alert"
              >
                {error}
              </p>
            ) : null}
          </div>

          {/* The action bar keeps cancellation separate from the irreversible import. */}
          <footer
            className="flex justify-end gap-3 border-t border-[var(--border-subtle)]
              p-5 sm:p-6"
          >
            <button
              className="rounded-sm border border-[var(--border-strong)] bg-white px-4 py-2.5
                text-sm font-semibold hover:bg-[var(--panel-surface)]
                focus-visible:outline-2 focus-visible:outline-offset-2
                focus-visible:outline-[var(--charcoal)] disabled:opacity-50"
              disabled={isSubmitting}
              onClick={closeDialog}
              type="button"
            >
              Cancel
            </button>
            <button
              className="rounded-sm bg-[var(--charcoal)] px-4 py-2.5 text-sm font-semibold
                text-white hover:bg-[var(--primary-hover)] focus-visible:outline-2
                focus-visible:outline-offset-2 focus-visible:outline-[var(--charcoal)]
                disabled:cursor-not-allowed disabled:opacity-50"
              disabled={isSubmitting}
              type="submit"
            >
              {isSubmitting ? "Importing…" : "Import dataset"}
            </button>
          </footer>
        </form>

        {/* The contract pane explains the mapping performed during import. */}
        <aside className="hidden bg-[var(--panel-surface)] p-6 lg:block">
          <p
            className="font-mono text-[10px] font-bold uppercase tracking-[0.14em]
              text-[var(--tone-black)]"
          >
            Accepted structure
          </p>
          <p className="mt-3 max-w-md text-sm leading-6 text-[var(--tone-black)]">
            Document names are matched against files in the selected corpus. Unknown or
            ambiguous names are skipped and reported after import.
          </p>
          <pre
            className="mt-6 overflow-x-auto border border-[var(--border-strong)] bg-white
              p-4 font-mono text-xs leading-6 text-[var(--charcoal)]"
          >
            <code>{DATASET_FORMAT_EXAMPLE}</code>
          </pre>
          <p className="mt-4 text-xs leading-5 text-[var(--muted-text)]">
            Questions are required. Reference answers and relevant documents are optional.
          </p>
        </aside>
      </section>
    </dialog>
  );
}
