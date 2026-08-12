"use client";

import { ChangeEvent, FormEvent, useState } from "react";
import { FiFilePlus, FiFileText } from "react-icons/fi";
import apiClient, { isAxiosError } from "@/lib/axios";
import { parseUploadResponse } from "@/validation/ingestion";

type UploadFormProps = {
  onCancel: () => void;
  onFailure: (message: string) => void;
  onSuccess: (filenames: string[], corpusName: string) => void;
  onUploadStateChange: (isUploading: boolean) => void;
};

/**
 * Converts an unsuccessful upload request into a user-facing message.
 *
 * @param error - The failed request error returned by axios.
 * @returns A useful upload failure message.
 */
function readUploadErrorMessage(error: unknown): string {
  // Prefer the backend's structured error message when it is available.
  if (isAxiosError(error)) {
    const payload = error.response?.data as {
      detail?: { filename?: string; message?: string } | string;
    };

    if (typeof payload.detail === "object" && payload.detail?.message) {
      // Prefix document-specific failures with the filename supplied by the backend.
      if (payload.detail.filename) {
        return `${payload.detail.filename}: ${payload.detail.message}`;
      }

      return payload.detail.message;
    }

    // Surface plain string details from FastAPI validation responses.
    if (typeof payload.detail === "string") {
      return payload.detail;
    }

    return "The documents could not be uploaded. Check the files and try again.";
  }

  // Surface the controlled runtime-contract error produced by response validation.
  if (error instanceof Error) {
    return error.message;
  }

  return "The documents could not be uploaded. Check the files and try again.";
}

/**
 * Uploads selected documents and reports the completed outcome to its modal.
 *
 * @param props - Modal callbacks for cancel, failure, and successful filenames.
 * @returns The interactive multipart document upload form.
 */
export default function UploadForm({
  onCancel,
  onFailure,
  onSuccess,
  onUploadStateChange,
}: UploadFormProps) {
  // Indicates that the form is waiting for the backend upload response.
  const [isUploading, setIsUploading] = useState(false);

  // Lists the names selected in the browser file picker for display in the form.
  const [selectedFileNames, setSelectedFileNames] = useState<string[]>([]);

  /**
   * Records filenames selected through the browser file picker.
   *
   * @param event - The file input change event containing the current files.
   * @returns Nothing. The selected filename state is updated.
   */
  function handleFileChange(event: ChangeEvent<HTMLInputElement>): void {
    const files = Array.from(event.target.files ?? []);
    setSelectedFileNames(files.map((file) => file.name));
  }

  /**
   * Sends the form's multipart file fields to the backend.
   *
   * @param event - The browser form submission event.
   * @returns A promise that resolves after the parent receives the upload outcome.
   */
  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setIsUploading(true);
    onUploadStateChange(true);

    const formData = new FormData(event.currentTarget);
    const corpusName = String(formData.get("corpusName") ?? "").trim();

    try {
      const response = await apiClient.post<unknown>("/uploads", formData);
      const uploadResult = parseUploadResponse(response.data);
      onSuccess(uploadResult.filenames, corpusName);
    } catch (error) {
      onFailure(readUploadErrorMessage(error));
    } finally {
      setIsUploading(false);
      onUploadStateChange(false);
    }
  }

  return (
    <form
      className="flex min-h-0 flex-1 flex-col overflow-hidden"
      onSubmit={handleSubmit}
    >
      {/* File picker groups the corpus identity and its source documents. */}
      <div className="min-h-0 overflow-y-auto p-5 sm:p-6">
        {/* Required naming field identifies the corpus created by this upload. */}
        <div className="mb-4">
          <label
            className="block text-sm font-semibold text-[var(--charcoal)]"
            htmlFor="corpusName"
          >
            Corpus name <span className="font-normal text-[var(--muted-text)]">(required)</span>
          </label>
          <input
            autoComplete="off"
            className="mt-2 block w-full rounded border border-[var(--border-strong)]
              bg-[var(--white)] px-3 py-2.5 font-mono text-sm text-[var(--charcoal)]
              outline-none transition-colors placeholder:text-[var(--placeholder-text)]
              hover:border-[var(--muted-text)] focus:border-[var(--charcoal)]
              focus:ring-1 focus:ring-[var(--charcoal)]"
            disabled={isUploading}
            id="corpusName"
            maxLength={100}
            name="corpusName"
            placeholder="e.g. customer_support_docs"
            required
            type="text"
          />
          <p className="mt-1.5 text-xs leading-5 text-[var(--muted-text)]">
            Give this document collection a name you will recognize later.
          </p>
        </div>

        {/* Hidden native input is controlled through the visible document drop zone. */}
        <label
          className="block text-sm font-semibold text-[var(--charcoal)]"
          htmlFor="files"
        >
          Files
        </label>
        <input
          className="peer sr-only"
          id="files"
          disabled={isUploading}
          multiple
          name="files"
          onChange={handleFileChange}
          required
          type="file"
        />
        <label
          className="mt-2 flex min-h-32 cursor-pointer flex-col items-center justify-center
            rounded-lg border-2 border-dashed border-[var(--border-strong)]
            bg-[var(--white)] px-6 py-6 text-center transition-colors
            hover:border-[var(--charcoal)] hover:bg-[var(--page-surface)]
            peer-disabled:cursor-not-allowed peer-disabled:opacity-60
            peer-focus-visible:outline-none peer-focus-visible:ring-2
            peer-focus-visible:ring-[var(--charcoal)] peer-focus-visible:ring-offset-2 sm:px-8"
          htmlFor="files"
        >
          <span
            className="flex size-10 items-center justify-center rounded border
              border-[var(--border-subtle)] bg-[var(--panel-surface)]"
          >
            <FiFilePlus aria-hidden="true" className="size-5 text-[var(--muted-text)]" />
          </span>
          <span className="mt-3 text-base font-semibold text-[var(--charcoal)]">
            {selectedFileNames.length > 0
              ? `${selectedFileNames.length} ${
                  selectedFileNames.length === 1 ? "file" : "files"
                } selected`
              : "Choose files"}
          </span>
          <span className="mt-1 text-sm leading-6 text-[var(--muted-text)]">
            {selectedFileNames.length > 0
              ? "Choose files again to replace this selection."
              : "Click to browse. Multiple files are supported."}
          </span>
        </label>

        {/* Selected document names remain visible without expanding the upload dialog. */}
        {selectedFileNames.length > 0 ? (
          <section
            aria-label="Selected files"
            className="mt-4 overflow-hidden rounded border
              border-[var(--border-subtle)] bg-[var(--white)]"
          >
            {/* List header separates the batch summary from the scrollable file rows. */}
            <header
              className="flex items-center justify-between border-b
                border-[var(--border-subtle)] bg-[var(--panel-surface)] px-3 py-2"
            >
              <h3
                className="font-mono text-[10px] font-bold uppercase tracking-[0.08em]
                  text-[var(--subtle-text)]"
              >
                Selected files
              </h3>
              <span className="font-mono text-xs text-[var(--muted-text)]">
                {selectedFileNames.length}
              </span>
            </header>

            {/* Filename rows scroll independently when the selected batch is long. */}
            <ol className="max-h-36 overflow-y-auto overscroll-contain">
              {/* Each row keeps its full filename available while protecting the layout. */}
              {selectedFileNames.map((fileName, index) => (
                <li
                  className="flex min-w-0 items-center gap-3 border-b
                    border-[var(--border-subtle)] px-3 py-2.5 last:border-b-0"
                  key={`${fileName}-${index}`}
                >
                  <FiFileText
                    aria-hidden="true"
                    className="size-4 shrink-0 text-[var(--muted-text)]"
                  />
                  <span
                    className="min-w-0 flex-1 truncate font-mono text-xs
                      text-[var(--charcoal)]"
                    title={fileName}
                  >
                    {fileName}
                  </span>
                  <span
                    aria-hidden="true"
                    className="font-mono text-[10px] tabular-nums
                      text-[var(--muted-text)]"
                  >
                    {String(index + 1).padStart(2, "0")}
                  </span>
                </li>
              ))}
            </ol>
          </section>
        ) : null}
      </div>

      {/* Modal footer makes cancellation and upload confirmation explicit. */}
      <footer
        className="flex shrink-0 flex-col gap-3 border-t border-[var(--border-subtle)]
          bg-[var(--page-surface)] px-5 py-3 sm:flex-row sm:items-center
          sm:justify-between sm:px-6"
      >
        <p aria-live="polite" className="text-sm text-[var(--tone-black)]">
          {isUploading
            ? "Uploading and parsing each document. Keep this dialog open."
            : "Documents are parsed and saved when you confirm this step."}
        </p>
        <div className="flex justify-end gap-3">
          <button
            className="rounded px-5 py-2.5 text-sm font-semibold
              text-[var(--tone-black)] transition-colors
              hover:bg-[var(--hover-surface)] hover:text-[var(--charcoal)]
              focus-visible:outline-none focus-visible:ring-2
              focus-visible:ring-[var(--charcoal)] disabled:cursor-not-allowed
              disabled:opacity-50"
            disabled={isUploading}
            onClick={onCancel}
            type="button"
          >
            Cancel
          </button>
          <button
            className="rounded bg-[var(--charcoal)] px-5 py-2.5 text-sm font-semibold
              text-[var(--white)] transition-colors hover:bg-[var(--primary-hover)]
              focus-visible:outline-none focus-visible:ring-2
              focus-visible:ring-[var(--charcoal)] focus-visible:ring-offset-2
              disabled:cursor-not-allowed disabled:opacity-50"
            disabled={isUploading || selectedFileNames.length === 0}
            type="submit"
          >
            {isUploading ? "Uploading…" : "Upload"}
          </button>
        </div>
      </footer>
    </form>
  );
}
