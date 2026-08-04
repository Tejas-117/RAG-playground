"use client";

import axios from "axios";
import { ChangeEvent, FormEvent, useState } from "react";
import { FiFilePlus } from "react-icons/fi";

type UploadResponse = {
  message: string;
  filenames: string[];
};

type UploadFormProps = {
  onCancel: () => void;
  onFailure: (message: string) => void;
  onSuccess: (filenames: string[], corpusName: string) => void;
};

const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

/**
 * Reads the upload API origin configured for the browser bundle.
 *
 * @returns The API base URL loaded by Next.js from the frontend environment file.
 */
function getApiBaseUrl(): string {
  // Fail with a clear setup message when the public API URL was not configured.
  if (!apiBaseUrl) {
    throw new Error("NEXT_PUBLIC_API_BASE_URL is not configured.");
  }

  return apiBaseUrl;
}

/**
 * Converts an unsuccessful upload request into a user-facing message.
 *
 * @param error - The failed request error returned by axios.
 * @returns A useful upload failure message.
 */
function readUploadErrorMessage(error: unknown): string {
  // Prefer the backend's structured error message when it is available.
  if (axios.isAxiosError(error)) {
    const payload = error.response?.data as {
      detail?: { message?: string } | string;
    };

    if (typeof payload.detail === "object" && payload.detail?.message) {
      return payload.detail.message;
    }

    // Surface plain string details from FastAPI validation responses.
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
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

    const formData = new FormData(event.currentTarget);
    const corpusName = String(formData.get("corpusName") ?? "").trim();

    try {
      const response = await axios.post<UploadResponse>(
        `${getApiBaseUrl()}/uploads`,
        formData,
      );
      onSuccess(response.data.filenames, corpusName);
    } catch (error) {
      onFailure(readUploadErrorMessage(error));
    } finally {
      setIsUploading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      {/* File picker gives the existing upload interaction the Stitch workbench treatment. */}
      <div className="p-6 sm:p-8">
        {/* Required naming field identifies the corpus created by this upload. */}
        <div className="mb-6">
          <label
            className="block text-sm font-semibold text-[var(--charcoal)]"
            htmlFor="corpusName"
          >
            Corpus name <span className="font-normal text-[var(--muted-text)]">(required)</span>
          </label>
          <input
            autoComplete="off"
            className="mt-2 block w-full rounded border border-[var(--border-strong)] bg-[var(--white)] px-3 py-2.5 font-mono text-sm text-[var(--charcoal)] outline-none transition-colors placeholder:text-[var(--placeholder-text)] hover:border-[var(--muted-text)] focus:border-[var(--charcoal)] focus:ring-1 focus:ring-[var(--charcoal)]"
            id="corpusName"
            maxLength={100}
            name="corpusName"
            placeholder="e.g. customer_support_docs"
            required
            type="text"
          />
          <p className="mt-2 text-xs leading-5 text-[var(--muted-text)]">
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
          multiple
          name="files"
          onChange={handleFileChange}
          required
          type="file"
        />
        <label
          className="mt-3 flex min-h-56 cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-[var(--border-strong)] bg-[var(--white)] px-6 py-10 text-center transition-colors hover:border-[var(--charcoal)] hover:bg-[var(--page-surface)] peer-focus-visible:outline-none peer-focus-visible:ring-2 peer-focus-visible:ring-[var(--charcoal)] peer-focus-visible:ring-offset-2 sm:px-8"
          htmlFor="files"
        >
          <span className="flex size-12 items-center justify-center rounded border border-[var(--border-subtle)] bg-[var(--panel-surface)]">
            <FiFilePlus aria-hidden="true" className="size-6 text-[var(--muted-text)]" />
          </span>
          <span className="mt-4 text-lg font-semibold text-[var(--charcoal)]">
            {selectedFileNames.length > 0
              ? `${selectedFileNames.length} ${selectedFileNames.length === 1 ? "file" : "files"} selected`
              : "Choose files"}
          </span>
          <span className="mt-1 max-w-2xl break-words text-sm leading-6 text-[var(--muted-text)]">
            {selectedFileNames.length > 0
              ? selectedFileNames.join(", ")
              : "Click to browse. Multiple files are supported."}
          </span>
        </label>
      </div>

      {/* Modal footer makes cancellation and upload confirmation explicit. */}
      <footer className="flex flex-col gap-4 border-t border-[var(--border-subtle)] bg-[var(--page-surface)] px-6 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-8">
        <p className="text-sm text-[var(--tone-black)]">
          Files are uploaded only when you confirm this step.
        </p>
        <div className="flex justify-end gap-3">
          <button
            className="rounded px-5 py-2.5 text-sm font-semibold text-[var(--tone-black)] transition-colors hover:bg-[var(--hover-surface)] hover:text-[var(--charcoal)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--charcoal)] disabled:cursor-not-allowed disabled:opacity-50"
            disabled={isUploading}
            onClick={onCancel}
            type="button"
          >
            Cancel
          </button>
          <button
            className="rounded bg-[var(--charcoal)] px-5 py-2.5 text-sm font-semibold text-[var(--white)] transition-colors hover:bg-[var(--primary-hover)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--charcoal)] focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
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
