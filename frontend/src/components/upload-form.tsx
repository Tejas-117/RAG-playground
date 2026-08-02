"use client";

import { ChangeEvent, FormEvent, useState } from "react";
import { FiFilePlus } from "react-icons/fi";

type UploadResponse = {
  filenames: string[];
};

type UploadFormProps = {
  onCancel: () => void;
  onFailure: (message: string) => void;
  onSuccess: (filenames: string[], corpusName: string) => void;
};

const apiBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

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
    formData.delete("corpusName");

    try {
      const response = await fetch(`${apiBaseUrl}/upload/`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("The documents could not be uploaded. Check the files and try again.");
      }

      const result = (await response.json()) as UploadResponse;
      onSuccess(result.filenames, corpusName);
    } catch (error) {
      onFailure(
        error instanceof Error
          ? error.message
          : "The documents could not be uploaded. Try again.",
      );
    } finally {
      setIsUploading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      {/* File picker gives the existing upload interaction the Stitch workbench treatment. */}
      <div className="p-6 sm:p-8">
        {/* Optional naming field lets users override the filename-derived corpus name. */}
        <div className="mb-6">
          <label
            className="block text-sm font-semibold text-[#181616]"
            htmlFor="corpusName"
          >
            Corpus name <span className="font-normal text-[#7f7575]">(optional)</span>
          </label>
          <input
            autoComplete="off"
            className="mt-2 block w-full rounded border border-[#d0c4c4] bg-white px-3 py-2.5 font-mono text-sm text-[#181616] outline-none transition-colors placeholder:text-[#9b9393] hover:border-[#7f7575] focus:border-[#181616] focus:ring-1 focus:ring-[#181616]"
            id="corpusName"
            maxLength={100}
            name="corpusName"
            placeholder="e.g. customer_support_docs"
            type="text"
          />
          <p className="mt-2 text-xs leading-5 text-[#7f7575]">
            Leave this empty to generate a name from the first uploaded document.
          </p>
        </div>

        {/* Hidden native input is controlled through the visible document drop zone. */}
        <label
          className="block text-sm font-semibold text-[#181616]"
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
          className="mt-3 flex min-h-56 cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-[#d0c4c4] bg-white px-6 py-10 text-center transition-colors hover:border-[#181616] hover:bg-[#f9f9f8] peer-focus-visible:outline-none peer-focus-visible:ring-2 peer-focus-visible:ring-[#181616] peer-focus-visible:ring-offset-2 sm:px-8"
          htmlFor="files"
        >
          <span className="flex size-12 items-center justify-center rounded border border-[#dadad9] bg-[#f3f4f3]">
            <FiFilePlus aria-hidden="true" className="size-6 text-[#7f7575]" />
          </span>
          <span className="mt-4 text-lg font-semibold text-[#181616]">
            {selectedFileNames.length > 0
              ? `${selectedFileNames.length} ${selectedFileNames.length === 1 ? "file" : "files"} selected`
              : "Choose files"}
          </span>
          <span className="mt-1 max-w-2xl break-words text-sm leading-6 text-[#7f7575]">
            {selectedFileNames.length > 0
              ? selectedFileNames.join(", ")
              : "Click to browse. Multiple files are supported."}
          </span>
        </label>
      </div>

      {/* Modal footer makes cancellation and upload confirmation explicit. */}
      <footer className="flex flex-col gap-4 border-t border-[#dadad9] bg-[#f9f9f8] px-6 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-8">
        <p className="text-sm text-[#5d5c5c]">
          Files are uploaded only when you confirm this step.
        </p>
        <div className="flex justify-end gap-3">
          <button
            className="rounded px-5 py-2.5 text-sm font-semibold text-[#5d5c5c] transition-colors hover:bg-[#e8e8e7] hover:text-[#181616] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#181616] disabled:cursor-not-allowed disabled:opacity-50"
            disabled={isUploading}
            onClick={onCancel}
            type="button"
          >
            Cancel
          </button>
          <button
            className="rounded bg-[#181616] px-5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-[#353232] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#181616] focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            disabled={isUploading || selectedFileNames.length === 0}
            type="submit"
          >
            {isUploading ? "Uploading…" : "Upload and Index"}
          </button>
        </div>
      </footer>
    </form>
  );
}
