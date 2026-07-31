"use client";

import { FormEvent, useState } from "react";

type UploadResponse = {
  filenames: string[];
};

const apiBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

/**
 * Uploads selected documents and reports the filenames accepted by FastAPI.
 */
export default function UploadForm() {
  const [message, setMessage] = useState("");
  const [isUploading, setIsUploading] = useState(false);

  /**
   * Sends the form's multipart file fields to the backend.
   *
   * @param event - The browser form submission event.
   * @returns A promise that resolves after the request and UI update complete.
   */
  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setIsUploading(true);
    setMessage("");

    const form = event.currentTarget;
    const formData = new FormData(form);

    try {
      const response = await fetch(`${apiBaseUrl}/upload/`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("The files could not be uploaded.");
      }

      const result = (await response.json()) as UploadResponse;
      setMessage(`Uploaded: ${result.filenames.join(", ")}`);
      form.reset();
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "The files could not be uploaded.",
      );
    } finally {
      setIsUploading(false);
    }
  }

  return (
    <form className="mt-8 space-y-5" onSubmit={handleSubmit}>
      <div>
        <label className="mb-2 block text-sm font-medium" htmlFor="files">
          Documents
        </label>
        <input
          className="block w-full rounded-lg border border-white/20 bg-[var(--charcoal)] px-4 py-3 text-sm file:mr-4 file:rounded-md file:border-0 file:bg-[var(--golden-yellow)] file:px-4 file:py-2 file:font-semibold file:text-[var(--charcoal)]"
          id="files"
          name="files"
          type="file"
          multiple
          required
        />
      </div>

      <button
        className="w-full rounded-lg bg-[var(--golden-yellow)] px-5 py-3 font-semibold text-[var(--charcoal)] transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
        type="submit"
        disabled={isUploading}
      >
        {isUploading ? "Uploading…" : "Upload files"}
      </button>

      <p className="min-h-6 text-sm text-white/80" aria-live="polite">
        {message}
      </p>
    </form>
  );
}
