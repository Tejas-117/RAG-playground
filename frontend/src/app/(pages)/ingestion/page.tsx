import UploadForm from "@/components/upload-form";

/**
 * Presents the document-ingestion page and its interactive upload form.
 */
export default function IngestionPage() {
  return (
    <main className="flex items-center justify-center bg-[var(--charcoal)] px-6 py-16 text-[var(--white)]">
      <section className="w-full max-w-xl rounded-2xl border border-[var(--light-gray)] bg-[var(--tone-black)] p-8 shadow-xl">
        <p className="mb-2 text-sm font-semibold uppercase tracking-widest text-[var(--golden-yellow)]">
          RAG Playground
        </p>
        <h1 className="text-3xl font-semibold">Upload your documents</h1>
        <p className="mt-3 text-sm leading-6 text-white/70">
          Select one or more files to start an ingestion experiment.
        </p>

        <UploadForm />
      </section>
    </main>
  );
}
