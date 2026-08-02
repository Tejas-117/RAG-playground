import Link from "next/link";
import { FiArrowRight } from "react-icons/fi";

const experimentStages = [
  { label: "Source", value: "your documents" },
  { label: "Chunk", value: "size · overlap · strategy" },
  { label: "Embed", value: "provider · model · dimensions" },
  { label: "Retrieve", value: "embedding · metric · top k" },
  { label: "Generate", value: "provider · model · prompt" },
  { label: "Evaluate", value: "quality · latency · cost" },
];

const capabilities = [
  {
    title: "Configure the pipeline",
    description:
      "Change chunking, embeddings, retrieval, reranking, and generation without losing sight of the full setup.",
  },
  {
    title: "Run with provenance",
    description:
      "Keep every input, retrieved chunk, score, response, timing, and error attached to the run that produced it.",
  },
  {
    title: "Evaluate separately",
    description:
      "Measure retrieval and answer quality again without repeating ingestion or generation.",
  },
  {
    title: "Compare what changed",
    description:
      "Place compatible runs side by side and trace a better result back to the configuration behind it.",
  },
];

/**
 * Presents the RAG Playground product overview and entry point to ingestion.
 *
 * @returns The server-rendered landing page for the root route.
 */
export default function Home() {
  return (
    <main className="landing-page overflow-hidden bg-[var(--white)] text-[var(--charcoal)]">
      {/* Minimal navigation frames the product and keeps the first action visible. */}
      <nav
        aria-label="Primary navigation"
        className="landing-reveal mx-auto flex w-full max-w-7xl items-center justify-between px-5 py-5 sm:px-8 lg:px-12"
      >
        <Link
          className="group inline-flex items-center gap-2.5 rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--charcoal)] focus-visible:ring-offset-4"
          href="/"
        >
          <span className="grid size-7 grid-cols-2 gap-0.5 rounded-sm bg-[var(--charcoal)] p-1.5">
            <span className="rounded-[1px] bg-[var(--white)]" />
            <span className="rounded-[1px] bg-[var(--white)] opacity-45" />
            <span className="rounded-[1px] bg-[var(--white)] opacity-45" />
            <span className="rounded-[1px] bg-[var(--white)]" />
          </span>
          <span className="text-sm font-semibold tracking-[-0.02em]">
            RAG Playground
          </span>
        </Link>

        <Link
          className="landing-secondary-action rounded-full border px-4 py-2 text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--charcoal)] focus-visible:ring-offset-4"
          href="/ingestion"
        >
          Add documents
        </Link>
      </nav>

      {/* The hero states the product thesis beside a visual trace of a complete RAG run. */}
      <section className="mx-auto grid w-full max-w-7xl gap-16 px-5 pb-24 pt-20 sm:px-8 sm:pb-32 sm:pt-24 lg:grid-cols-[1.15fr_0.85fr] lg:items-end lg:gap-20 lg:px-12 lg:pb-36 lg:pt-24">
        <div className="landing-reveal landing-reveal-delay-1 max-w-3xl">
          <p className="font-mono text-[0.68rem] font-medium uppercase tracking-[0.2em] text-[var(--tone-black)]">
            A workbench for RAG decisions
          </p>
          <h1 className="mt-7 text-[clamp(3.4rem,8vw,7.4rem)] font-semibold leading-[0.87] tracking-[-0.075em]">
            Test the path
            <span className="block text-[var(--tone-black)]">
              to the answer.
            </span>
          </h1>
          <p className="mt-9 max-w-xl text-base leading-7 text-[var(--tone-black)] sm:text-lg sm:leading-8">
            Configure, run, evaluate, and compare retrieval-augmented
            generation experiments over your own documents—while preserving
            exactly how each answer was made.
          </p>
          <div className="mt-10 flex flex-col items-start gap-5 sm:flex-row sm:items-center">
            <Link
              className="landing-primary-action inline-flex items-center gap-3 rounded-full bg-[var(--charcoal)] px-5 py-3 text-sm font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--charcoal)] focus-visible:ring-offset-4"
              href="/ingestion"
            >
              Start with documents
              <FiArrowRight aria-hidden="true" className="size-4" />
            </Link>
            <a
              className="landing-secondary-action inline-flex items-center rounded-full border px-5 py-3 text-sm font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--charcoal)] focus-visible:ring-offset-4"
              href="#workbench"
            >
              See what the workbench tracks
            </a>
          </div>
        </div>

        <div className="landing-reveal landing-reveal-delay-2 relative lg:-translate-y-8 lg:pb-1">
          <div className="mb-4 flex items-center justify-between font-mono text-[0.65rem] uppercase tracking-[0.16em] text-[var(--tone-black)]">
            <span>Experiment trace</span>
            <span>One run · full context</span>
          </div>
          <div className="relative overflow-hidden rounded-xl border border-[var(--light-gray)] bg-[var(--white)]">
            <div className="pipeline-line absolute bottom-0 left-[2.1rem] top-0 w-px bg-[var(--light-gray)] sm:left-[2.6rem]" />
            {experimentStages.map((stage, index) => (
              <div
                className="relative grid grid-cols-[2.25rem_1fr] gap-3 border-b border-[var(--light-gray)] px-4 py-5 last:border-b-0 sm:grid-cols-[3rem_1fr] sm:px-5"
                key={stage.label}
              >
                <span className="relative z-10 flex size-6 items-center justify-center rounded-full border border-[var(--charcoal)] bg-[var(--white)] font-mono text-[0.6rem] font-semibold">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <div className="flex min-w-0 flex-col gap-1 sm:flex-row sm:items-baseline sm:justify-between sm:gap-4">
                  <span className="text-sm font-semibold">{stage.label}</span>
                  <span className="font-mono text-[0.67rem] leading-5 text-[var(--tone-black)] sm:text-right">
                    {stage.value}
                  </span>
                </div>
              </div>
            ))}
          </div>
          <p className="mt-4 max-w-sm font-mono text-[0.65rem] leading-5 text-[var(--tone-black)]">
            Every run stores an immutable snapshot, so yesterday&apos;s result
            never depends on today&apos;s defaults.
          </p>
        </div>
      </section>

      {/* Capability notes explain the four outcomes the workbench is being built to support. */}
      <section
        className="border-y border-[var(--light-gray)] bg-[var(--landing-soft)]"
        id="workbench"
      >
        <div className="mx-auto w-full max-w-7xl px-5 py-20 sm:px-8 sm:py-24 lg:px-12">
          <div className="grid gap-7 border-b border-[var(--light-gray)] pb-12 md:grid-cols-[0.65fr_1.35fr] md:items-end">
            <p className="font-mono text-[0.68rem] font-medium uppercase tracking-[0.2em] text-[var(--tone-black)]">
              What we&apos;re building
            </p>
            <h2 className="max-w-3xl text-3xl font-semibold leading-tight tracking-[-0.04em] sm:text-5xl">
              An honest record of what changed, what came back, and what
              worked.
            </h2>
          </div>

          <div className="grid md:grid-cols-2">
            {capabilities.map((capability, index) => (
              <article
                className="border-b border-[var(--light-gray)] py-8 md:min-h-52 md:px-8 md:py-9 md:odd:border-r md:odd:pl-0 md:even:pr-0"
                key={capability.title}
              >
                <span className="font-mono text-[0.65rem] text-[var(--tone-black)]">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <h3 className="mt-7 text-xl font-semibold tracking-[-0.03em]">
                  {capability.title}
                </h3>
                <p className="mt-3 max-w-md text-sm leading-6 text-[var(--tone-black)]">
                  {capability.description}
                </p>
              </article>
            ))}
          </div>
        </div>
      </section>

      {/* Closing call to action returns the explanation to the first concrete workflow step. */}
      <section className="mx-auto flex w-full max-w-7xl flex-col gap-10 px-5 py-20 sm:px-8 sm:py-28 md:flex-row md:items-end md:justify-between lg:px-12">
        <div className="max-w-2xl">
          <p className="font-mono text-[0.68rem] font-medium uppercase tracking-[0.2em] text-[var(--tone-black)]">
            First input
          </p>
          <h2 className="mt-5 text-4xl font-semibold leading-[1.05] tracking-[-0.05em] sm:text-6xl">
            Bring the documents.
            <span className="block text-[var(--tone-black)]">
              We&apos;ll keep the trail.
            </span>
          </h2>
        </div>
        <Link
          className="landing-primary-action inline-flex w-fit shrink-0 items-center gap-3 rounded-full bg-[var(--charcoal)] px-5 py-3 text-sm font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--charcoal)] focus-visible:ring-offset-4"
          href="/ingestion"
        >
          Add documents
          <FiArrowRight aria-hidden="true" className="size-4" />
        </Link>
      </section>

      {/* Footer identifies the current single-user, local-first project scope. */}
      <footer className="border-t border-[var(--light-gray)]">
        <div className="mx-auto flex w-full max-w-7xl flex-col gap-2 px-5 py-6 font-mono text-[0.65rem] uppercase tracking-[0.12em] text-[var(--tone-black)] sm:flex-row sm:items-center sm:justify-between sm:px-8 lg:px-12">
          <span>RAG Playground</span>
          <span>Single-user · local-first experiments</span>
        </div>
      </footer>
    </main>
  );
}
