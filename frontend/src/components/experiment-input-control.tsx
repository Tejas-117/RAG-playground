import { FiDatabase, FiFileText, FiMessageSquare } from "react-icons/fi";

/** Supported input shapes for an experiment run. */
export type ExperimentInputMode = "question" | "dataset";

/** Values and callbacks used to edit the experiment input. */
type ExperimentInputControlProps = {
  mode: ExperimentInputMode;
  question: string;
  datasetFile: File | null;
  onModeChange: (mode: ExperimentInputMode) => void;
  onQuestionChange: (question: string) => void;
  onDatasetChange: (dataset: File | null) => void;
};

/**
 * Renders the mutually exclusive question and evaluation-dataset inputs.
 *
 * @param props - Current input mode, values, and callbacks for updating them.
 * @returns An accessible run-input chooser with the matching input control.
 */
export default function ExperimentInputControl({
  mode,
  question,
  datasetFile,
  onModeChange,
  onQuestionChange,
  onDatasetChange,
}: ExperimentInputControlProps) {
  return (
    /* The run-input panel separates test data from reusable pipeline configuration. */
    <section
      aria-labelledby="run-input-heading"
      className="rounded border border-[var(--border-subtle)] bg-[var(--white)]"
    >
      {/* The panel introduction explains why the input changes available metrics. */}
      <div className="p-5 sm:p-6">
        <p
          className={`
            font-mono text-[10px] font-bold uppercase tracking-[0.14em]
            text-[var(--muted-text)]
          `}
        >
          Run input
        </p>
        <h2
          className="mt-2 text-lg font-bold tracking-[-0.03em]"
          id="run-input-heading"
        >
          Choose what to test
        </h2>
        <p className="mt-1 text-sm leading-6 text-[var(--tone-black)]">
          Ask one question for a quick check, or run an annotated dataset for a benchmark.
        </p>
      </div>

      {/* The mode switch uses radio semantics because exactly one input source is active. */}
      <div className="border-t border-[var(--border-soft)] p-5 sm:p-6">
        <fieldset>
          <legend className="sr-only">Experiment input type</legend>
          <div className="grid gap-3 sm:grid-cols-2">
            <label
              className={`
                flex cursor-pointer items-start gap-3 rounded border p-4
                transition-colors focus-within:ring-2
                focus-within:ring-[var(--charcoal)]
                ${
                  mode === "question"
                    ? "border-[var(--charcoal)] bg-[var(--panel-surface)]"
                    : "border-[var(--border-subtle)] bg-[var(--white)]"
                }
              `}
            >
              <input
                checked={mode === "question"}
                className="mt-1 accent-[var(--charcoal)]"
                name="experiment-input-mode"
                onChange={() => onModeChange("question")}
                type="radio"
                value="question"
              />
              <span className="flex min-w-0 gap-3">
                <FiMessageSquare aria-hidden="true" className="mt-0.5 size-4 shrink-0" />
                <span>
                  <span className="block text-sm font-bold">Single question</span>
                  <span className="mt-1 block text-xs leading-5 text-[var(--muted-text)]">
                    Inspect one answer with metrics that need no ground truth.
                  </span>
                </span>
              </span>
            </label>

            <label
              className={`
                flex cursor-pointer items-start gap-3 rounded border p-4
                transition-colors focus-within:ring-2
                focus-within:ring-[var(--charcoal)]
                ${
                  mode === "dataset"
                    ? "border-[var(--charcoal)] bg-[var(--panel-surface)]"
                    : "border-[var(--border-subtle)] bg-[var(--white)]"
                }
              `}
            >
              <input
                checked={mode === "dataset"}
                className="mt-1 accent-[var(--charcoal)]"
                name="experiment-input-mode"
                onChange={() => onModeChange("dataset")}
                type="radio"
                value="dataset"
              />
              <span className="flex min-w-0 gap-3">
                <FiDatabase aria-hidden="true" className="mt-0.5 size-4 shrink-0" />
                <span>
                  <span className="block text-sm font-bold">Evaluation dataset</span>
                  <span className="mt-1 block text-xs leading-5 text-[var(--muted-text)]">
                    Compare a batch using its reference answers and relevance labels.
                  </span>
                </span>
              </span>
            </label>
          </div>
        </fieldset>

        {/* Only the control belonging to the selected input mode remains visible. */}
        {mode === "question" ? (
          <label className="mt-5 block" htmlFor="experiment-question">
            <span
              className={`
                block font-mono text-[10px] font-bold uppercase
                tracking-[0.12em] text-[var(--muted-text)]
              `}
            >
              Question
              <span aria-hidden="true" className="ml-1 text-[var(--toast-error-text)]">
                *
              </span>
              <span className="sr-only"> (required)</span>
            </span>
            <textarea
              aria-required="true"
              className={`
                mt-2 min-h-28 w-full resize-y rounded border
                border-[var(--border-subtle)] bg-[var(--page-surface)]
                px-3 py-2.5 text-sm leading-6 text-[var(--charcoal)]
                outline-none transition-colors placeholder:text-[var(--placeholder-text)]
                focus:border-[var(--charcoal)] focus:ring-1
                focus:ring-[var(--charcoal)]
              `}
              id="experiment-question"
              onChange={(event) => onQuestionChange(event.target.value)}
              placeholder="Ask a question that the selected corpus can answer"
              required
              value={question}
            />
          </label>
        ) : (
          <label className="mt-5 block" htmlFor="evaluation-dataset">
            <span
              className={`
                block font-mono text-[10px] font-bold uppercase
                tracking-[0.12em] text-[var(--muted-text)]
              `}
            >
              Dataset file
              <span aria-hidden="true" className="ml-1 text-[var(--toast-error-text)]">
                *
              </span>
              <span className="sr-only"> (required)</span>
            </span>
            <span
              className={`
                mt-2 flex min-h-24 cursor-pointer items-center justify-between
                gap-4 rounded border border-dashed border-[var(--border-strong)]
                bg-[var(--page-surface)] px-4 py-4 transition-colors
                hover:border-[var(--charcoal)] focus-within:ring-2
                focus-within:ring-[var(--charcoal)]
              `}
            >
              <span className="flex min-w-0 items-center gap-3">
                <FiFileText aria-hidden="true" className="size-5 shrink-0" />
                <span className="min-w-0">
                  <span className="block truncate text-sm font-bold">
                    {datasetFile?.name ?? "Choose an evaluation dataset"}
                  </span>
                  <span className="mt-1 block text-xs leading-5 text-[var(--muted-text)]">
                    Include reference answers and relevant source labels for every metric.
                  </span>
                </span>
              </span>
              <span
                className={`
                  shrink-0 rounded border border-[var(--border-subtle)]
                  bg-[var(--white)] px-3 py-2 text-xs font-bold
                `}
              >
                Browse
              </span>
              <input
                className="sr-only"
                id="evaluation-dataset"
                onChange={(event) => onDatasetChange(event.target.files?.[0] ?? null)}
                required
                type="file"
              />
            </span>
          </label>
        )}
      </div>
    </section>
  );
}
