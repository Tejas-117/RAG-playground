"use client";

import { useEffect, useRef, useState } from "react";
import { FiAlertCircle, FiCheck, FiPlay, FiRefreshCw, FiSearch } from "react-icons/fi";
import ExperimentInputControl, {
  type ExperimentInputMode,
} from "@/components/experiment-input-control";
import MetricCheckboxGroup from "@/components/metric-checkbox-group";
import NumberControl from "@/components/number-control";
import SelectControl from "@/components/select-control";
import Toast from "@/components/toast";
import WorkbenchGridCanvas from "@/components/workbench-grid-canvas";
import WorkbenchSidebar from "@/components/workbench-sidebar";
import apiClient, { isCancel } from "@/lib/axios";
import { type CorpusOption, parseCorpora } from "@/validation/corpora";
import {
  type PipelineOptions,
  parsePipelineOptions,
} from "@/validation/pipeline-options";

/** The ordered stages shown in the experiment pipeline rail. */
const experimentStages = [
  { label: "Source", detail: "Choose a corpus" },
  { label: "Chunk", detail: "Split the source" },
  { label: "Embed", detail: "Represent meaning" },
  { label: "Retrieve", detail: "Find context" },
  { label: "Generate", detail: "Write an answer" },
  { label: "Evaluate", detail: "Measure the result" },
];

/** Editable values initialized from the backend catalog's resolved defaults. */
type ExperimentConfiguration = {
  chunkingStrategy: string;
  chunkSizeTokens: string;
  chunkOverlapTokens: string;
  embeddingProvider: string;
  embeddingModel: string;
  distanceMetric: string;
  topK: string;
  generationProvider: string;
  generationModel: string;
  temperature: string;
  maxOutputTokens: string;
  retrievalMetrics: string[];
  answerMetrics: string[];
};

/**
 * Creates the first editable configuration from backend-supported defaults.
 *
 * @param options - Validated pipeline option catalog.
 * @returns A complete local form state containing stable backend identifiers.
 */
function createDefaultConfiguration(options: PipelineOptions): ExperimentConfiguration {
  // Use the first catalog entries only where the backend does not publish a separate default.
  const embeddingProvider = options.embedding.providers[0];
  const generationProvider = options.generation.providers[0];

  return {
    chunkingStrategy: options.chunking.strategies[0].value,
    chunkSizeTokens: String(options.chunking.chunk_size_tokens.default),
    chunkOverlapTokens: String(options.chunking.chunk_overlap_tokens.default),
    embeddingProvider: embeddingProvider.value,
    embeddingModel: embeddingProvider.models[0].value,
    distanceMetric: options.embedding.distance_metrics[0].value,
    topK: String(options.retrieval.top_k.default),
    generationProvider: generationProvider.value,
    generationModel: generationProvider.models[0].value,
    temperature: String(options.generation.temperature.default),
    maxOutputTokens: String(options.generation.max_output_tokens.default),
    retrievalMetrics: options.evaluation.retrieval_metrics
      .filter((metric) => metric.selected_by_default)
      .map((metric) => metric.value),
    answerMetrics: options.evaluation.answer_metrics
      .filter((metric) => metric.selected_by_default && !metric.requires_reference_answer)
      .map((metric) => metric.value),
  };
}

/**
 * Renders the experiment setup workspace from backend-owned option data.
 *
 * @returns The interactive experiment configuration page.
 */
export default function ExperimentWorkbench() {
  // Tracks the stage selected from the sticky pipeline rail.
  const [activeStage, setActiveStage] = useState("Source");

  // Stores the validated backend-owned configuration catalog.
  const [pipelineOptions, setPipelineOptions] = useState<PipelineOptions | null>(null);

  // Stores the editable configuration initialized from backend defaults.
  const [configuration, setConfiguration] = useState<ExperimentConfiguration | null>(null);

  // Selects whether this run evaluates one question or an annotated dataset.
  const [inputMode, setInputMode] = useState<ExperimentInputMode>("question");

  // Stores the ad hoc question used for a single-question run.
  const [question, setQuestion] = useState("");

  // Stores the local evaluation dataset selected for a batch run.
  const [datasetFile, setDatasetFile] = useState<File | null>(null);

  // Stores the persisted corpora available to the source stage.
  const [corpora, setCorpora] = useState<CorpusOption[]>([]);

  // Stores the stable identifier of the selected immutable corpus.
  const [selectedCorpusId, setSelectedCorpusId] = useState("");

  // Stores the visible corpus query used to filter the source picker.
  const [corpusSearch, setCorpusSearch] = useState("");

  // Controls whether matching corpus results are visible.
  const [isCorpusDropdownOpen, setIsCorpusDropdownOpen] = useState(false);

  // Indicates that catalog and corpus data are being requested.
  const [isLoading, setIsLoading] = useState(true);

  // Stores a readable loading or contract-validation failure.
  const [loadError, setLoadError] = useState<string | null>(null);

  // Stores the reason a run cannot start after the user presses the unavailable action.
  const [runNotice, setRunNotice] = useState<string | null>(null);

  // Increments when the user asks to retry both initial API requests.
  const [loadAttempt, setLoadAttempt] = useState(0);

  // Stores each configuration card so the stage rail can move focus to it.
  const stageSectionRefs = useRef<Record<string, HTMLElement | null>>({});

  // Loads the backend catalog and current corpus inventory whenever a retry is requested.
  useEffect(() => {
    const abortController = new AbortController();

    /**
     * Loads and validates data required to configure every visible stage.
     *
     * @returns A promise resolved after loading, error, and form state are updated.
     */
    async function loadExperimentData(): Promise<void> {
      setIsLoading(true);
      setLoadError(null);

      try {
        const [optionsResponse, corporaResponse] = await Promise.all([
          apiClient.get<unknown>("/pipeline/options", {
            signal: abortController.signal,
          }),
          apiClient.get<unknown>("/corpora/", {
            signal: abortController.signal,
          }),
        ]);
        const validatedOptions = parsePipelineOptions(optionsResponse.data);
        const validatedCorpora = parseCorpora(corporaResponse.data);

        setPipelineOptions(validatedOptions);
        setConfiguration(createDefaultConfiguration(validatedOptions));
        setCorpora(validatedCorpora);

        // Select the first persisted corpus only when the inventory is non-empty.
        if (validatedCorpora.length > 0) {
          setSelectedCorpusId(validatedCorpora[0].id);
          setCorpusSearch(validatedCorpora[0].name);
        } else {
          setSelectedCorpusId("");
          setCorpusSearch("");
        }
      } catch (error) {
        // Ignore cancellation during unmount and surface every actionable failure.
        if (isCancel(error)) {
          return;
        }

        setPipelineOptions(null);
        setConfiguration(null);
        setLoadError(
          error instanceof Error
            ? error.message
            : "The experiment configuration could not be loaded.",
        );
      } finally {
        // Avoid updating loading state after a cancelled request has unmounted.
        if (!abortController.signal.aborted) {
          setIsLoading(false);
        }
      }
    }

    void loadExperimentData();

    return () => {
      abortController.abort();
    };
  }, [loadAttempt]);

  // Limits the source picker to persisted corpus names matching the user's query.
  const filteredCorpora = corpora.filter((corpus) =>
    corpus.name.toLocaleLowerCase().includes(corpusSearch.trim().toLocaleLowerCase()),
  );

  // Calculates the portion of the vertical stage trace completed by the selected stage.
  const activeStageIndex = experimentStages.findIndex((stage) => stage.label === activeStage);
  const stageProgress = activeStageIndex / (experimentStages.length - 1);

  // Resolves provider-specific models so incompatible choices never appear together.
  const embeddingProvider = pipelineOptions?.embedding.providers.find(
    (provider) => provider.value === configuration?.embeddingProvider,
  );
  const generationProvider = pipelineOptions?.generation.providers.find(
    (provider) => provider.value === configuration?.generationProvider,
  );
  const generationModel = generationProvider?.models.find(
    (model) => model.value === configuration?.generationModel,
  );
  const selectedStrategy = pipelineOptions?.chunking.strategies.find(
    (strategy) => strategy.value === configuration?.chunkingStrategy,
  );

  // Question runs exclude metrics that require dataset annotations or reference answers.
  const availableAnswerMetrics =
    inputMode === "question"
      ? (pipelineOptions?.evaluation.answer_metrics.filter(
          (metric) => !metric.requires_reference_answer,
        ) ?? [])
      : (pipelineOptions?.evaluation.answer_metrics ?? []);

  // A run requires one corpus and a value for its selected input mode.
  const hasRunInput =
    inputMode === "question" ? question.trim().length > 0 : datasetFile !== null;

  // Indicates whether every required value for starting a run is present.
  const isRunReady = Boolean(configuration && selectedCorpusId && hasRunInput);

  // Describes the selected generation model's provider-advertised token limits.
  let generationModelHelperText: string | undefined;

  // Build capability guidance only after a generation model has been selected.
  if (generationModel) {
    const contextWindow = generationModel.capabilities.context_window_tokens.toLocaleString();
    const modelOutputLimit = generationModel.capabilities.max_output_tokens;
    const outputGuidance =
      modelOutputLimit === null
        ? "output must fit within that window"
        : `up to ${modelOutputLimit.toLocaleString()} output tokens`;

    generationModelHelperText = `${contextWindow} token context window; ${outputGuidance}.`;
  }

  /**
   * Selects, scrolls to, and focuses a configuration stage from the pipeline rail.
   *
   * @param stage - The pipeline stage to bring into view.
   * @returns Nothing. The matching configuration card receives focus after scrolling.
   */
  function focusStage(stage: string): void {
    setActiveStage(stage);

    window.requestAnimationFrame(() => {
      const stageSection = stageSectionRefs.current[stage];

      // Focus only stages currently rendered after successful data loading.
      if (stageSection) {
        stageSection.scrollIntoView({ behavior: "smooth", block: "start" });
        stageSection.focus({ preventScroll: true });
      }
    });
  }

  /**
   * Updates one scalar value in the local experiment configuration.
   *
   * @param field - Configuration field receiving the new value.
   * @param value - Stable option identifier or numeric input string.
   * @returns Nothing. Local form state is updated when available.
   */
  function updateConfiguration(field: keyof ExperimentConfiguration, value: string): void {
    setConfiguration((currentConfiguration) =>
      currentConfiguration ? { ...currentConfiguration, [field]: value } : currentConfiguration,
    );
  }

  /**
   * Updates one optional evaluation metric list in the local configuration.
   *
   * @param field - Retrieval or answer metric list receiving the new selection.
   * @param values - Stable metric identifiers in backend catalog order.
   * @returns Nothing. Local form state is updated when available.
   */
  function updateMetrics(
    field: "retrievalMetrics" | "answerMetrics",
    values: string[],
  ): void {
    setConfiguration((currentConfiguration) =>
      currentConfiguration
        ? { ...currentConfiguration, [field]: values }
        : currentConfiguration,
    );
  }

  /**
   * Changes the chunker and clears overlap when the strategy does not support it.
   *
   * @param strategyValue - Stable chunking strategy identifier.
   * @returns Nothing. Strategy and compatible overlap state are updated together.
   */
  function updateChunkingStrategy(strategyValue: string): void {
    const strategy = pipelineOptions?.chunking.strategies.find(
      (option) => option.value === strategyValue,
    );

    setConfiguration((currentConfiguration) =>
      currentConfiguration
        ? {
            ...currentConfiguration,
            chunkingStrategy: strategyValue,
            chunkOverlapTokens: strategy?.supports_overlap
              ? String(pipelineOptions?.chunking.chunk_overlap_tokens.default ?? 0)
              : "0",
          }
        : currentConfiguration,
    );
  }

  /**
   * Changes the embedding provider and selects its first compatible model.
   *
   * @param providerValue - Stable embedding provider identifier.
   * @returns Nothing. Provider and model state are updated atomically.
   */
  function updateEmbeddingProvider(providerValue: string): void {
    const provider = pipelineOptions?.embedding.providers.find(
      (option) => option.value === providerValue,
    );

    setConfiguration((currentConfiguration) =>
      currentConfiguration && provider
        ? {
            ...currentConfiguration,
            embeddingProvider: providerValue,
            embeddingModel: provider.models[0].value,
          }
        : currentConfiguration,
    );
  }

  /**
   * Changes the generation provider and selects its first compatible model.
   *
   * @param providerValue - Stable generation provider identifier.
   * @returns Nothing. Provider and model state are updated atomically.
   */
  function updateGenerationProvider(providerValue: string): void {
    const provider = pipelineOptions?.generation.providers.find(
      (option) => option.value === providerValue,
    );

    setConfiguration((currentConfiguration) =>
      currentConfiguration && provider
        ? {
            ...currentConfiguration,
            generationProvider: providerValue,
            generationModel: provider.models[0].value,
          }
        : currentConfiguration,
    );
  }

  /**
   * Switches run input mode and resets metrics that are incompatible with that mode.
   *
   * @param mode - The newly selected question or dataset input mode.
   * @returns Nothing. Input mode and dependent metric selections are updated together.
   */
  function updateInputMode(mode: ExperimentInputMode): void {
    setInputMode(mode);

    setConfiguration((currentConfiguration) => {
      // Keep state unchanged until the option catalog has initialized the form.
      if (!currentConfiguration || !pipelineOptions) {
        return currentConfiguration;
      }

      const allowedAnswerMetricValues = new Set(
        pipelineOptions.evaluation.answer_metrics
          .filter((metric) => mode === "dataset" || !metric.requires_reference_answer)
          .map((metric) => metric.value),
      );

      return {
        ...currentConfiguration,
        retrievalMetrics: mode === "dataset" ? currentConfiguration.retrievalMetrics : [],
        answerMetrics: currentConfiguration.answerMetrics.filter((metric) =>
          allowedAnswerMetricValues.has(metric),
        ),
      };
    });
  }

  /**
   * Explains missing run requirements when the primary action is unavailable.
   *
   * @returns Nothing. A readable notice is shown when a required value is missing.
   */
  function handleRunExperiment(): void {
    // Wait for the configuration catalog before allowing a run to proceed.
    if (!configuration) {
      setRunNotice("Wait for the pipeline configuration to finish loading.");
      return;
    }

    // Require one immutable corpus as the source for the experiment.
    if (!selectedCorpusId) {
      setRunNotice("Select a corpus before running the experiment.");
      return;
    }

    // Require the input selected by the user for this experiment.
    if (!hasRunInput) {
      setRunNotice(
        inputMode === "question"
          ? "Enter a question before running the experiment."
          : "Choose an evaluation dataset before running the experiment.",
      );
      return;
    }

    setRunNotice(null);
  }

  return (
    <main
      className={`
        min-h-screen bg-[var(--page-surface)] text-[var(--charcoal)] lg:grid
        lg:grid-cols-[240px_minmax(0,1fr)]
      `}
    >
      <WorkbenchSidebar activeLabel="Experiments" />

      {/* The canvas holds the title, stage rail, and backend-backed configuration panels. */}
      <WorkbenchGridCanvas className="px-5 py-8 sm:px-8 lg:px-12">
        <div className="mx-auto w-full max-w-6xl">
          {/* The header frames configuration as a reproducible pipeline run. */}
          <header
            className={`
              mb-8 flex flex-col gap-6 border-b border-[var(--border-subtle)] pb-8
              lg:flex-row lg:items-end lg:justify-between
            `}
          >
            <div className="max-w-2xl">
              <p
                className={`
                  mb-2 font-mono text-[10px] font-bold uppercase tracking-[0.16em]
                  text-[var(--muted-text)] lg:hidden
                `}
              >
                RAG Playground / Experiments
              </p>
              <p
                className={`
                  font-mono text-[10px] font-bold uppercase tracking-[0.16em]
                  text-[var(--muted-text)]
                `}
              >
                New experiment
              </p>
              <h1 className="mt-3 text-4xl font-bold tracking-[-0.055em] sm:text-5xl">
                Test the path to the answer.
              </h1>
              <p className="mt-3 max-w-xl text-sm leading-6 text-[var(--tone-black)]">
                Configure each supported stage, then keep the exact setup attached to the run it
                produces.
              </p>
            </div>

            {/* The primary action explains any missing requirement when pressed. */}
            <button
              aria-disabled={!isRunReady}
              className={`
                inline-flex shrink-0 items-center gap-2 rounded
                bg-[var(--charcoal)] px-4 py-2.5 text-sm font-semibold
                text-[var(--white)] transition-colors hover:bg-[var(--primary-hover)]
                focus-visible:outline-none focus-visible:ring-2
                focus-visible:ring-[var(--charcoal)] focus-visible:ring-offset-2
                ${isRunReady ? "" : "cursor-not-allowed opacity-45"}
              `}
              onClick={handleRunExperiment}
              type="button"
            >
              <FiPlay aria-hidden="true" className="size-4" />
              Run experiment
            </button>
          </header>

          {/* The content split keeps pipeline order visible beside stage controls. */}
          <div className="grid gap-8 lg:grid-cols-[270px_minmax(0,1fr)]">
            {/* The stage rail communicates pipeline order and current focus. */}
            <aside className="h-fit lg:sticky lg:top-8">
              <nav
                aria-label="Experiment stages"
                className={`
                  relative overflow-x-auto rounded border
                  border-[var(--border-subtle)] bg-[var(--white)] p-2
                  lg:overflow-visible
                `}
              >
                {/* The trace fills as focus moves through the ordered pipeline. */}
                <div
                  className={`
                    pointer-events-none absolute bottom-5 left-8 top-5 hidden
                    w-px bg-[var(--border-subtle)] lg:block
                  `}
                >
                  <div
                    className={`
                      h-full w-full origin-top bg-[var(--charcoal)]
                      transition-transform duration-300
                    `}
                    style={{ transform: `scaleY(${stageProgress})` }}
                  />
                </div>

                <div className="relative flex min-w-max gap-1 lg:min-w-0 lg:flex-col">
                  {experimentStages.map((stage, index) => {
                    const isActive = stage.label === activeStage;
                    const isComplete = index < activeStageIndex;

                    return (
                      <button
                        aria-current={isActive ? "step" : undefined}
                        className={`
                          flex min-w-36 items-center gap-3 rounded-sm px-3 py-3
                          text-left transition-colors hover:bg-[var(--panel-surface)]
                          focus-visible:outline-none focus-visible:ring-2
                          focus-visible:ring-[var(--charcoal)] lg:min-w-0
                          ${
                            isActive
                              ? "bg-[var(--panel-surface)] text-[var(--charcoal)]"
                              : "text-[var(--muted-text)]"
                          }
                        `}
                        key={stage.label}
                        onClick={() => focusStage(stage.label)}
                        type="button"
                      >
                        <span
                          className={`
                            relative z-10 flex size-6 shrink-0 items-center
                            justify-center rounded-full border font-mono text-[9px]
                            font-bold
                            ${
                              isActive || isComplete
                                ? `border-[var(--charcoal)] bg-[var(--charcoal)]
                                text-[var(--white)]`
                                : `border-[var(--border-strong)]
                                bg-[var(--white)]`
                            }
                          `}
                        >
                          {isComplete ? (
                            <FiCheck aria-hidden="true" className="size-3" />
                          ) : (
                            String(index + 1).padStart(2, "0")
                          )}
                        </span>
                        <span className="min-w-0">
                          <span className="block text-xs font-semibold">{stage.label}</span>
                          <span
                            className={`
                              mt-0.5 block truncate font-mono text-[9px] uppercase
                              tracking-wide text-[var(--muted-text)]
                            `}
                          >
                            {stage.detail}
                          </span>
                        </span>
                      </button>
                    );
                  })}
                </div>
              </nav>
            </aside>

            {/* The form area reports request state before rendering backend-backed controls. */}
            <div className="space-y-4">
              {isLoading ? (
                <section
                  aria-live="polite"
                  className={`
                    rounded border border-[var(--border-subtle)]
                    bg-[var(--white)] p-8
                  `}
                >
                  <div className="flex items-center gap-3 text-sm font-semibold">
                    <FiRefreshCw aria-hidden="true" className="size-4 animate-spin" />
                    Loading pipeline options…
                  </div>
                </section>
              ) : loadError || !pipelineOptions || !configuration ? (
                <section
                  className={`
                    rounded border border-[var(--toast-error-border)]
                    bg-[var(--toast-error-surface)] p-6
                  `}
                  role="alert"
                >
                  <div className="flex items-start gap-3">
                    <FiAlertCircle
                      aria-hidden="true"
                      className={`
                        mt-0.5 size-5 shrink-0 text-[var(--toast-error-text)]
                      `}
                    />
                    <div>
                      <h2 className="text-sm font-bold text-[var(--toast-error-text)]">
                        Configuration options are unavailable
                      </h2>
                      <p className="mt-1 text-sm leading-6 text-[var(--tone-black)]">
                        {loadError ?? "The backend response could not be read."}
                      </p>
                      <button
                        className={`
                          mt-4 inline-flex items-center gap-2 rounded border
                          border-[var(--toast-error-border)] bg-[var(--white)]
                          px-3 py-2 text-xs font-bold text-[var(--toast-error-text)]
                          focus-visible:outline-none focus-visible:ring-2
                          focus-visible:ring-[var(--toast-error-text)]
                        `}
                        onClick={() => setLoadAttempt((attempt) => attempt + 1)}
                        type="button"
                      >
                        <FiRefreshCw aria-hidden="true" className="size-3.5" />
                        Try again
                      </button>
                    </div>
                  </div>
                </section>
              ) : (
                <>
                  {/* Source exposes only immutable corpora already persisted by ingestion. */}
                  <article
                    className={`
                      scroll-mt-8 rounded border bg-[var(--white)] outline-none
                      ${
                        activeStage === "Source"
                          ? "border-[var(--charcoal)]"
                          : "border-[var(--border-subtle)]"
                      }
                    `}
                    onClick={() => setActiveStage("Source")}
                    ref={(element) => {
                      stageSectionRefs.current.Source = element;
                    }}
                    tabIndex={-1}
                  >
                    <div className="p-5 sm:p-6">
                      <p
                        className={`
                          font-mono text-[10px] font-bold uppercase
                          tracking-[0.14em] text-[var(--muted-text)]
                        `}
                      >
                        01 / Source
                      </p>
                      <h2 className="mt-2 text-lg font-bold tracking-[-0.03em]">Choose a corpus</h2>
                      <p className="mt-1 text-sm leading-6 text-[var(--tone-black)]">
                        Select one immutable corpus already uploaded to this workspace.
                      </p>

                      {/* The source search keeps results attached to the input. */}
                      <div className="relative mt-5">
                        <label className="relative block">
                          <span className="sr-only">Search uploaded corpora</span>
                          <FiSearch
                            aria-hidden="true"
                            className={`
                              pointer-events-none absolute left-3 top-1/2 size-4
                              -translate-y-1/2 text-[var(--muted-text)]
                            `}
                          />
                          <input
                            aria-controls="uploaded-corpus-results"
                            aria-expanded={isCorpusDropdownOpen}
                            aria-haspopup="listbox"
                            className={`
                              w-full rounded border border-[var(--border-subtle)]
                              bg-[var(--page-surface)] py-2.5 pl-9 pr-3 text-sm
                              outline-none transition-colors
                              placeholder:text-[var(--placeholder-text)]
                              focus:border-[var(--charcoal)] focus:ring-1
                              focus:ring-[var(--charcoal)] disabled:cursor-not-allowed
                              disabled:opacity-60
                            `}
                            disabled={corpora.length === 0}
                            onBlur={() =>
                              window.setTimeout(() => setIsCorpusDropdownOpen(false), 0)
                            }
                            onChange={(event) => {
                              setCorpusSearch(event.target.value);
                              setIsCorpusDropdownOpen(true);
                            }}
                            onFocus={() => setIsCorpusDropdownOpen(true)}
                            placeholder={
                              corpora.length === 0
                                ? "Upload a corpus before running an experiment"
                                : "Search uploaded corpora"
                            }
                            role="combobox"
                            type="search"
                            value={corpusSearch}
                          />
                        </label>

                        {/* Search results list only real corpus IDs returned by FastAPI. */}
                        {isCorpusDropdownOpen ? (
                          <div
                            className={`
                              absolute left-0 right-0 top-full z-20 mt-1
                              overflow-hidden rounded border
                              border-[var(--border-strong)] bg-[var(--white)]
                              shadow-lg
                            `}
                            id="uploaded-corpus-results"
                            role="listbox"
                          >
                            <div className="max-h-56 overflow-y-auto py-1">
                              {filteredCorpora.length > 0 ? (
                                filteredCorpora.map((corpus) => {
                                  const isSelected = corpus.id === selectedCorpusId;

                                  return (
                                    <button
                                      aria-selected={isSelected}
                                      className={`
                                        flex w-full items-center justify-between
                                        gap-4 px-4 py-3 text-left text-sm font-semibold
                                        hover:bg-[var(--landing-soft)]
                                        focus-visible:outline-none focus-visible:ring-2
                                        focus-visible:ring-inset
                                        focus-visible:ring-[var(--charcoal)]
                                        ${
                                          isSelected
                                            ? "bg-[var(--panel-surface)]"
                                            : "bg-[var(--white)]"
                                        }
                                      `}
                                      key={corpus.id}
                                      onClick={() => {
                                        setSelectedCorpusId(corpus.id);
                                        setCorpusSearch(corpus.name);
                                        setIsCorpusDropdownOpen(false);
                                      }}
                                      onMouseDown={(event) => event.preventDefault()}
                                      role="option"
                                      type="button"
                                    >
                                      {corpus.name}
                                      {isSelected ? (
                                        <FiCheck
                                          aria-label="Selected corpus"
                                          className="size-4 shrink-0"
                                        />
                                      ) : null}
                                    </button>
                                  );
                                })
                              ) : (
                                <p className="px-4 py-4 text-sm text-[var(--muted-text)]">
                                  No uploaded corpus matches that search.
                                </p>
                              )}
                            </div>
                          </div>
                        ) : null}
                      </div>
                    </div>
                  </article>

                  {/* Configuration cards expose only fields present in the backend catalog. */}
                  <div className="grid gap-4 xl:grid-cols-2">
                    <article
                      className={`
                        scroll-mt-8 rounded border bg-[var(--white)] outline-none
                        ${
                          activeStage === "Chunk"
                            ? "border-[var(--charcoal)]"
                            : "border-[var(--border-subtle)]"
                        }
                      `}
                      onClick={() => setActiveStage("Chunk")}
                      ref={(element) => {
                        stageSectionRefs.current.Chunk = element;
                      }}
                      tabIndex={-1}
                    >
                      {/* Chunk controls follow overlap support from the catalog. */}
                      <div className="p-5 sm:p-6">
                        <p
                          className={`
                            font-mono text-[10px] font-bold uppercase
                            tracking-[0.14em] text-[var(--muted-text)]
                          `}
                        >
                          02 / Chunk
                        </p>
                        <h2 className="mt-2 text-lg font-bold tracking-[-0.03em]">
                          Shape the context
                        </h2>
                        <p className="mt-1 text-sm leading-6 text-[var(--tone-black)]">
                          Keep related ideas together without making retrieval noisy.
                        </p>
                      </div>
                      <div className="space-y-4 border-t border-[var(--border-soft)] p-5 sm:p-6">
                        <SelectControl
                          helperText={selectedStrategy?.description ?? undefined}
                          id="chunking-strategy"
                          label="Strategy"
                          onChange={updateChunkingStrategy}
                          options={pipelineOptions.chunking.strategies}
                          value={configuration.chunkingStrategy}
                        />
                        <NumberControl
                          id="chunk-size"
                          label="Chunk size (tokens)"
                          onChange={(value) => updateConfiguration("chunkSizeTokens", value)}
                          setting={pipelineOptions.chunking.chunk_size_tokens}
                          value={configuration.chunkSizeTokens}
                        />
                        {selectedStrategy?.supports_overlap ? (
                          <NumberControl
                            helperText="Must be smaller than the chunk size."
                            id="chunk-overlap"
                            label="Chunk overlap (tokens)"
                            onChange={(value) => updateConfiguration("chunkOverlapTokens", value)}
                            setting={pipelineOptions.chunking.chunk_overlap_tokens}
                            value={configuration.chunkOverlapTokens}
                          />
                        ) : null}
                      </div>
                    </article>

                    <article
                      className={`
                        scroll-mt-8 rounded border bg-[var(--white)] outline-none
                        ${
                          activeStage === "Embed"
                            ? "border-[var(--charcoal)]"
                            : "border-[var(--border-subtle)]"
                        }
                      `}
                      onClick={() => setActiveStage("Embed")}
                      ref={(element) => {
                        stageSectionRefs.current.Embed = element;
                      }}
                      tabIndex={-1}
                    >
                      {/* Embedding models are filtered by their selected provider. */}
                      <div className="p-5 sm:p-6">
                        <p
                          className={`
                            font-mono text-[10px] font-bold uppercase
                            tracking-[0.14em] text-[var(--muted-text)]
                          `}
                        >
                          03 / Embed
                        </p>
                        <h2 className="mt-2 text-lg font-bold tracking-[-0.03em]">
                          Map the meaning
                        </h2>
                        <p className="mt-1 text-sm leading-6 text-[var(--tone-black)]">
                          Choose how chunks and questions share a vector space.
                        </p>
                      </div>
                      <div className="space-y-4 border-t border-[var(--border-soft)] p-5 sm:p-6">
                        <SelectControl
                          id="embedding-provider"
                          label="Provider"
                          onChange={updateEmbeddingProvider}
                          options={pipelineOptions.embedding.providers}
                          value={configuration.embeddingProvider}
                        />
                        <SelectControl
                          id="embedding-model"
                          label="Model"
                          onChange={(value) => updateConfiguration("embeddingModel", value)}
                          options={embeddingProvider?.models ?? []}
                          value={configuration.embeddingModel}
                        />
                        <SelectControl
                          id="distance-metric"
                          label="Distance metric"
                          onChange={(value) => updateConfiguration("distanceMetric", value)}
                          options={pipelineOptions.embedding.distance_metrics}
                          value={configuration.distanceMetric}
                        />
                      </div>
                    </article>

                    <article
                      className={`
                        scroll-mt-8 rounded border bg-[var(--white)] outline-none
                        ${
                          activeStage === "Retrieve"
                            ? "border-[var(--charcoal)]"
                            : "border-[var(--border-subtle)]"
                        }
                      `}
                      onClick={() => setActiveStage("Retrieve")}
                      ref={(element) => {
                        stageSectionRefs.current.Retrieve = element;
                      }}
                      tabIndex={-1}
                    >
                      {/* Retrieval exposes only top K in the current catalog. */}
                      <div className="p-5 sm:p-6">
                        <p
                          className={`
                            font-mono text-[10px] font-bold uppercase
                            tracking-[0.14em] text-[var(--muted-text)]
                          `}
                        >
                          04 / Retrieve
                        </p>
                        <h2 className="mt-2 text-lg font-bold tracking-[-0.03em]">
                          Bring back evidence
                        </h2>
                        <p className="mt-1 text-sm leading-6 text-[var(--tone-black)]">
                          Control how many indexed chunks become answer context.
                        </p>
                      </div>
                      <div className="border-t border-[var(--border-soft)] p-5 sm:p-6">
                        <NumberControl
                          helperText={"The number of nearest chunks returned for each question."}
                          id="top-k"
                          label="Top K"
                          onChange={(value) => updateConfiguration("topK", value)}
                          setting={pipelineOptions.retrieval.top_k}
                          value={configuration.topK}
                        />
                      </div>
                    </article>

                    <article
                      className={`
                        scroll-mt-8 rounded border bg-[var(--white)] outline-none
                        ${
                          activeStage === "Generate"
                            ? "border-[var(--charcoal)]"
                            : "border-[var(--border-subtle)]"
                        }
                      `}
                      onClick={() => setActiveStage("Generate")}
                      ref={(element) => {
                        stageSectionRefs.current.Generate = element;
                      }}
                      tabIndex={-1}
                    >
                      {/* Generation pairs models with catalog-defined controls. */}
                      <div className="p-5 sm:p-6">
                        <p
                          className={`
                            font-mono text-[10px] font-bold uppercase
                            tracking-[0.14em] text-[var(--muted-text)]
                          `}
                        >
                          05 / Generate
                        </p>
                        <h2 className="mt-2 text-lg font-bold tracking-[-0.03em]">
                          Write the answer
                        </h2>
                        <p className="mt-1 text-sm leading-6 text-[var(--tone-black)]">
                          Select the response model and its output behavior.
                        </p>
                      </div>
                      <div className="space-y-4 border-t border-[var(--border-soft)] p-5 sm:p-6">
                        <SelectControl
                          id="generation-provider"
                          label="Provider"
                          onChange={updateGenerationProvider}
                          options={pipelineOptions.generation.providers}
                          value={configuration.generationProvider}
                        />
                        <SelectControl
                          helperText={generationModelHelperText}
                          id="generation-model"
                          label="Model"
                          onChange={(value) => updateConfiguration("generationModel", value)}
                          options={generationProvider?.models ?? []}
                          value={configuration.generationModel}
                        />
                        <NumberControl
                          id="temperature"
                          label="Temperature"
                          onChange={(value) => updateConfiguration("temperature", value)}
                          setting={pipelineOptions.generation.temperature}
                          step={0.1}
                          value={configuration.temperature}
                        />
                        <NumberControl
                          id="max-output-tokens"
                          label="Max output tokens"
                          onChange={(value) => updateConfiguration("maxOutputTokens", value)}
                          setting={pipelineOptions.generation.max_output_tokens}
                          value={configuration.maxOutputTokens}
                        />
                      </div>
                    </article>

                    <article
                      className={`
                        scroll-mt-8 rounded border bg-[var(--white)] outline-none
                        xl:col-span-2
                        ${
                          activeStage === "Evaluate"
                            ? "border-[var(--charcoal)]"
                            : "border-[var(--border-subtle)]"
                        }
                      `}
                      onClick={() => setActiveStage("Evaluate")}
                      ref={(element) => {
                        stageSectionRefs.current.Evaluate = element;
                      }}
                      tabIndex={-1}
                    >
                      {/* Evaluation options follow the evidence available in the run input. */}
                      <div className="p-5 sm:p-6">
                        <p
                          className={`
                            font-mono text-[10px] font-bold uppercase
                            tracking-[0.14em] text-[var(--muted-text)]
                          `}
                        >
                          06 / Evaluate
                        </p>
                        <h2 className="mt-2 text-lg font-bold tracking-[-0.03em]">
                          Check the result
                        </h2>
                        <p className="mt-1 text-sm leading-6 text-[var(--tone-black)]">
                          {inputMode === "question"
                            ? "Use answer metrics that do not require labelled ground truth."
                            : "Measure retrieval and answers against the dataset annotations."}
                        </p>
                      </div>
                      <div
                        className={`
                          grid gap-4 border-t border-[var(--border-soft)] p-5
                          ${inputMode === "dataset" ? "sm:grid-cols-2" : "sm:grid-cols-1"}
                          sm:p-6
                        `}
                      >
                        {inputMode === "dataset" ? (
                          <MetricCheckboxGroup
                            helperText={
                              "Select any metrics supported by the dataset's relevance labels."
                            }
                            id="retrieval-metrics"
                            label="Retrieval metrics"
                            onChange={(values) => updateMetrics("retrievalMetrics", values)}
                            options={pipelineOptions.evaluation.retrieval_metrics}
                            selectedValues={configuration.retrievalMetrics}
                          />
                        ) : null}
                        <MetricCheckboxGroup
                          helperText="Leave all metrics unchecked to skip answer evaluation."
                          id="answer-metrics"
                          label="Answer metrics"
                          onChange={(values) => updateMetrics("answerMetrics", values)}
                          options={availableAnswerMetrics}
                          selectedValues={configuration.answerMetrics}
                        />
                      </div>
                    </article>
                  </div>

                  {/* Run input follows metrics so test data comes after pipeline configuration. */}
                  <ExperimentInputControl
                    datasetFile={datasetFile}
                    mode={inputMode}
                    onDatasetChange={setDatasetFile}
                    onModeChange={updateInputMode}
                    onQuestionChange={setQuestion}
                    question={question}
                  />

                  {/* The footer clarifies that this pass configures rather than executes a run. */}
                  <div
                    className={`
                      flex flex-col items-start justify-between gap-4 border-t
                      border-[var(--border-subtle)] pt-6 sm:flex-row sm:items-center
                    `}
                  >
                    <p className="max-w-md text-xs leading-5 text-[var(--muted-text)]">
                      Options and defaults come from the backend. Run persistence and execution will
                      connect in the next pass.
                    </p>
                    <button
                      aria-disabled={!isRunReady}
                      className={`
                        inline-flex items-center gap-2 rounded bg-[var(--charcoal)]
                        px-4 py-2.5 text-sm font-semibold text-[var(--white)]
                        transition-colors hover:bg-[var(--primary-hover)]
                        focus-visible:outline-none focus-visible:ring-2
                        focus-visible:ring-[var(--charcoal)]
                        focus-visible:ring-offset-2
                        ${isRunReady ? "" : "cursor-not-allowed opacity-45"}
                      `}
                      onClick={handleRunExperiment}
                      type="button"
                    >
                      <FiPlay aria-hidden="true" className="size-4" />
                      Run experiment
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </WorkbenchGridCanvas>

      {/* The run notice explains why an unavailable action cannot proceed. */}
      {runNotice ? (
        <Toast message={runNotice} onDismiss={() => setRunNotice(null)} type="error" />
      ) : null}
    </main>
  );
}
