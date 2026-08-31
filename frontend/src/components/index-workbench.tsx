"use client";

import { useEffect, useRef, useState } from "react";
import {
  FiAlertCircle,
  FiCheck,
  FiDatabase,
  FiPlus,
  FiRefreshCw,
  FiSearch,
  FiX,
} from "react-icons/fi";
import NumberControl from "@/components/number-control";
import SelectControl from "@/components/select-control";
import WorkbenchGridCanvas from "@/components/workbench-grid-canvas";
import WorkbenchSidebar from "@/components/workbench-sidebar";
import apiClient, { isCancel } from "@/lib/axios";
import { type CorpusOption, parseCorpora } from "@/validation/corpora";
import {
  type PipelineOptions,
  parsePipelineOptions,
} from "@/validation/pipeline-options";

/** The three persisted stages represented by a prepared vector index. */
const indexStages = [
  { label: "Corpus", detail: "Select data" },
  { label: "Chunk", detail: "Split text" },
  { label: "Embed", detail: "Vectorize" },
];

/** Editable values needed to describe a prepared index without retrieval settings. */
type IndexConfiguration = {
  chunkingStrategy: string;
  chunkSizeTokens: string;
  chunkOverlapTokens: string;
  embeddingProvider: string;
  embeddingModel: string;
  distanceMetric: string;
};

/**
 * Creates the initial index configuration from backend-owned defaults.
 *
 * @param options - Validated pipeline option catalog returned by FastAPI.
 * @returns A complete editable configuration for chunking and embedding.
 */
function createDefaultIndexConfiguration(options: PipelineOptions): IndexConfiguration {
  // The first provider is the backend-defined default when no explicit default is published.
  const embeddingProvider = options.embedding.providers[0];

  return {
    chunkingStrategy: options.chunking.strategies[0].value,
    chunkSizeTokens: String(options.chunking.chunk_size_tokens.default),
    chunkOverlapTokens: String(options.chunking.chunk_overlap_tokens.default),
    embeddingProvider: embeddingProvider.value,
    embeddingModel: embeddingProvider.models[0].value,
    distanceMetric: options.embedding.distance_metrics[0].value,
  };
}

/**
 * Renders one labeled stage in the index preparation rail.
 *
 * @param props - Stage identity, supporting text, position, and active state.
 * @returns A compact rail item that communicates the preparation sequence.
 */
function IndexStage({
  label,
  detail,
  position,
  isActive,
}: {
  label: string;
  detail: string;
  position: number;
  isActive: boolean;
}) {
  return (
    /* Each stage maps directly to an index-building operation. */
    <div
      className={`relative flex gap-4 px-3 py-4 ${
        isActive ? "bg-[var(--hover-surface)]" : "bg-transparent"
      }`}
    >
      {/* The marker distinguishes the selected stage from later configuration stages. */}
      <span
        className={`relative z-10 flex size-7 shrink-0 items-center justify-center rounded-full \
          font-mono text-[10px] font-bold ${
          isActive
            ? "bg-[var(--charcoal)] text-white"
            : "bg-[var(--border-subtle)] text-[var(--tone-black)]"
        }`}
      >
        {isActive ? <FiCheck aria-hidden="true" className="size-4" /> : `0${position}`}
      </span>

      {/* Stage copy names both the object and its pipeline operation. */}
      <span>
        <span className="block text-base font-semibold text-[var(--charcoal)]">
          {label}
        </span>
        <span
          className="mt-1 block font-mono text-[10px] font-bold uppercase \
            tracking-[0.12em] text-[var(--tone-black)]"
        >
          {detail}
        </span>
      </span>
    </div>
  );
}

/**
 * Renders the frontend-only workspace used to configure a reusable vector index.
 *
 * @returns The responsive corpus, chunking, embedding, and naming interface.
 */
export default function IndexWorkbench() {
  // Stores the validated backend-owned configuration catalog.
  const [pipelineOptions, setPipelineOptions] = useState<PipelineOptions | null>(null);

  // Stores the editable chunking and embedding choices initialized from defaults.
  const [configuration, setConfiguration] = useState<IndexConfiguration | null>(null);

  // Stores the persisted corpora available for index preparation.
  const [corpora, setCorpora] = useState<CorpusOption[]>([]);

  // Stores the stable identifier of the corpus selected for the index.
  const [selectedCorpusId, setSelectedCorpusId] = useState("");

  // Stores the visible corpus query used to filter the picker.
  const [corpusSearch, setCorpusSearch] = useState("");

  // Controls whether the corpus result menu is visible.
  const [isCorpusDropdownOpen, setIsCorpusDropdownOpen] = useState(false);

  // Stores the required human-readable identity for the future persisted index.
  const [indexName, setIndexName] = useState("");

  // Records whether the name field was blurred or submitted for error visibility.
  const [isNameTouched, setIsNameTouched] = useState(false);

  // Indicates that catalog and corpus data are being requested.
  const [isLoading, setIsLoading] = useState(true);

  // Stores a readable loading or API contract failure.
  const [loadError, setLoadError] = useState<string | null>(null);

  // Stores a frontend-only submission notice while backend creation is deferred.
  const [buildNotice, setBuildNotice] = useState<string | null>(null);

  // Increments when the user asks to retry initial API requests.
  const [loadAttempt, setLoadAttempt] = useState(0);

  // Gives the header action direct focus access to the required identity field.
  const indexNameInputRef = useRef<HTMLInputElement | null>(null);

  // Loads existing option and corpus APIs whenever the page opens or retries.
  useEffect(() => {
    const abortController = new AbortController();

    /**
     * Loads and validates the existing data required by the index form.
     *
     * @returns A promise resolved after loading, error, and defaults are updated.
     */
    async function loadIndexData(): Promise<void> {
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
        setConfiguration(createDefaultIndexConfiguration(validatedOptions));
        setCorpora(validatedCorpora);

        // Select the first corpus so an existing workspace is immediately configurable.
        if (validatedCorpora.length > 0) {
          setSelectedCorpusId(validatedCorpora[0].id);
          setCorpusSearch(validatedCorpora[0].name);
        }
      } catch (error) {
        // Cancellation is expected when the user leaves while requests are active.
        if (isCancel(error)) {
          return;
        }

        setPipelineOptions(null);
        setConfiguration(null);
        setLoadError(
          error instanceof Error
            ? error.message
            : "The index configuration could not be loaded.",
        );
      } finally {
        // Avoid updating state after the owning page has unmounted.
        if (!abortController.signal.aborted) {
          setIsLoading(false);
        }
      }
    }

    void loadIndexData();

    return () => {
      abortController.abort();
    };
  }, [loadAttempt]);

  // Match corpus names locally because the existing endpoint returns the full inventory.
  const filteredCorpora = corpora.filter((corpus) =>
    corpus.name.toLocaleLowerCase().includes(corpusSearch.trim().toLocaleLowerCase()),
  );

  // Resolve provider-specific models so incompatible model choices are never shown.
  const embeddingProvider = pipelineOptions?.embedding.providers.find(
    (provider) => provider.value === configuration?.embeddingProvider,
  );

  // Resolve the selected strategy to describe its behavior and overlap support.
  const selectedStrategy = pipelineOptions?.chunking.strategies.find(
    (strategy) => strategy.value === configuration?.chunkingStrategy,
  );

  // A trimmed non-empty name is required before the frontend can initialize a build.
  const hasValidName = indexName.trim().length > 0;

  // Every source and configuration choice must exist before the action becomes available.
  const canInitialize = Boolean(
    hasValidName && selectedCorpusId && configuration && !isLoading && !loadError,
  );

  /**
   * Focuses the required name field when the page-level create action is used.
   *
   * @returns Nothing. Focus and validation visibility are updated in place.
   */
  function focusIndexName(): void {
    setIsNameTouched(true);
    indexNameInputRef.current?.focus();
    indexNameInputRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  /**
   * Validates the index form without calling an unavailable backend creation API.
   *
   * @param event - Native form submission event from the build action.
   * @returns Nothing. The required field or frontend-only notice is surfaced.
   */
  function handleInitialize(event: React.FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    setIsNameTouched(true);
    setBuildNotice(null);

    // Keep focus on the missing required value instead of submitting incomplete data.
    if (!hasValidName) {
      indexNameInputRef.current?.focus();
      return;
    }

    // Do not claim a persisted build while this task intentionally excludes backend work.
    if (canInitialize) {
      setBuildNotice(
        `“${indexName.trim()}” is ready. Index creation will be connected to the backend next.`,
      );
    }
  }

  return (
    <main className="grid min-h-screen lg:grid-cols-[17.5rem_minmax(0,1fr)]">
      <WorkbenchSidebar activeLabel="Indexes" />

      {/* The grid canvas contains the complete index preparation workspace. */}
      <WorkbenchGridCanvas className="min-h-screen px-5 py-8 sm:px-8 lg:px-12 lg:py-10">
        {/* The page header states the single outcome and shortcuts to its required identity. */}
        <header
          className="mx-auto flex max-w-[74rem] flex-col gap-6 border-b-2 \
            border-[var(--charcoal)] pb-7 sm:flex-row sm:items-end sm:justify-between"
        >
          <div>
            <p
              className="font-mono text-[10px] font-bold uppercase tracking-[0.16em] \
                text-[var(--tone-black)]"
            >
              New index
            </p>
            <h1
              className="mt-3 max-w-3xl text-3xl font-semibold tracking-[-0.035em] \
                text-[var(--charcoal)] sm:text-4xl"
            >
              Define the data foundation.
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-[var(--tone-black)]">
              Configure how a corpus is chunked and embedded into a reusable vector index.
            </p>
          </div>

          <button
            className="flex w-fit items-center gap-2 rounded-sm bg-[var(--charcoal)] px-4 py-3 \
              text-sm font-semibold text-white transition-colors hover:bg-[var(--primary-hover)] \
              focus-visible:outline-2 focus-visible:outline-offset-2 \
              focus-visible:outline-[var(--charcoal)]"
            onClick={focusIndexName}
            type="button"
          >
            <FiPlus aria-hidden="true" className="size-5" />
            Create index
          </button>
        </header>

        {/* The workbench pairs a pipeline rail with configuration surfaces. */}
        <div
          className="mx-auto mt-8 grid max-w-[74rem] items-start gap-6 \
            lg:grid-cols-[15rem_minmax(0,1fr)]"
        >
          {/* The rail makes the boundaries of the prepared artifact explicit. */}
          <aside
            className="border border-[var(--border-strong)] bg-white p-2 lg:sticky lg:top-8"
            aria-label="Index stages"
          >
            {indexStages.map((stage, index) => (
              <IndexStage
                detail={stage.detail}
                isActive={index === 0}
                key={stage.label}
                label={stage.label}
                position={index + 1}
              />
            ))}
          </aside>

          {/* Loading and failure states replace unusable controls with a clear next action. */}
          {isLoading ? (
            <section
              className="border border-[var(--border-strong)] bg-white p-8"
              aria-live="polite"
            >
              <FiRefreshCw aria-hidden="true" className="size-5 animate-spin" />
              <p className="mt-4 text-sm text-[var(--tone-black)]">
                Loading index configuration…
              </p>
            </section>
          ) : loadError ? (
            <section className="border border-[var(--toast-error-border)] bg-white p-8">
              <FiAlertCircle
                aria-hidden="true"
                className="size-5 text-[var(--toast-error-text)]"
              />
              <h2 className="mt-4 text-lg font-semibold text-[var(--charcoal)]">
                Configuration unavailable
              </h2>
              <p className="mt-2 text-sm text-[var(--tone-black)]">{loadError}</p>
              <button
                className="mt-5 flex items-center gap-2 border border-[var(--charcoal)] \
                  bg-white px-3 py-2 text-sm font-semibold text-[var(--charcoal)]"
                onClick={() => setLoadAttempt((attempt) => attempt + 1)}
                type="button"
              >
                <FiRefreshCw aria-hidden="true" />
                Retry
              </button>
            </section>
          ) : pipelineOptions && configuration ? (
            <form className="space-y-6" noValidate onSubmit={handleInitialize}>
              {/* Corpus selection anchors the configuration to one immutable source. */}
              <section className="border border-[var(--border-strong)] bg-white">
                <div className="border-b border-[var(--border-strong)] p-6">
                  <p
                    className="font-mono text-[10px] font-bold uppercase \
                      tracking-[0.12em] text-[var(--tone-black)]"
                  >
                    01 / Corpus
                  </p>
                  <h2 className="mt-3 text-xl font-semibold text-[var(--charcoal)]">
                    Choose a corpus
                  </h2>
                  <p className="mt-2 text-sm text-[var(--tone-black)]">
                    Select one uploaded corpus as the immutable source for this index.
                  </p>
                </div>

                {/* The searchable picker keeps large corpus inventories manageable. */}
                <div className="relative bg-[var(--page-surface)] p-6">
                  <FiSearch
                    aria-hidden="true"
                    className="absolute left-9 top-1/2 size-5 -translate-y-1/2 \
                      text-[var(--tone-black)]"
                  />
                  <input
                    aria-label="Search corpora"
                    className="w-full rounded-sm border border-[var(--border-strong)] bg-white \
                      py-3 pl-11 pr-11 text-sm text-[var(--charcoal)] outline-none \
                      focus:border-[var(--charcoal)] focus:ring-1 focus:ring-[var(--charcoal)]"
                    onChange={(event) => {
                      setCorpusSearch(event.target.value);
                      setSelectedCorpusId("");
                      setIsCorpusDropdownOpen(true);
                    }}
                    onFocus={() => setIsCorpusDropdownOpen(true)}
                    placeholder="Search uploaded corpora"
                    value={corpusSearch}
                  />
                  {corpusSearch ? (
                    <button
                      aria-label="Clear corpus selection"
                      className="absolute right-9 top-1/2 -translate-y-1/2 bg-transparent p-1 \
                        text-[var(--tone-black)]"
                      onClick={() => {
                        setCorpusSearch("");
                        setSelectedCorpusId("");
                        setIsCorpusDropdownOpen(true);
                      }}
                      type="button"
                    >
                      <FiX aria-hidden="true" className="size-5" />
                    </button>
                  ) : null}

                  {/* Matching corpus records appear directly below the search field. */}
                  {isCorpusDropdownOpen ? (
                    <div
                      className="absolute inset-x-6 top-[calc(100%-1.5rem)] z-20 max-h-52 \
                        overflow-y-auto border border-[var(--charcoal)] bg-white p-1"
                    >
                      {filteredCorpora.length > 0 ? (
                        filteredCorpora.map((corpus) => (
                          <button
                            className="block w-full px-3 py-2 text-left text-sm \
                              text-[var(--charcoal)] hover:bg-[var(--hover-surface)]"
                            key={corpus.id}
                            onClick={() => {
                              setSelectedCorpusId(corpus.id);
                              setCorpusSearch(corpus.name);
                              setIsCorpusDropdownOpen(false);
                            }}
                            type="button"
                          >
                            {corpus.name}
                          </button>
                        ))
                      ) : (
                        <p className="px-3 py-3 text-xs text-[var(--muted-text)]">
                          No matching corpus. Upload documents before creating an index.
                        </p>
                      )}
                    </div>
                  ) : null}
                </div>
              </section>

              {/* Chunking and embedding form the two configurable transformation stages. */}
              <div className="grid gap-6 md:grid-cols-2">
                <section className="border border-[var(--border-strong)] bg-white">
                  <div className="border-b border-[var(--border-strong)] p-6">
                    <p
                      className="font-mono text-[10px] font-bold uppercase \
                        tracking-[0.12em] text-[var(--tone-black)]"
                    >
                      02 / Chunk
                    </p>
                    <h2 className="mt-3 text-xl font-semibold text-[var(--charcoal)]">
                      Shape the context
                    </h2>
                    <p className="mt-2 text-sm leading-5 text-[var(--tone-black)]">
                      Keep related ideas together without making retrieval noisy.
                    </p>
                  </div>

                  {/* Chunk controls use backend-provided strategies and numeric bounds. */}
                  <div className="space-y-5 bg-[var(--page-surface)] p-6">
                    <SelectControl
                      helperText={selectedStrategy?.description ?? undefined}
                      id="index-chunking-strategy"
                      label="Strategy"
                      onChange={(value) =>
                        setConfiguration((current) =>
                          current ? { ...current, chunkingStrategy: value } : current,
                        )
                      }
                      options={pipelineOptions.chunking.strategies}
                      value={configuration.chunkingStrategy}
                    />
                    <NumberControl
                      id="index-chunk-size"
                      label="Chunk size (tokens)"
                      onChange={(value) =>
                        setConfiguration((current) =>
                          current ? { ...current, chunkSizeTokens: value } : current,
                        )
                      }
                      setting={pipelineOptions.chunking.chunk_size_tokens}
                      value={configuration.chunkSizeTokens}
                    />
                    <NumberControl
                      helperText={
                        selectedStrategy?.supports_overlap
                          ? "Must remain smaller than the chunk size."
                          : "This strategy does not use overlap."
                      }
                      id="index-chunk-overlap"
                      label="Chunk overlap (tokens)"
                      onChange={(value) =>
                        setConfiguration((current) =>
                          current ? { ...current, chunkOverlapTokens: value } : current,
                        )
                      }
                      setting={pipelineOptions.chunking.chunk_overlap_tokens}
                      value={configuration.chunkOverlapTokens}
                    />
                  </div>
                </section>

                <section className="border border-[var(--border-strong)] bg-white">
                  <div className="border-b border-[var(--border-strong)] p-6">
                    <p
                      className="font-mono text-[10px] font-bold uppercase \
                        tracking-[0.12em] text-[var(--tone-black)]"
                    >
                      03 / Embed
                    </p>
                    <h2 className="mt-3 text-xl font-semibold text-[var(--charcoal)]">
                      Map the meaning
                    </h2>
                    <p className="mt-2 text-sm leading-5 text-[var(--tone-black)]">
                      Choose how chunks and questions share one vector space.
                    </p>
                  </div>

                  {/* Embedding controls keep models scoped to their selected provider. */}
                  <div className="space-y-5 bg-[var(--page-surface)] p-6">
                    <SelectControl
                      id="index-embedding-provider"
                      label="Provider"
                      onChange={(value) => {
                        const provider = pipelineOptions.embedding.providers.find(
                          (candidate) => candidate.value === value,
                        );

                        // Reset the model so the provider and model remain compatible.
                        setConfiguration((current) =>
                          current && provider
                            ? {
                                ...current,
                                embeddingProvider: value,
                                embeddingModel: provider.models[0].value,
                              }
                            : current,
                        );
                      }}
                      options={pipelineOptions.embedding.providers}
                      value={configuration.embeddingProvider}
                    />
                    <SelectControl
                      id="index-embedding-model"
                      label="Model"
                      onChange={(value) =>
                        setConfiguration((current) =>
                          current ? { ...current, embeddingModel: value } : current,
                        )
                      }
                      options={embeddingProvider?.models ?? []}
                      value={configuration.embeddingModel}
                    />
                    <SelectControl
                      id="index-distance-metric"
                      label="Distance metric"
                      onChange={(value) =>
                        setConfiguration((current) =>
                          current ? { ...current, distanceMetric: value } : current,
                        )
                      }
                      options={pipelineOptions.embedding.distance_metrics}
                      value={configuration.distanceMetric}
                    />
                  </div>
                </section>
              </div>

              {/* The identity strip is the final checkpoint before creating the artifact. */}
              <section className="border-2 border-[var(--charcoal)] bg-white p-5 sm:p-6">
                <div className="flex flex-col gap-5 lg:flex-row lg:items-end">
                  <label className="min-w-0 flex-1" htmlFor="index-name">
                    <span
                      className="block font-mono text-[10px] font-bold uppercase \
                        tracking-[0.12em] text-[var(--charcoal)]"
                    >
                      Index name (required)
                    </span>
                    <input
                      aria-describedby={
                        isNameTouched && !hasValidName ? "index-name-error" : undefined
                      }
                      aria-invalid={isNameTouched && !hasValidName}
                      className={`mt-2 w-full rounded-sm border bg-[var(--page-surface)] px-3 \
                        py-3 font-mono text-sm text-[var(--charcoal)] outline-none \
                        focus:ring-1 focus:ring-[var(--charcoal)] ${
                          isNameTouched && !hasValidName
                            ? "border-[var(--toast-error-text)]"
                            : "border-[var(--border-strong)] focus:border-[var(--charcoal)]"
                        }`}
                      id="index-name"
                      maxLength={100}
                      onBlur={() => setIsNameTouched(true)}
                      onChange={(event) => {
                        setIndexName(event.target.value);
                        setBuildNotice(null);
                      }}
                      placeholder="e.g. product-docs · recursive-800 · nomic"
                      ref={indexNameInputRef}
                      required
                      type="text"
                      value={indexName}
                    />
                    {isNameTouched && !hasValidName ? (
                      <span
                        className="mt-2 block text-xs text-[var(--toast-error-text)]"
                        id="index-name-error"
                        role="alert"
                      >
                        Enter a name so this index can be identified in experiments.
                      </span>
                    ) : (
                      <span className="mt-2 block text-xs text-[var(--muted-text)]">
                        This name will identify the prepared index in experiment selectors.
                      </span>
                    )}
                  </label>

                  <button
                    className="flex min-h-11 items-center justify-center gap-2 rounded-sm \
                      bg-[var(--charcoal)] px-5 text-sm font-semibold text-white \
                      transition-colors enabled:hover:bg-[var(--primary-hover)] \
                      disabled:cursor-not-allowed disabled:opacity-40"
                    disabled={!canInitialize}
                    type="submit"
                  >
                    <FiDatabase aria-hidden="true" className="size-4" />
                    Initialize build
                  </button>
                </div>

                {/* The notice reports frontend readiness without claiming persistence. */}
                {buildNotice ? (
                  <p
                    className="mt-5 border-t border-[var(--border-subtle)] pt-4 text-xs \
                      text-[var(--tone-black)]"
                    aria-live="polite"
                  >
                    {buildNotice}
                  </p>
                ) : null}
              </section>
            </form>
          ) : null}
        </div>
      </WorkbenchGridCanvas>
    </main>
  );
}
