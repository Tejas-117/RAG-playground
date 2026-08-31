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
import apiClient, { isAxiosError, isCancel } from "@/lib/axios";
import {
  createPreparedIndex,
  listPreparedIndexes,
} from "@/lib/prepared-index-api";
import { type CorpusOption, parseCorpora } from "@/validation/corpora";
import {
  type PipelineOptions,
  parsePipelineOptions,
} from "@/validation/pipeline-options";
import {
  type PreparedIndex,
  type PreparedIndexCreateRequest,
  preparedIndexCreateRequestSchema,
  parsePreparedIndexApiError,
} from "@/validation/prepared-indexes";

/** Poll active preparation jobs once per second without producing fake progress. */
const PREPARED_INDEX_POLL_INTERVAL_MS = 1_000;

/** Eight characters keep stable IDs recognizable without overwhelming list rows. */
const PREPARED_INDEX_ID_DISPLAY_LENGTH = 8;

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
 * Replaces one prepared-index lifecycle record while preserving newest-first order.
 *
 * @param indexes - Current validated prepared-index inventory.
 * @param replacement - Newer lifecycle representation of one inventory item.
 * @returns Updated inventory with the replacement inserted or substituted.
 */
function replacePreparedIndex(
  indexes: PreparedIndex[],
  replacement: PreparedIndex,
): PreparedIndex[] {
  const existingIndex = indexes.findIndex((index) => index.id === replacement.id);

  // Newly-created records belong at the front of the newest-first inventory.
  if (existingIndex === -1) {
    return [replacement, ...indexes];
  }

  // Preserve the backend list order while replacing only the matching identity.
  return indexes.map((index) =>
    index.id === replacement.id ? replacement : index,
  );
}

/**
 * Formats a backend UTC timestamp for compact local display.
 *
 * @param timestamp - ISO-like timestamp returned by the prepared-index API.
 * @returns Localized date and time, or the original value when parsing fails.
 */
function formatPreparedIndexTimestamp(timestamp: string): string {
  const date = new Date(timestamp);

  // Retain the backend value if a future timestamp format is not browser-readable.
  if (Number.isNaN(date.getTime())) {
    return timestamp;
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

/**
 * Converts editable form state into the exact prepared-index request contract.
 *
 * @param name - Required user-facing prepared-index label.
 * @param corpusId - Stable selected corpus identifier.
 * @param configuration - Editable chunking and embedding values.
 * @returns A validated request payload, or null when local values are invalid.
 */
function createPreparedIndexPayload(
  name: string,
  corpusId: string,
  configuration: IndexConfiguration,
): PreparedIndexCreateRequest | null {
  const result = preparedIndexCreateRequestSchema.safeParse({
    name,
    corpus_id: corpusId,
    configuration: {
      chunking: {
        strategy: configuration.chunkingStrategy,
        chunk_size_tokens: Number(configuration.chunkSizeTokens),
        chunk_overlap_tokens: Number(configuration.chunkOverlapTokens),
      },
      embedding: {
        provider: configuration.embeddingProvider,
        model: configuration.embeddingModel,
        distance_metric: configuration.distanceMetric,
      },
    },
  });

  // Keep invalid numeric or identifier state on the client instead of sending it.
  if (!result.success) {
    return null;
  }

  return result.data;
}

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
 * Renders the prepared-index inventory as a compact operational table.
 *
 * @param props - Validated indexes, corpus labels, and live refresh state.
 * @returns A table that exposes index identity, provenance, model, and lifecycle.
 */
function PreparedIndexInventory({
  preparedIndexes,
  corpora,
  isUpdating,
}: {
  preparedIndexes: PreparedIndex[];
  corpora: CorpusOption[];
  isUpdating: boolean;
}) {
  return (
    /* The inventory is the page's primary surface whenever prepared indexes exist. */
    <section
      className="mx-auto mt-8 max-w-[74rem] overflow-hidden border \
        border-[var(--border-strong)] bg-white"
    >
      {/* The inventory header identifies the artifact type and live polling state. */}
      <div
        className="flex items-center justify-between gap-4 border-b \
          border-[var(--border-strong)] p-5 sm:p-6"
      >
        <div>
          <p
            className="font-mono text-[10px] font-bold uppercase tracking-[0.12em] \
              text-[var(--tone-black)]"
          >
            Prepared indexes
          </p>
          <h2 className="mt-2 text-xl font-semibold text-[var(--charcoal)]">
            Recent builds
          </h2>
        </div>

        {isUpdating ? (
          <span
            className="flex items-center gap-2 font-mono text-[10px] uppercase \
              tracking-[0.1em] text-[var(--tone-black)]"
          >
            <FiRefreshCw aria-hidden="true" className="size-3.5 animate-spin" />
            Updating
          </span>
        ) : null}
      </div>

      {/* Horizontal overflow preserves the table hierarchy on narrow viewports. */}
      <div className="overflow-x-auto">
        <table className="w-full min-w-[54rem] border-collapse text-left">
          {/* Column labels mirror the provenance needed to choose an index later. */}
          <thead className="bg-[var(--panel-surface)]">
            <tr
              className="border-b border-[var(--border-strong)] font-mono text-[10px] \
                font-bold uppercase tracking-[0.1em] text-[var(--tone-black)]"
            >
              <th className="px-5 py-3 font-inherit sm:px-6">Name</th>
              <th className="px-5 py-3 font-inherit sm:px-6">Corpus</th>
              <th className="px-5 py-3 font-inherit sm:px-6">Embedding</th>
              <th className="px-5 py-3 font-inherit sm:px-6">Status</th>
              <th className="px-5 py-3 font-inherit sm:px-6">Created</th>
            </tr>
          </thead>

          {/* Every row represents one durable prepared-index identity. */}
          <tbody className="divide-y divide-[var(--border-subtle)]">
            {preparedIndexes.map((preparedIndex) => {
              // Resolve the stable corpus ID to the user-facing name loaded with the page.
              const corpusName =
                corpora.find((corpus) => corpus.id === preparedIndex.corpus_id)?.name ??
                `Corpus ${preparedIndex.corpus_id.slice(
                  0,
                  PREPARED_INDEX_ID_DISPLAY_LENGTH,
                )}`;

              return (
                <tr
                  className="transition-colors hover:bg-[var(--page-surface)]"
                  key={preparedIndex.id}
                >
                  {/* The name cell retains stable identity and any safe terminal error. */}
                  <td className="px-5 py-4 align-top sm:px-6">
                    <p className="max-w-56 truncate font-mono text-xs text-[var(--charcoal)]">
                      {preparedIndex.name}
                    </p>
                    <p className="mt-1 font-mono text-[9px] text-[var(--muted-text)]">
                      ID {preparedIndex.id.slice(0, PREPARED_INDEX_ID_DISPLAY_LENGTH)}
                    </p>
                    {preparedIndex.error ? (
                      <p className="mt-2 max-w-64 text-xs text-[var(--toast-error-text)]">
                        {preparedIndex.error.message}
                      </p>
                    ) : null}
                  </td>

                  {/* Corpus provenance prevents similarly named indexes becoming ambiguous. */}
                  <td className="px-5 py-4 text-xs text-[var(--tone-black)] sm:px-6">
                    {corpusName}
                  </td>

                  {/* Embedding identity exposes the compatibility boundary at a glance. */}
                  <td className="px-5 py-4 align-top sm:px-6">
                    <p className="font-mono text-xs text-[var(--charcoal)]">
                      {preparedIndex.embedding.model}
                    </p>
                    <p className="mt-1 text-[10px] text-[var(--muted-text)]">
                      {preparedIndex.embedding.provider}
                      {" · "}
                      {preparedIndex.embedding.distance_metric}
                    </p>
                  </td>

                  {/* Lifecycle state stays textual and never implies fabricated progress. */}
                  <td className="px-5 py-4 align-top sm:px-6">
                    <span className="flex items-center gap-2 text-xs capitalize">
                      <span
                        aria-hidden="true"
                        className={`size-2 rounded-full ${
                          preparedIndex.status === "failed"
                            ? "bg-[var(--toast-error-text)]"
                            : preparedIndex.status === "ready"
                              ? "bg-[var(--toast-success-text)]"
                              : "animate-pulse bg-[var(--tone-black)]"
                        }`}
                      />
                      {preparedIndex.status}
                    </span>
                    <p className="mt-1 text-[10px] capitalize text-[var(--muted-text)]">
                      {preparedIndex.current_stage ??
                        `${preparedIndex.embedding.vector_count ?? 0} vectors`}
                    </p>
                  </td>

                  {/* Localized creation time gives duplicate names a visible chronology. */}
                  <td
                    className="whitespace-nowrap px-5 py-4 font-mono text-[10px] \
                      text-[var(--muted-text)] sm:px-6"
                  >
                    {formatPreparedIndexTimestamp(preparedIndex.created_at)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/**
 * Renders the connected workspace used to configure a reusable vector index.
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

  // Stores validated named indexes returned by the backend in newest-first order.
  const [preparedIndexes, setPreparedIndexes] = useState<PreparedIndex[]>([]);

  // Controls whether the index configuration workbench is visible.
  const [isCreateFormOpen, setIsCreateFormOpen] = useState(false);

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

  // Stores the latest successful creation or lifecycle notice.
  const [buildNotice, setBuildNotice] = useState<string | null>(null);

  // Stores a safe request or polling failure without replacing the loaded form.
  const [buildError, setBuildError] = useState<string | null>(null);

  // Indicates that a create request is awaiting its durable backend response.
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Increments when the user asks to retry initial API requests.
  const [loadAttempt, setLoadAttempt] = useState(0);

  // Gives the header action direct focus access to the required identity field.
  const indexNameInputRef = useRef<HTMLInputElement | null>(null);

  // Gives the page action a stable scroll target when it reveals the builder.
  const creationWorkspaceRef = useRef<HTMLDivElement | null>(null);

  // Distinguishes a button-triggered reveal from the automatic empty-state reveal.
  const shouldScrollToCreationRef = useRef(false);

  // Loads options, corpora, and prepared-index inventory whenever the page retries.
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
        const [optionsResponse, corporaResponse, indexes] = await Promise.all([
          apiClient.get<unknown>("/pipeline/options", {
            signal: abortController.signal,
          }),
          apiClient.get<unknown>("/corpora/", {
            signal: abortController.signal,
          }),
          listPreparedIndexes(undefined, abortController.signal),
        ]);
        const validatedOptions = parsePipelineOptions(optionsResponse.data);
        const validatedCorpora = parseCorpora(corporaResponse.data);

        setPipelineOptions(validatedOptions);
        setConfiguration(createDefaultIndexConfiguration(validatedOptions));
        setCorpora(validatedCorpora);
        setPreparedIndexes(indexes);

        // An empty inventory needs the creation form as its immediate next action.
        if (indexes.length === 0) {
          setIsCreateFormOpen(true);
        }

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

  // Scrolls the newly revealed builder into view only after the user requests it.
  useEffect(() => {
    // Automatic empty-state reveals should not unexpectedly move browser focus.
    if (!isCreateFormOpen || !shouldScrollToCreationRef.current) {
      return;
    }

    shouldScrollToCreationRef.current = false;
    const frameId = window.requestAnimationFrame(() => {
      creationWorkspaceRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    });

    return () => {
      window.cancelAnimationFrame(frameId);
    };
  }, [isCreateFormOpen]);

  // Active durable jobs keep the inventory polling until all reach terminal state.
  const hasActivePreparedIndexes = preparedIndexes.some(
    (index) => index.status === "pending" || index.status === "running",
  );

  // Polls only while the loaded inventory contains pending or running preparation.
  useEffect(() => {
    // Avoid timers and requests when every visible preparation is terminal.
    if (!hasActivePreparedIndexes) {
      return;
    }

    const abortController = new AbortController();
    let timeoutId: ReturnType<typeof setTimeout> | undefined;

    /**
     * Refreshes all prepared indexes so concurrent jobs remain correctly ordered.
     *
     * @returns A promise resolved after inventory state or a safe error is updated.
     */
    async function pollPreparedIndexes(): Promise<void> {
      try {
        const indexes = await listPreparedIndexes(
          undefined,
          abortController.signal,
        );
        setPreparedIndexes(indexes);
        setBuildError(null);
      } catch (error) {
        // Navigation and effect cleanup intentionally cancel an in-flight poll.
        if (isCancel(error)) {
          return;
        }

        setBuildError("Index status could not be refreshed. Retrying automatically.");
      }

      // Continue only while this effect instance still owns the polling lifecycle.
      if (!abortController.signal.aborted) {
        timeoutId = setTimeout(
          () => void pollPreparedIndexes(),
          PREPARED_INDEX_POLL_INTERVAL_MS,
        );
      }
    }

    timeoutId = setTimeout(
      () => void pollPreparedIndexes(),
      PREPARED_INDEX_POLL_INTERVAL_MS,
    );

    return () => {
      abortController.abort();

      // Clear the scheduled request when all jobs finish or the page unmounts.
      if (timeoutId !== undefined) {
        clearTimeout(timeoutId);
      }
    };
  }, [hasActivePreparedIndexes]);

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

  // Convert current values once so button state and submission use identical validation.
  const preparedIndexPayload = configuration
    ? createPreparedIndexPayload(indexName, selectedCorpusId, configuration)
    : null;

  // Every source and configuration choice must exist before the action becomes available.
  const canInitialize = Boolean(
    hasValidName &&
      preparedIndexPayload &&
      !isLoading &&
      !isSubmitting &&
      !loadError,
  );

  /**
   * Opens or closes the index builder from the page-level action.
   *
   * @returns Nothing. Builder visibility and reveal scrolling are updated in place.
   */
  function toggleCreateForm(): void {
    // Existing inventories may collapse the builder back to the primary list view.
    if (isCreateFormOpen && preparedIndexes.length > 0) {
      setIsCreateFormOpen(false);
      return;
    }

    // Mark explicit reveals so the following render scrolls to the workbench once.
    shouldScrollToCreationRef.current = true;
    setIsCreateFormOpen(true);
  }

  /**
   * Validates and submits one durable prepared-index request.
   *
   * @param event - Native form submission event from the build action.
   * @returns A promise resolved after creation state or a safe error is shown.
   */
  async function handleInitialize(
    event: React.FormEvent<HTMLFormElement>,
  ): Promise<void> {
    event.preventDefault();
    setIsNameTouched(true);
    setBuildNotice(null);
    setBuildError(null);

    // Keep focus on the missing required value instead of submitting incomplete data.
    if (!hasValidName) {
      indexNameInputRef.current?.focus();
      return;
    }

    // Invalid numeric or compatibility-shaped form values remain client-side.
    if (!preparedIndexPayload || !canInitialize) {
      setBuildError("Review the corpus, chunk size, overlap, and embedding settings.");
      return;
    }

    setIsSubmitting(true);

    try {
      const preparedIndex = await createPreparedIndex(preparedIndexPayload);
      setPreparedIndexes((current) =>
        replacePreparedIndex(current, preparedIndex),
      );
      setBuildNotice(
        `“${preparedIndex.name}” was queued. Status will update automatically.`,
      );
      setIndexName("");
      setIsNameTouched(false);
    } catch (error) {
      // Prefer the backend's safe structured message for rejected requests.
      const apiMessage = isAxiosError(error)
        ? parsePreparedIndexApiError(error.response?.data)
        : null;
      setBuildError(
        apiMessage ??
          (error instanceof Error
            ? error.message
            : "The prepared index could not be created."),
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="grid min-h-screen lg:grid-cols-[17.5rem_minmax(0,1fr)]">
      <WorkbenchSidebar activeLabel="Indexes" />

      {/* The grid canvas contains the complete index preparation workspace. */}
      <WorkbenchGridCanvas className="min-h-screen px-5 py-8 sm:px-8 lg:px-12 lg:py-10">
        {/* The page header frames the inventory and opens its secondary creation mode. */}
        <header
          className="mx-auto flex max-w-[74rem] flex-col gap-6 border-b-2 \
            border-[var(--charcoal)] pb-7 sm:flex-row sm:items-end sm:justify-between"
        >
          <div>
            <p
              className="font-mono text-[10px] font-bold uppercase tracking-[0.16em] \
                text-[var(--tone-black)]"
            >
              Prepared indexes
            </p>
            <h1
              className="mt-3 max-w-3xl text-3xl font-semibold tracking-[-0.035em] \
                text-[var(--charcoal)] sm:text-4xl"
            >
              Define the data foundation.
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-[var(--tone-black)]">
              Build and monitor reusable vector foundations for retrieval experiments.
            </p>
          </div>

          {/* Existing inventories use this control to enter or leave creation mode. */}
          {!isLoading && !loadError && preparedIndexes.length > 0 ? (
            <button
              aria-controls="index-creation-workspace"
              aria-expanded={isCreateFormOpen}
              className="flex w-fit items-center gap-2 rounded-sm bg-[var(--charcoal)] px-4 \
                py-3 text-sm font-semibold text-white transition-colors \
                hover:bg-[var(--primary-hover)] focus-visible:outline-2 \
                focus-visible:outline-offset-2 focus-visible:outline-[var(--charcoal)]"
              onClick={toggleCreateForm}
              type="button"
            >
              {isCreateFormOpen ? (
                <FiX aria-hidden="true" className="size-5" />
              ) : (
                <FiPlus aria-hidden="true" className="size-5" />
              )}
              {isCreateFormOpen ? "Close builder" : "Create index"}
            </button>
          ) : null}
        </header>

        {/* Loading, failure, and creation states occupy the focused workbench region. */}
        {isLoading || loadError || isCreateFormOpen ? (
          <div
            className="index-builder-reveal mx-auto mt-8 grid max-w-[74rem] scroll-mt-8 \
              items-start gap-6 lg:grid-cols-[15rem_minmax(0,1fr)]"
            id="index-creation-workspace"
            ref={creationWorkspaceRef}
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
                      onChange={(value) => {
                        const strategy = pipelineOptions.chunking.strategies.find(
                          (candidate) => candidate.value === value,
                        );

                        // Clear overlap when the selected strategy cannot apply it.
                        setConfiguration((current) =>
                          current
                            ? {
                                ...current,
                                chunkingStrategy: value,
                                chunkOverlapTokens: strategy?.supports_overlap
                                  ? current.chunkOverlapTokens
                                  : "0",
                              }
                            : current,
                        );
                      }}
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
                        setBuildError(null);
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
                    aria-busy={isSubmitting}
                    className="flex min-h-11 items-center justify-center gap-2 rounded-sm \
                      bg-[var(--charcoal)] px-5 text-sm font-semibold text-white \
                      transition-colors enabled:hover:bg-[var(--primary-hover)] \
                      disabled:cursor-not-allowed disabled:opacity-40"
                    disabled={!canInitialize}
                    type="submit"
                  >
                    {isSubmitting ? (
                      <FiRefreshCw
                        aria-hidden="true"
                        className="size-4 animate-spin"
                      />
                    ) : (
                      <FiDatabase aria-hidden="true" className="size-4" />
                    )}
                    {isSubmitting ? "Queueing…" : "Initialize build"}
                  </button>
                </div>

                {/* Creation feedback reflects only validated backend responses. */}
                {buildNotice ? (
                  <p
                    className="mt-5 border-t border-[var(--border-subtle)] pt-4 text-xs \
                      text-[var(--tone-black)]"
                    aria-live="polite"
                  >
                    {buildNotice}
                  </p>
                ) : null}

                {buildError ? (
                  <p
                    className="mt-5 border-t border-[var(--toast-error-border)] pt-4 \
                      text-xs text-[var(--toast-error-text)]"
                    aria-live="polite"
                    role="alert"
                  >
                    {buildError}
                  </p>
                ) : null}
              </section>

            </form>
          ) : null}
          </div>
        ) : null}

        {/* Existing builds remain visible independently of the optional creation form. */}
        {!isLoading && !loadError && preparedIndexes.length > 0 ? (
          <PreparedIndexInventory
            corpora={corpora}
            isUpdating={hasActivePreparedIndexes}
            preparedIndexes={preparedIndexes}
          />
        ) : null}
      </WorkbenchGridCanvas>
    </main>
  );
}
