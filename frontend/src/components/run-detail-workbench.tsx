"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  FiAlertCircle,
  FiArrowLeft,
  FiCheck,
  FiClock,
  FiCopy,
  FiDatabase,
  FiFileText,
  FiSearch,
} from "react-icons/fi";
import WorkbenchSidebar from "@/components/workbench-sidebar";
import WorkbenchGridCanvas from "@/components/workbench-grid-canvas";
import {
  EvaluationSummaryCard,
  QuestionEvaluationPanel,
  RetrievalEvaluationWorkspace,
} from "@/components/retrieval-evaluation-workspace";
import { useRunDetail } from "@/lib/use-run-detail";
import { useRetrievalEvaluations } from "@/lib/use-retrieval-evaluations";
import type { BenchmarkRunDetail } from "@/validation/benchmark-runs";
import type { RetrievalEvaluationDetail } from "@/validation/benchmark-runs";
import styles from "./run-detail-workbench.module.css";

type RunQuestion = BenchmarkRunDetail["examples"][number];
type RunChunk = NonNullable<RunQuestion["retrieval"]>["chunks"][number];
type QuestionFilter = "all" | RunQuestion["status"];
type EvaluationQuestion = RetrievalEvaluationDetail["questions"][number];

/** Format a recorded millisecond duration for a compact data label. */
function formatDuration(value: number | null): string {
  // No result means no measured duration; zero is a real measurement.
  if (value === null) return "—";
  return value < 1000 ? `${value} ms` : `${(value / 1000).toFixed(2)} s`;
}

/** Label a lifecycle value as it appears in the question navigator. */
function statusLabel(status: RunQuestion["status"]): string {
  // Keep concise status words consistent across navigator and inspector.
  return status.charAt(0).toUpperCase() + status.slice(1);
}

/** Format wall-clock execution time without confusing it with saved stage durations. */
function formatElapsedTime(startedAt: string, now: number): string {
  // Clock skew must never show negative elapsed time.
  const seconds = Math.max(0, Math.floor((now - Date.parse(startedAt)) / 1000));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;

  return hours > 0
    ? `${hours}h ${String(minutes).padStart(2, "0")}m ${String(remainder).padStart(2, "0")}s`
    : `${minutes}m ${String(remainder).padStart(2, "0")}s`;
}

/** Render a ranked piece of source evidence with its prompt inclusion and provenance. */
function EvidenceChunk({
  chunk,
  metric,
  contextOrder,
  isRelevantMatch,
}: {
  chunk: RunChunk;
  metric: string;
  contextOrder: number | null;
  isRelevantMatch: boolean;
}) {
  return (
    <article className={styles.chunk} data-relevant-match={isRelevantMatch}>
      {/* Rank, distance, and prompt inclusion are separate pieces of evidence. */}
      <div className={styles.chunkHead}>
        <strong className={styles.rank}>
          Rank {chunk.rank}
        </strong>
        <span className={styles.mono}>
          Raw {metric.replaceAll("_", " ")} distance {chunk.raw_distance.toFixed(3)}
        </span>
        {contextOrder !== null && (
          <span className={styles.contextBadge}>
            Included in prompt · context {contextOrder}
          </span>
        )}
        {isRelevantMatch && (
          <span className={styles.relevantBadge}>
            Relevant label match
          </span>
        )}
      </div>
      {/* Source labels keep the excerpt tied to the uploaded document. */}
      <p className={styles.chunkSource}>
        <FiFileText aria-hidden="true" />
        <span>{chunk.original_filename}</span>
        {chunk.page_start !== null && (
          <span>· Page {chunk.page_start}{chunk.page_end !== chunk.page_start &&
            chunk.page_end !== null ? `–${chunk.page_end}` : ""}</span>
        )}
      </p>
      <p className={styles.chunkText}>
        {chunk.text}
      </p>
      {/* Full metadata stays available without crowding the ranking list. */}
      <details className={styles.chunkDetails}>
        <summary>Source provenance</summary>
        <dl className={styles.provenance}>
          <div><dt>Chunk ID</dt><dd>{chunk.chunk_id}</dd></div>
          <div><dt>Document ID</dt><dd>{chunk.source_document_id}</dd></div>
          <div>
            <dt>Character offsets</dt>
            <dd>{chunk.character_start_offset ?? "—"}–
              {chunk.character_end_offset ?? "—"}</dd>
          </div>
          <div>
            <dt>Prompt context</dt>
            <dd>
              {contextOrder === null ? "Not included" : `Position ${contextOrder}`}
            </dd>
          </div>
        </dl>
      </details>
    </article>
  );
}

/** Show the selected question's input, answer, retrieval, and saved-stage states. */
function QuestionInspector({
  question,
  topK,
  evaluation,
  evaluationQuestion,
}: {
  question: RunQuestion;
  topK: number;
  evaluation: RetrievalEvaluationDetail | null;
  evaluationQuestion: EvaluationQuestion | null;
}) {
  // A generation can be missing while retrieval has already been saved.
  const retrieval = question.retrieval;
  const generation = question.generation;
  return (
    <div className={styles.inspector}>
      {/* The original question and reference remain distinct from model output. */}
      <section className={styles.panel} aria-labelledby="question-heading">
        <div className={styles.panelHead}>
          <h2 id="question-heading">Question {question.ordinal + 1}</h2>
          <span className={styles.mono}>{question.example_id}</span>
        </div>
        <p className={styles.questionText}>{question.question}</p>
        <div className={styles.reference}>
          <span className={styles.eyebrow}>Dataset reference · not generated</span>
          <p>{question.reference_answer ??
            "No reference answer was supplied for this question."}</p>
        </div>
      </section>

      {/* The answer panel reflects the selected example's persisted lifecycle. */}
      <section className={styles.panel} aria-labelledby="generation-heading">
        <div className={styles.panelHead}>
          <h2 id="generation-heading">Generated answer</h2>
          <span className={styles.sectionMeta}>
            {generation ? formatDuration(generation.duration_ms) : statusLabel(question.status)}
          </span>
        </div>
        {generation?.answer ? (
          <>
            <p className={styles.answer}>{generation.answer}</p>
            <div className={styles.provenanceLine}>
              <span>{generation.provider} / {generation.provider_model ?? generation.model}</span>
              <span>Finish: {generation.finish_reason ?? "—"}</span>
              <span>
                {generation.prompt_tokens ?? "—"} input ·
                {" "}{generation.completion_tokens ?? "—"} output
              </span>
            </div>
            <details className={styles.chunkDetails}>
              <summary>Generation provenance</summary>
              <dl className={styles.provenance}>
                <div>
                  <dt>Prompt template</dt>
                  <dd>{generation.prompt_template_version ?? "—"}</dd>
                </div>
                <div>
                  <dt>Provider policy</dt>
                  <dd>{generation.provider_policy_version ?? "—"}</dd>
                </div>
                <div><dt>Provider called</dt><dd>{generation.provider_called === null ? "—" :
                  generation.provider_called ? "Yes" : "No"}</dd></div>
              </dl>
            </details>
          </>
        ) : (
          <p className={styles.pendingMessage}>
            {question.status === "running" ?
              question.current_stage === "generation" ?
                "Retrieval is complete. This question is generating an answer." :
                "Retrieval is in progress." :
              question.status === "failed" ?
                question.error?.message ?? "Generation stopped before an answer was saved." :
                question.status === "completed" ?
                  "No answer text was saved for this question." :
                  "This question has not started. No answer is available yet."}
          </p>
        )}
      </section>

      {/* Evaluation connects selected metrics to labels and preserved chunk ranks. */}
      <QuestionEvaluationPanel
        chunks={retrieval?.chunks ?? []}
        evaluation={evaluation}
        outcome={evaluationQuestion}
        topK={topK}
      />

      {/* Ranked evidence can exist even when generation has not completed. */}
      <section className={styles.panel} aria-labelledby="retrieval-heading">
        <div className={styles.panelHead}>
          <h2 id="retrieval-heading">Retrieval evidence</h2>
          <span className={styles.sectionMeta}>
            Top K {retrieval?.requested_top_k ?? topK} ·
            {" "}{retrieval?.returned_count ?? 0} returned ·
            {" "}{formatDuration(retrieval?.duration_ms ?? null)}
          </span>
        </div>
        <p className={styles.distanceNote}>
          Values are raw {retrieval?.distance_metric.replaceAll("_", " ") ?? "vector"} distances,
          not relevance percentages. Prompt badges show
          which retrieved chunks were included in generation.
        </p>
        {retrieval && retrieval.chunks.length > 0 ? (
          <div className={styles.chunkList}>
            {retrieval.chunks.map((chunk) => (
              <EvidenceChunk
                key={chunk.chunk_id}
                chunk={chunk}
                metric={retrieval.distance_metric}
                contextOrder={generation?.context_chunks.find((item) =>
                  item.chunk_id === chunk.chunk_id && item.retrieval_rank === chunk.rank
                )?.ordinal ?? null}
                isRelevantMatch={evaluationQuestion?.matching_ranks.includes(chunk.rank) ?? false}
              />
            ))}
          </div>
        ) : (
          <p className={styles.pendingMessage}>
            {retrieval === null ?
              "Retrieval has not completed for this question." :
              "Retrieval completed without returning any chunks."}
          </p>
        )}
      </section>
    </div>
  );
}

/** Present the validated run snapshot with local question search and filtering. */
export default function RunDetailWorkbench({ requestedRunId }: { requestedRunId: string }) {
  // The detail hook owns request cancellation, polling, and retry.
  const { run, corpusName, loading, error, retry } = useRunDetail(requestedRunId);

  // The latest marker restarts history loading when automatic scoring changes lifecycle state.
  const latestEvaluationMarker = run?.latest_evaluation
    ? `${run.latest_evaluation.id}:${run.latest_evaluation.status}`
    : "";

  // Evaluation history and selected evidence remain independent from benchmark polling.
  const evaluationState = useRetrievalEvaluations(
    requestedRunId,
    run !== null,
    latestEvaluationMarker,
  );

  // Search narrows the question navigator without a network request.
  const [search, setSearch] = useState("");

  // The status filter controls which questions appear in the navigator.
  const [filter, setFilter] = useState<QuestionFilter>("all");

  // Persist selection across filtering so the inspector does not jump unexpectedly.
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Accessible feedback reports whether copying the run ID succeeded.
  const [notice, setNotice] = useState("");

  // Advance the active run's elapsed label without requesting extra snapshots.
  const [now, setNow] = useState(() => Date.now());

  // Only active runs need a one-second display clock; polling remains in the data hook.
  useEffect(() => {
    if (run?.status !== "running") return;

    const timer = setInterval(() => {
      // Hidden tabs do not need display updates; refresh immediately on return.
      if (!document.hidden) setNow(Date.now());
    }, 1000);

    return () => clearInterval(timer);
  }, [run?.status]);

  // A missing selection falls back to the first saved example on every fresh run.
  const selected = run?.examples.find((question) => question.id === selectedId)
    ?? run?.examples[0];

  // Match evaluation evidence by immutable dataset example ID, not display position.
  const selectedEvaluationQuestion = selected
    ? evaluationState.detail?.questions.find(
      (question) => question.example_id === selected.example_id,
    ) ?? null
    : null;

  // Apply both local controls while preserving the dataset's original question order.
  const visibleQuestions = (run?.examples ?? []).filter((question) => (
    (filter === "all" || question.status === filter)
    && question.question.toLowerCase().includes(search.toLowerCase())
  ));

  // Status counts describe all examples, independent of the active filter.
  const statuses: QuestionFilter[] = ["all", "completed", "running", "failed", "pending"];

  /** Copy the requested run ID and report browser clipboard failures. */
  async function copyRunId() {
    try {
      await navigator.clipboard.writeText(requestedRunId);
      setNotice("Run ID copied.");
    } catch {
      setNotice("Unable to copy the run ID.");
    }
  }

  return (
    <main className={styles.shell}>
      <WorkbenchSidebar activeLabel="Runs" />
      <WorkbenchGridCanvas className={styles.workspace}>
        {/* Request errors stay visible while any earlier validated snapshot remains usable. */}
        {error && (
          <div className={styles.preview} role="alert">
            {error} <button type="button" onClick={retry}>Retry</button>
          </div>
        )}

        {/* Initial and missing states cannot display invented example results. */}
        {!run && (
          <div className={styles.preview} role="status">
            {loading ? "Loading run details…" : "No run details are available."}
          </div>
        )}

        {run && (
          <>

        {/* Persisted execution failures remain distinct from request-refresh errors. */}
        {run.status === "failed" && (
          <div className={styles.runFailure} role="alert">
            <FiAlertCircle aria-hidden="true" />
            <span>
              {run.error?.message ?? "Execution failed."}
              {run.error?.code && ` (${run.error.code})`}
              {" "}Partial results remain available below.
            </span>
          </div>
        )}

        {/* Navigation and identity connect this inspection view to the runs inventory. */}
        <header className={styles.header}>
          <Link href="/runs" className={styles.back}>
            <FiArrowLeft aria-hidden="true" />
            Back to runs
          </Link>
          <div className={styles.headerMain}>
            <div>
              <p className={styles.eyebrow}>Run inspection</p>
              <div className={styles.titleLine}>
                <h1>Run {run.id}</h1>
                <button
                  type="button"
                  onClick={copyRunId}
                  aria-label="Copy run ID"
                  title="Copy run ID"
                >
                  <FiCopy aria-hidden="true" />
                </button>
              </div>
            </div>
            <span className={styles.status}>
              <span className={styles.statusDot} data-active={run.status === "running"} />
              {statusLabel(run.status)}
            </span>
          </div>
          <div className={styles.identity}>
            <span>Corpus: {corpusName}</span>
            <span>Index: {run.prepared_index_name}</span>
            <span>Dataset: {run.dataset_name}</span>
            <span>Created: {new Date(run.created_at).toLocaleString()}</span>
            {run.status === "running" && run.started_at && (
              <span className={styles.elapsed} role="timer">
                <FiClock aria-hidden="true" />
                Elapsed: {formatElapsedTime(run.started_at, now)}
              </span>
            )}
          </div>
        </header>

        {/* A compact metrics rail summarizes execution without implying answer quality. */}
        <section className={styles.metrics} aria-label="Run summary">
          <article className={styles.metricCard}>
            <span className={styles.eyebrow}>Question progress</span>
            <strong>
              {run.completed_examples} <small>/ {run.total_examples} completed</small>
            </strong>
            <progress max={run.total_examples} value={run.completed_examples}>
              {run.completed_examples} of {run.total_examples}
            </progress>
            <p>{run.running_examples} running · {run.failed_examples} failed ·
              {" "}{run.pending_examples} pending</p>
          </article>
          <article className={styles.metricCard}>
            <span className={styles.eyebrow}>Stage duration</span>
            <div className={styles.pairedMetrics}>
              <div>
                <strong>{formatDuration(run.metrics.average_retrieval_duration_ms)}</strong>
                <span>Retrieval avg</span>
                <small>{run.metrics.retrieval_result_count} results</small>
              </div>
              <div>
                <strong>{formatDuration(run.metrics.average_generation_duration_ms)}</strong>
                <span>Generation avg</span>
                <small>{run.metrics.generation_result_count} results</small>
              </div>
            </div>
          </article>
          <article className={styles.metricCard}>
            <span className={styles.eyebrow}>Reported token usage</span>
            <strong>{run.metrics.prompt_tokens ?? "—"} <small>in</small> /
              {" "}{run.metrics.completion_tokens ?? "—"} <small>out</small></strong>
            <p>Usage reported by {run.metrics.prompt_token_result_count} input /
              {" "}{run.metrics.completion_token_result_count} output of
              {" "}{run.metrics.generation_result_count} generations</p>
          </article>
          <EvaluationSummaryCard
            className={styles.metricCard}
            evaluation={evaluationState.attempts[0] ?? run.latest_evaluation}
            topK={run.configuration.retrieval.top_k}
          />
        </section>

        {/* An expandable snapshot follows the pipeline's real stage order. */}
        <details className={styles.configuration} open>
          <summary>
            <FiDatabase aria-hidden="true" />
            Saved configuration and provenance
            <span>Immutable snapshot</span>
          </summary>
          <div className={styles.configurationGrid}>
            <div><b>Preparation</b>
              <span>{run.configuration.chunking.strategy} ·
                {" "}{run.configuration.chunking.chunk_size_tokens} tokens ·
                {" "}{run.configuration.chunking.chunk_overlap_tokens} overlap</span>
              <span>{run.configuration.embedding.provider} /
                {" "}{run.configuration.embedding.model} ·
                {" "}{run.configuration.embedding.distance_metric}</span></div>
            <div><b>Retrieval</b><span>Vector search · top K
              {" "}{run.configuration.retrieval.top_k}</span>
              <span>Raw {run.configuration.embedding.distance_metric} distance</span></div>
            <div><b>Generation</b><span>{run.configuration.generation.provider} /
              {" "}{run.configuration.generation.model}</span>
              <span>Temperature {run.configuration.generation.temperature} · max output
                {" "}{run.configuration.generation.max_output_tokens}</span></div>
            <div><b>Evaluation targets</b>
              <span>{[
                ...run.configuration.evaluation.retrieval_metrics,
                ...run.configuration.evaluation.answer_metrics,
              ].join(", ") || "None configured"}</span></div>
          </div>
        </details>

        {/* The evaluation ledger keeps repeatable scoring separate from run execution. */}
        <RetrievalEvaluationWorkspace
          runStatus={run.status}
          savedMetrics={run.configuration.evaluation.retrieval_metrics}
          state={evaluationState}
          topK={run.configuration.retrieval.top_k}
        />

        {/* The question rail is the page's primary interaction and mirrors the Stitch layout. */}
        <div className={styles.questionGrid}>
          <section className={styles.navigator} aria-label="Execution questions">
            <div className={styles.navigatorHead}>
              <div className={styles.navigatorTitle}>
                <h2>Execution questions</h2>
                <span>{run.total_examples} total</span>
              </div>
              <label className={styles.search}>
                <FiSearch aria-hidden="true" />
                <input
                  aria-label="Search questions"
                  placeholder="Filter questions…"
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                />
              </label>
              <div className={styles.filterBar} aria-label="Filter questions by status">
                {statuses.map((status) => {
                  const count = status === "all" ? run.total_examples :
                    run.examples.filter((question) => question.status === status).length;

                  return (
                    <button
                      key={status}
                      type="button"
                      aria-pressed={filter === status}
                      onClick={() => setFilter(status)}
                    >
                      {status === "all" ? "All" : statusLabel(status)} ({count})
                    </button>
                  );
                })}
              </div>
            </div>
            <div className={styles.questionList}>
              {visibleQuestions.map((question) => (
                <button
                  className={styles.questionItem}
                  data-selected={selected?.id === question.id}
                  key={question.id}
                  type="button"
                  onClick={() => setSelectedId(question.id)}
                >
                  <span className={styles.questionItemTop}>
                    <b>Q{question.ordinal + 1}</b>
                    <span>{formatDuration(question.duration_ms)}</span>
                    <span className={styles.questionStatus} data-status={question.status}>
                      {statusLabel(question.status)}
                    </span>
                  </span>
                  <span className={styles.questionItemText}>{question.question}</span>
                  <span className={styles.questionItemFoot}>
                    {question.current_stage ? `Stage: ${question.current_stage}` :
                      question.status === "pending" ? "Waiting to start" : "Execution saved"}
                    {question.retrieval &&
                      ` · ${question.retrieval.returned_count ?? 0} retrieved`}
                    {question.generation?.answer && " · Answer saved"}
                  </span>
                </button>
              ))}
              {visibleQuestions.length === 0 && (
                <p className={styles.noMatch}>No questions match this search and status.</p>
              )}
            </div>
          </section>
          {selected && (
            <QuestionInspector
              evaluation={evaluationState.detail}
              evaluationQuestion={selectedEvaluationQuestion}
              question={selected}
              topK={run.configuration.retrieval.top_k}
            />
          )}
        </div>
        <p className={styles.srNotice} role="status">
          {notice && <><FiCheck aria-hidden="true" />{notice}</>}
        </p>
        <footer className={styles.footer}>
          <FiClock aria-hidden="true" />
          Execution and evaluation results are saved as independent snapshots.
        </footer>
          </>
        )}
      </WorkbenchGridCanvas>
    </main>
  );
}
