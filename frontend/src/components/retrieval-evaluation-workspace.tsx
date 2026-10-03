"use client";

import { useState } from "react";
import {
  FiAlertCircle,
  FiCheckCircle,
  FiClock,
  FiPlay,
  FiRefreshCw,
  FiTarget,
} from "react-icons/fi";
import type { EvaluationsState } from "@/lib/use-retrieval-evaluations";
import type {
  RetrievalEvaluationDetail,
  RetrievalEvaluationSummary,
  AnswerMetric,
  EvaluationMetric,
  RetrievalMetric,
} from "@/validation/benchmark-runs";
import styles from "./retrieval-evaluation-workspace.module.css";

/** Stable display order keeps metric controls and results visually consistent. */
const RETRIEVAL_METRICS: RetrievalMetric[] = [
  "hit_rate_at_k",
  "recall_at_k",
  "mrr",
];

/** Stable judge-metric order matches the experiment configuration catalog. */
const ANSWER_METRICS: AnswerMetric[] = [
  "groundedness",
  "answer_relevance",
  "answer_correctness",
];

type EvaluationQuestion = RetrievalEvaluationDetail["questions"][number];

type RankedChunk = {
  rank: number;
  chunk_id: string;
  source_document_id: string;
  original_filename: string;
  context_order?: number | null;
};

/**
 * Convert one metric identifier into its user-facing label at the run's saved K.
 *
 * @param metric - Stable retrieval metric identifier.
 * @param topK - Retrieval cutoff saved by the benchmark.
 * @returns A compact metric label containing K where applicable.
 */
export function metricLabel(metric: RetrievalMetric, topK: number): string {
  // K belongs to the immutable benchmark configuration used to retrieve the chunks.
  if (metric === "hit_rate_at_k") {
    return `Hit Rate@${topK}`;
  }

  if (metric === "recall_at_k") {
    return `Recall@${topK}`;
  }

  return "MRR";
}

/** Convert any supported metric identifier into a concise audit-ledger label. */
function evaluationMetricLabel(metric: EvaluationMetric, topK: number): string {
  // Retrieval labels include the immutable cutoff while answer labels name the judgment.
  if (RETRIEVAL_METRICS.includes(metric as RetrievalMetric)) {
    return metricLabel(metric as RetrievalMetric, topK);
  }

  return metric === "groundedness"
    ? "Groundedness"
    : metric === "answer_relevance"
      ? "Answer relevance"
      : "Answer correctness";
}

/**
 * Format a normalized metric value while keeping missing results visibly unavailable.
 *
 * @param metric - Metric whose conventional display format should be used.
 * @param value - Normalized score or null when no eligible aggregate exists.
 * @returns Percentage or decimal display text.
 */
export function metricValue(metric: RetrievalMetric, value: number | null): string {
  // Percentage metrics are easier to scan as rates; MRR retains its conventional decimal form.
  if (value === null) {
    return "—";
  }

  return metric === "mrr" ? value.toFixed(3) : `${(value * 100).toFixed(1)}%`;
}

/** Format all normalized aggregates consistently while preserving MRR convention. */
function evaluationMetricValue(metric: EvaluationMetric, value: number | null): string {
  return metricValue(metric as RetrievalMetric, value);
}

/** Return selected metrics in retrieval-then-answer display order. */
function selectedMetrics(
  evaluation: RetrievalEvaluationSummary,
): EvaluationMetric[] {
  return [
    ...evaluation.configuration.retrieval_metrics,
    ...evaluation.configuration.answer_metrics,
  ];
}

/**
 * Convert a lifecycle value into concise interface copy.
 *
 * @param status - Persisted evaluation lifecycle state.
 * @returns Human-readable status text.
 */
function evaluationStatus(status: RetrievalEvaluationSummary["status"]): string {
  // Pending attempts are queued in the shared durable worker.
  return status === "pending"
    ? "Queued"
    : status.charAt(0).toUpperCase() + status.slice(1);
}

/**
 * Format a stored UTC timestamp in the user's local browser time.
 *
 * @param timestamp - ISO timestamp returned by the backend.
 * @returns Localized date and time text.
 */
function evaluationTime(timestamp: string): string {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(timestamp));
}

/**
 * Render the latest evaluation inside the existing run summary rail.
 *
 * @param props - Latest attempt, saved K, and parent-owned card class.
 * @returns A lifecycle summary with aggregate scores when available.
 */
export function EvaluationSummaryCard({
  evaluation,
  topK,
  className,
}: {
  evaluation: RetrievalEvaluationSummary | null;
  topK: number;
  className: string;
}) {
  // Absence means the run has no automatic or manually requested evaluation attempt.
  if (evaluation === null) {
    return (
      <article className={className}>
        <span className={styles.eyebrow}>Evaluation status</span>
        <strong className={styles.summaryState}>Not requested</strong>
        <p>No evaluation attempt is saved for this run.</p>
      </article>
    );
  }

  const metrics = selectedMetrics(evaluation);

  return (
    <article className={className}>
      <span className={styles.eyebrow}>Evaluation status</span>
      <div className={styles.summaryHeading}>
        <strong className={styles.summaryState} data-status={evaluation.status}>
          {evaluationStatus(evaluation.status)}
        </strong>
        {evaluation.status === "running" || evaluation.status === "pending" ? (
          <FiRefreshCw className={styles.spinner} aria-hidden="true" />
        ) : null}
      </div>
      {evaluation.status === "completed" ? (
        <div className={styles.summaryScores}>
          {metrics.map((metric) => (
            <span key={metric}>
              <b>{evaluationMetricValue(metric, evaluation.aggregates[metric] ?? null)}</b>
              <small>{evaluationMetricLabel(metric, topK)}</small>
            </span>
          ))}
        </div>
      ) : null}
      <p>
        {evaluation.status === "completed"
          ? evaluation.has_errors
            ? "Partial results · some judge requests failed"
            : `${evaluation.eligible_count ?? 0} of ${evaluation.total_count ?? 0} eligible`
          : evaluation.status === "failed"
            ? evaluation.error?.message ?? "Evaluation failed."
            : "Scoring saved retrieval and generation results."}
      </p>
    </article>
  );
}

/**
 * Render one attempt's lifecycle, configuration, aggregates, and coverage.
 *
 * @param props - Selected evaluation detail and benchmark retrieval cutoff.
 * @returns The selected attempt inspection panel.
 */
function AttemptDetail({
  detail,
  topK,
}: {
  detail: RetrievalEvaluationDetail;
  topK: number;
}) {
  const metrics = selectedMetrics(detail);

  return (
    <article className={styles.attemptDetail} aria-live="polite">
      {/* Attempt identity keeps timestamps and selected metrics beside their result. */}
      <header className={styles.attemptDetailHead}>
        <div>
          <span className={styles.eyebrow}>Selected attempt</span>
          <h3>{evaluationTime(detail.created_at)}</h3>
          <code>{detail.id}</code>
        </div>
        <span className={styles.statusBadge} data-status={detail.status}>
          {evaluationStatus(detail.status)}
        </span>
      </header>

      {/* Metric cells preserve the selection even before aggregate values are available. */}
      <div className={styles.aggregateGrid}>
        {metrics.map((metric) => (
          <div key={metric}>
            <span>{evaluationMetricLabel(metric, topK)}</span>
            <strong>{evaluationMetricValue(metric, detail.aggregates[metric] ?? null)}</strong>
            {detail.coverage[metric] ? (
              <small className={styles.coverageCounts}>
                {detail.coverage[metric].eligible} eligible · {detail.coverage[metric].scored}
                {" "}scored · {detail.coverage[metric].skipped} skipped ·
                {" "}{detail.coverage[metric].error} errors
              </small>
            ) : null}
          </div>
        ))}
      </div>

      {/* Coverage and failure copy explain whether aggregates represent every question. */}
      {detail.status === "completed" ? (
        <>
          {detail.has_errors ? (
            <div className={styles.partialWarning} role="alert">
              <FiAlertCircle aria-hidden="true" />
              <span>Partial results: one or more LLM judge requests failed.</span>
            </div>
          ) : null}
          {detail.configuration.evaluator ? (
            <details className={styles.evaluatorDetails}>
              <summary>LLM judge configuration</summary>
              <dl>
                <div><dt>Evaluator</dt><dd>
                  {detail.configuration.evaluator.provider} /
                  {" "}{detail.configuration.evaluator.model}
                </dd></div>
                <div><dt>Prompt</dt><dd>
                  {detail.configuration.evaluator.prompt_version}
                </dd></div>
                <div><dt>Rubric</dt><dd>
                  {detail.configuration.evaluator.rubric_version}
                </dd></div>
                <div><dt>Fixed settings</dt><dd>
                  Temperature {detail.configuration.evaluator.temperature} ·
                  {" "}{detail.configuration.evaluator.reasoning_effort} reasoning ·
                  {" "}{detail.configuration.evaluator.structured_output}
                </dd></div>
              </dl>
            </details>
          ) : null}
        </>
      ) : detail.status === "failed" ? (
        <div className={styles.attemptFailure} role="alert">
          <FiAlertCircle aria-hidden="true" />
          <span>{detail.error?.message ?? "Evaluation failed."}</span>
        </div>
      ) : (
        <div className={styles.activeMessage} role="status">
          <FiRefreshCw className={styles.spinner} aria-hidden="true" />
          <span>Scoring the benchmark&apos;s saved chunk ranking.</span>
        </div>
      )}
    </article>
  );
}

/**
 * Present evaluation history and controls for independently scoring a completed run.
 *
 * @param props - Evaluation state, run lifecycle, saved metric defaults, and retrieval cutoff.
 * @returns The complete evaluation history and reevaluation workspace.
 */
export function EvaluationWorkspace({
  state,
  runStatus,
  savedRetrievalMetrics,
  savedAnswerMetrics,
  topK,
}: {
  state: EvaluationsState;
  runStatus: "pending" | "running" | "completed" | "failed";
  savedRetrievalMetrics: string[];
  savedAnswerMetrics: string[];
  topK: number;
}) {
  // The form opens only when the user asks to create another durable attempt.
  const [formOpen, setFormOpen] = useState(false);

  // Only backend-supported retrieval identifiers may be submitted from saved configuration.
  const defaultRetrievalMetrics = RETRIEVAL_METRICS.filter((metric) =>
    savedRetrievalMetrics.includes(metric)
  );

  // Saved answer selections initialize every manual re-evaluation form.
  const defaultAnswerMetrics = ANSWER_METRICS.filter((metric) =>
    savedAnswerMetrics.includes(metric)
  );

  // Metric selection resets from the immutable run snapshot whenever the form opens.
  const [selectedRetrievalMetrics, setSelectedRetrievalMetrics] =
    useState<RetrievalMetric[]>(defaultRetrievalMetrics);

  // Answer choices remain independent because they invoke the configured LLM judge.
  const [selectedAnswerMetrics, setSelectedAnswerMetrics] =
    useState<AnswerMetric[]>(defaultAnswerMetrics);

  const hasActiveAttempt = state.attempts.some((attempt) =>
    attempt.status === "pending" || attempt.status === "running"
  );
  const canEvaluate = runStatus === "completed"
    && !hasActiveAttempt
    && !state.creating
    && !state.loading
    && !state.error;

  /**
   * Add or remove one metric while preserving the evaluator's stable display order.
   *
   * @param metric - Metric whose checkbox changed.
   * @param checked - Whether the metric should remain selected.
   * @returns Nothing; local form state receives the ordered selection.
   */
  function toggleRetrievalMetric(metric: RetrievalMetric, checked: boolean): void {
    setSelectedRetrievalMetrics(
      RETRIEVAL_METRICS.filter((candidate) =>
        candidate === metric ? checked : selectedRetrievalMetrics.includes(candidate)
      ),
    );
  }

  /** Add or remove one LLM-judged metric while retaining stable display order. */
  function toggleAnswerMetric(metric: AnswerMetric, checked: boolean): void {
    setSelectedAnswerMetrics(
      ANSWER_METRICS.filter((candidate) =>
        candidate === metric ? checked : selectedAnswerMetrics.includes(candidate)
      ),
    );
  }

  /**
   * Open or close the form and restore the run's saved selection when opening it.
   *
   * @returns Nothing; local form visibility and selection are updated.
   */
  function toggleEvaluationForm(): void {
    setFormOpen((current) => {
      // Each new opening starts from the immutable run configuration.
      if (!current) {
        setSelectedRetrievalMetrics(defaultRetrievalMetrics);
        setSelectedAnswerMetrics(defaultAnswerMetrics);
      }

      return !current;
    });
  }

  /**
   * Queue the selected metrics and close the form after a successful response.
   *
   * @returns A promise resolved after creation succeeds or reports an error.
   */
  async function submitEvaluation(): Promise<void> {
    // The interface rejects an empty attempt even though no provider call would occur.
    if (
      selectedRetrievalMetrics.length + selectedAnswerMetrics.length === 0
      || hasActiveAttempt
    ) {
      return;
    }

    const created = await state.evaluate(
      selectedRetrievalMetrics,
      selectedAnswerMetrics,
    );

    // Keep a failed form open so its structured error remains actionable.
    if (created !== null) {
      setFormOpen(false);
    }
  }

  return (
    <section className={styles.workspace} aria-labelledby="evaluation-results-heading">
      {/* Evaluation has one primary action and explains its saved-input boundary. */}
      <header className={styles.workspaceHead}>
        <div>
          <span className={styles.eyebrow}>Saved-result scoring</span>
          <h2 id="evaluation-results-heading">Evaluation results</h2>
          <p>
            Score saved retrieval ranks and generated answers without rerunning the pipeline.
          </p>
        </div>
        <button
          className={styles.evaluateButton}
          disabled={!canEvaluate}
          onClick={toggleEvaluationForm}
          type="button"
        >
          <FiPlay aria-hidden="true" />
          {state.attempts.length === 0 ? "Run evaluation" : "Evaluate again"}
        </button>
      </header>

      {/* The inline form keeps the metric choice beside the attempt it will create. */}
      {formOpen ? (
        <div className={styles.evaluationForm}>
          <fieldset>
            <legend>Select retrieval metrics</legend>
            <div className={styles.metricChoices}>
              {RETRIEVAL_METRICS.map((metric) => (
                <label key={metric}>
                  <input
                    checked={selectedRetrievalMetrics.includes(metric)}
                    onChange={(event) =>
                      toggleRetrievalMetric(metric, event.target.checked)
                    }
                    type="checkbox"
                  />
                  <span>{metricLabel(metric, topK)}</span>
                </label>
              ))}
            </div>
          </fieldset>
          <fieldset>
            <legend>LLM-judged answer metrics</legend>
            <p className={styles.judgeNote}>
              These scores are model judgments, not objective measurements.
            </p>
            <div className={styles.metricChoices}>
              {ANSWER_METRICS.map((metric) => (
                <label key={metric}>
                  <input
                    checked={selectedAnswerMetrics.includes(metric)}
                    onChange={(event) =>
                      toggleAnswerMetric(metric, event.target.checked)
                    }
                    type="checkbox"
                  />
                  <span>{evaluationMetricLabel(metric, topK)}</span>
                </label>
              ))}
            </div>
          </fieldset>
          <div className={styles.formActions}>
            <button onClick={() => setFormOpen(false)} type="button">
              Cancel
            </button>
            <button
              className={styles.primaryAction}
              disabled={
                selectedRetrievalMetrics.length + selectedAnswerMetrics.length === 0
                || state.creating
                || hasActiveAttempt
              }
              onClick={() => void submitEvaluation()}
              type="button"
            >
              {state.creating ? "Queueing…" : "Run evaluation"}
            </button>
          </div>
          {selectedRetrievalMetrics.length + selectedAnswerMetrics.length === 0 ? (
            <p className={styles.formError}>Select at least one evaluation metric.</p>
          ) : null}
          {state.createError ? (
            <p className={styles.formError} role="alert">{state.createError}</p>
          ) : null}
        </div>
      ) : null}

      {/* History and detail form a durable ledger rather than replacing prior attempts. */}
      {state.error ? (
        <div className={styles.loadError} role="alert">
          <FiAlertCircle aria-hidden="true" />
          <span>{state.error}</span>
          <button onClick={state.retry} type="button">Retry</button>
        </div>
      ) : null}

      {state.loading && state.attempts.length === 0 ? (
        <div className={styles.emptyState} role="status">
          <FiRefreshCw className={styles.spinner} aria-hidden="true" />
          Loading evaluation history…
        </div>
      ) : state.attempts.length === 0 ? (
        <div className={styles.emptyState}>
          <FiTarget aria-hidden="true" />
          <div>
            <strong>No evaluation attempts</strong>
            <p>Score saved retrieval results, generated answers, or both.</p>
          </div>
        </div>
      ) : (
        <div className={styles.attemptGrid}>
          <nav className={styles.attemptList} aria-label="Evaluation attempts">
            {state.attempts.map((attempt, index) => (
              <button
                aria-current={state.selectedId === attempt.id ? "true" : undefined}
                className={styles.attemptButton}
                data-status={attempt.status}
                key={attempt.id}
                onClick={() => state.selectAttempt(attempt.id)}
                type="button"
              >
                <span className={styles.attemptNumber}>
                  {index === 0 ? "Latest" : `Attempt ${state.attempts.length - index}`}
                </span>
                <strong>{evaluationTime(attempt.created_at)}</strong>
                <small>
                  {selectedMetrics(attempt)
                    .map((metric) => evaluationMetricLabel(metric, topK))
                    .join(" · ")}
                </small>
                <span className={styles.attemptStatus}>
                  {attempt.status === "completed" ? (
                    <FiCheckCircle aria-hidden="true" />
                  ) : attempt.status === "failed" ? (
                    <FiAlertCircle aria-hidden="true" />
                  ) : (
                    <FiClock aria-hidden="true" />
                  )}
                  {evaluationStatus(attempt.status)}
                </span>
              </button>
            ))}
          </nav>
          {state.detail ? (
            <AttemptDetail detail={state.detail} topK={topK} />
          ) : (
            <div className={styles.detailLoading} role="status">
              <FiRefreshCw className={styles.spinner} aria-hidden="true" />
              Loading selected attempt…
            </div>
          )}
        </div>
      )}
    </section>
  );
}

/**
 * Render one question's saved scores and chunk-rank document evidence.
 *
 * @param props - Selected attempt, question outcome, ranked chunks, and retrieval cutoff.
 * @returns A score panel with relevant labels and preserved chunk ranks.
 */
export function QuestionEvaluationPanel({
  evaluation,
  outcome,
  chunks,
  topK,
}: {
  evaluation: RetrievalEvaluationDetail | null;
  outcome: EvaluationQuestion | null;
  chunks: RankedChunk[];
  topK: number;
}) {
  /**
   * Resolve a retrieved document ID to the filename already present in run evidence.
   *
   * @param documentId - Stable source document identifier.
   * @returns A retrieved filename when known, otherwise the stable identifier.
   */
  function documentName(documentId: string): string {
    return chunks.find((chunk) => chunk.source_document_id === documentId)
      ?.original_filename ?? documentId;
  }

  return (
    <section className={styles.questionPanel} aria-labelledby="question-evaluation-heading">
      <header className={styles.questionPanelHead}>
        <div>
          <span className={styles.eyebrow}>Selected evaluation attempt</span>
          <h2 id="question-evaluation-heading">Question evaluation</h2>
        </div>
        {evaluation ? (
          <span className={styles.statusBadge} data-status={evaluation.status}>
            {evaluationStatus(evaluation.status)}
          </span>
        ) : null}
      </header>

      {/* Empty and active states avoid presenting missing evidence as a zero score. */}
      {evaluation === null ? (
        <p className={styles.questionMessage}>Select an evaluation attempt to inspect scores.</p>
      ) : evaluation.status === "pending" || evaluation.status === "running" ? (
        <p className={styles.questionMessage}>Question evidence will appear after scoring.</p>
      ) : evaluation.status === "failed" ? (
        <p className={styles.questionMessage}>
          {evaluation.error?.message ?? "This evaluation attempt failed."}
        </p>
      ) : outcome === null ? (
        <p className={styles.questionMessage}>No saved result exists for this question.</p>
      ) : (
        <>
          {/* Retrieval scoring remains independent from generated-answer judgments. */}
          {evaluation.configuration.retrieval_metrics.length > 0 ? (
            <div className={styles.metricGroup}>
              <span className={styles.eyebrow}>Deterministic retrieval metrics</span>
              {outcome.skip_reason !== null ? (
                <p className={styles.questionMessage}>
                  Skipped because this question has no resolved document labels.
                </p>
              ) : (
                <div className={styles.questionScores}>
                  {evaluation.configuration.retrieval_metrics.map((metric) => (
                    <div key={metric}>
                      <span>{metricLabel(metric, topK)}</span>
                      <strong>{metricValue(metric, outcome.scores?.[metric] ?? null)}</strong>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : null}

          {/* Judge cards retain the original rubric score, rationale, and failure state. */}
          {evaluation.configuration.answer_metrics.length > 0 ? (
            <div className={styles.metricGroup}>
              <span className={styles.eyebrow}>LLM-judged answer metrics</span>
              <p className={styles.judgeNote}>
                These are model judgments under the saved rubric, not objective measurements.
              </p>
              <div className={styles.answerResults}>
                {evaluation.configuration.answer_metrics.map((metric) => {
                  const result = outcome.answer_scores[metric];
                  const skip = outcome.answer_skips[metric];
                  const failed = outcome.answer_error?.metrics.includes(metric) ?? false;

                  return (
                    <article key={metric}>
                      <header>
                        <strong>{evaluationMetricLabel(metric, topK)}</strong>
                        {result ? (
                          <span>{result.rubric_score} / 4 ·
                            {" "}{(result.score * 100).toFixed(1)}%</span>
                        ) : null}
                      </header>
                      {result ? (
                        <>
                          <p>{result.rationale}</p>
                          {metric === "groundedness" && result.evidence_ranks.length > 0 ? (
                            <small>
                              Supporting prompt context: {result.evidence_ranks.join(", ")}
                            </small>
                          ) : null}
                        </>
                      ) : failed ? (
                        <p className={styles.answerError}>
                          {outcome.answer_error?.message ?? "The judge request failed."}
                        </p>
                      ) : (
                        <p className={styles.answerSkip}>
                          {skip === "no_reference_answer"
                            ? "Skipped: no reference answer was supplied."
                            : "Skipped: no generated answer was saved."}
                        </p>
                      )}
                    </article>
                  );
                })}
              </div>
              {outcome.judge ? (
                <details className={styles.judgeProvenance}>
                  <summary>Question judge provenance</summary>
                  <dl>
                    <div><dt>Model</dt><dd>{outcome.judge.provider_model ?? "—"}</dd></div>
                    <div><dt>Duration</dt><dd>{outcome.judge.duration_ms ?? "—"} ms</dd></div>
                    <div><dt>Tokens</dt><dd>
                      {outcome.judge.prompt_tokens ?? "—"} input ·
                      {" "}{outcome.judge.completion_tokens ?? "—"} output ·
                      {" "}{outcome.judge.total_tokens ?? "—"} total
                    </dd></div>
                    <div><dt>Request ID</dt><dd>
                      {outcome.judge.provider_request_id ?? "—"}
                    </dd></div>
                  </dl>
                </details>
              ) : null}
            </div>
          ) : null}

          {/* Relevant labels state the expected documents independently of retrieved ranks. */}
          {evaluation.configuration.retrieval_metrics.length > 0 ? (
          <div className={styles.expectedDocuments}>
            <span className={styles.eyebrow}>Relevant document labels</span>
            <div>
              {outcome.relevant_document_ids.map((documentId) => (
                <span key={documentId} title={documentId}>
                  {documentName(documentId)}
                </span>
              ))}
            </div>
          </div>
          ) : null}

          {/* The rank tape deliberately keeps duplicate documents at their chunk positions. */}
          {evaluation.configuration.retrieval_metrics.length > 0 ? (
          <ol className={styles.rankTape} aria-label="Ranked retrieved document evidence">
            {outcome.ranked_document_ids.map((documentId, index) => {
              const rank = index + 1;
              const matched = outcome.matching_ranks.includes(rank);

              return (
                <li data-matched={matched} key={`${rank}-${documentId}`}>
                  <span>{String(rank).padStart(2, "0")}</span>
                  <div>
                    <strong>{documentName(documentId)}</strong>
                    <code>{documentId}</code>
                  </div>
                  {matched ? <b>Label match</b> : <small>Retrieved</small>}
                </li>
              );
            })}
          </ol>
          ) : null}
        </>
      )}
    </section>
  );
}
