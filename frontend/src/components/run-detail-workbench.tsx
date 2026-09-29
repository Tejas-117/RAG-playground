"use client";

import Link from "next/link";
import { useState } from "react";
import {
  FiArrowLeft,
  FiCheck,
  FiClock,
  FiCopy,
  FiDatabase,
  FiFileText,
  FiInfo,
  FiSearch,
} from "react-icons/fi";
import WorkbenchSidebar from "@/components/workbench-sidebar";
import WorkbenchGridCanvas from "@/components/workbench-grid-canvas";
import {
  PREVIEW_QUESTIONS,
  type PreviewChunk,
  type PreviewQuestion,
} from "@/lib/run-detail-preview";
import styles from "./run-detail-workbench.module.css";

// This sample identity belongs to the design preview and is never read from the URL.
const SAMPLE_RUN_ID = "8f29a1b0-4c8d-4e92-ba77-19db943d0e21";

// The sample models a run after two completions and while its third question generates.
const SAMPLE_TOTAL = PREVIEW_QUESTIONS.length;
const SAMPLE_COMPLETED = 2;

type QuestionFilter = "all" | PreviewQuestion["status"];

/** Format a recorded millisecond duration for a compact data label. */
function formatDuration(value: number | null): string {
  // No result means no measured duration; zero is a real measurement.
  if (value === null) return "—";
  return value < 1000 ? `${value} ms` : `${(value / 1000).toFixed(2)} s`;
}

/** Label a lifecycle value as it appears in the question navigator. */
function statusLabel(status: PreviewQuestion["status"]): string {
  // Keep concise status words consistent across navigator and inspector.
  return status.charAt(0).toUpperCase() + status.slice(1);
}

/** Render a ranked piece of source evidence with its prompt inclusion and provenance. */
function EvidenceChunk({ chunk }: { chunk: PreviewChunk }) {
  return (
    <article className={styles.chunk}>
      {/* Rank, distance, and prompt inclusion are separate pieces of evidence. */}
      <div className={styles.chunkHead}>
        <strong className={styles.rank}>
          Rank {chunk.rank}
        </strong>
        <span className={styles.mono}>
          Raw cosine distance {chunk.distance.toFixed(3)}
        </span>
        {chunk.contextOrder !== null && (
          <span className={styles.contextBadge}>
            Included in prompt · context {chunk.contextOrder}
          </span>
        )}
      </div>
      {/* Source labels keep the excerpt tied to the uploaded document. */}
      <p className={styles.chunkSource}>
        <FiFileText aria-hidden="true" />
        <span>{chunk.filename}</span>
        {chunk.page !== null && <span>· Page {chunk.page}</span>}
      </p>
      <p className={styles.chunkText}>
        {chunk.text}
      </p>
      {/* Full metadata stays available without crowding the ranking list. */}
      <details className={styles.chunkDetails}>
        <summary>Source provenance</summary>
        <dl className={styles.provenance}>
          <div><dt>Chunk ID</dt><dd>{chunk.id}</dd></div>
          <div><dt>Document ID</dt><dd>{chunk.documentId}</dd></div>
          <div>
            <dt>Character offsets</dt>
            <dd>{chunk.characterStart}–{chunk.characterEnd}</dd>
          </div>
          <div>
            <dt>Prompt context</dt>
            <dd>
              {chunk.contextOrder === null ? "Not included" : `Position ${chunk.contextOrder}`}
            </dd>
          </div>
        </dl>
      </details>
    </article>
  );
}

/** Show the selected question's input, answer, retrieval, and saved-stage states. */
function QuestionInspector({ question }: { question: PreviewQuestion }) {
  return (
    <div className={styles.inspector}>
      {/* The original question and reference remain distinct from model output. */}
      <section className={styles.panel} aria-labelledby="question-heading">
        <div className={styles.panelHead}>
          <h2 id="question-heading">Question {question.ordinal + 1}</h2>
          <span className={styles.mono}>{question.id}</span>
        </div>
        <p className={styles.questionText}>{question.question}</p>
        <div className={styles.reference}>
          <span className={styles.eyebrow}>Dataset reference · not generated</span>
          <p>{question.reference ?? "No reference answer was supplied for this question."}</p>
        </div>
      </section>

      {/* The answer panel reflects the selected example's actual preview lifecycle. */}
      <section className={styles.panel} aria-labelledby="generation-heading">
        <div className={styles.panelHead}>
          <h2 id="generation-heading">Generated answer</h2>
          <span className={styles.sectionMeta}>
            {question.answer ? formatDuration(question.generationMs) : statusLabel(question.status)}
          </span>
        </div>
        {question.answer ? (
          <>
            <p className={styles.answer}>{question.answer}</p>
            <div className={styles.provenanceLine}>
              <span>Groq / qwen-qwen3-32b</span>
              <span>Finish: stop</span>
              <span>
                {question.promptTokens ?? "—"} input · {question.completionTokens ?? "—"} output
              </span>
            </div>
            <details className={styles.chunkDetails}>
              <summary>Generation provenance</summary>
              <dl className={styles.provenance}>
                <div><dt>Prompt template</dt><dd>rag-answer-v1</dd></div>
                <div><dt>Provider policy</dt><dd>groq-chat-v1</dd></div>
                <div><dt>Provider called</dt><dd>Yes</dd></div>
              </dl>
            </details>
          </>
        ) : (
          <p className={styles.pendingMessage}>
            {question.status === "running" ?
              "Retrieval is complete. This question is generating an answer." :
              question.status === "failed" ?
                question.error ?? "Generation stopped before an answer was saved." :
                "This question has not started. No answer is available yet."}
          </p>
        )}
      </section>

      {/* Ranked evidence can exist even when generation has not completed. */}
      <section className={styles.panel} aria-labelledby="retrieval-heading">
        <div className={styles.panelHead}>
          <h2 id="retrieval-heading">Retrieval evidence</h2>
          <span className={styles.sectionMeta}>
            Top K 5 · {question.chunks.length} returned · {formatDuration(question.retrievalMs)}
          </span>
        </div>
        <p className={styles.distanceNote}>
          Values are raw cosine distances, not relevance percentages. Prompt badges show
          which retrieved chunks were included in generation.
        </p>
        {question.chunks.length > 0 ? (
          <div className={styles.chunkList}>
            {question.chunks.map((chunk) => (
              <EvidenceChunk key={chunk.id} chunk={chunk} />
            ))}
          </div>
        ) : (
          <p className={styles.pendingMessage}>
            {question.retrievalMs === null ?
              "Retrieval has not completed for this question." :
              "Retrieval completed without returning any chunks."}
          </p>
        )}
      </section>
    </div>
  );
}

/** Present Stitch-inspired run inspection using local sample data only. */
export default function RunDetailWorkbench({ requestedRunId }: { requestedRunId: string }) {
  // Search narrows the local sample question navigator without a network request.
  const [search, setSearch] = useState("");

  // The status filter controls which sample questions appear in the navigator.
  const [filter, setFilter] = useState<QuestionFilter>("all");

  // Persist selection across filtering so the inspector does not jump unexpectedly.
  const [selectedId, setSelectedId] = useState(PREVIEW_QUESTIONS[0].id);

  // Accessible feedback reports whether copying the sample ID succeeded.
  const [notice, setNotice] = useState("");

  // Select the requested example from the sample dataset, falling back to its first question.
  const selected = PREVIEW_QUESTIONS.find((question) => question.id === selectedId)
    ?? PREVIEW_QUESTIONS[0];

  // Apply both local controls while preserving the dataset's original question order.
  const visibleQuestions = PREVIEW_QUESTIONS.filter((question) => (
    (filter === "all" || question.status === filter)
    && question.question.toLowerCase().includes(search.toLowerCase())
  ));

  // Status counts describe all sample questions, independent of the active filter.
  const statuses: QuestionFilter[] = ["all", "completed", "running", "failed", "pending"];

  /** Copy the sample ID and report browser clipboard failures; returns a promise. */
  async function copySampleId() {
    try {
      await navigator.clipboard.writeText(SAMPLE_RUN_ID);
      setNotice("Sample run ID copied.");
    } catch {
      setNotice("Unable to copy the sample run ID.");
    }
  }

  return (
    <main className={styles.shell}>
      <WorkbenchSidebar activeLabel="Runs" />
      <WorkbenchGridCanvas className={styles.workspace}>
        {/* The preview notice prevents fixture results from masquerading as a real run. */}
        <div className={styles.preview} role="note">
          <FiInfo aria-hidden="true" />
          <span>
            Design preview · Sample run data only. The requested run
            {" "}<code>{requestedRunId}</code> has not been loaded.
          </span>
        </div>

        {/* Navigation and identity connect this inspection view to the runs inventory. */}
        <header className={styles.header}>
          <Link href="/runs" className={styles.back}>
            <FiArrowLeft aria-hidden="true" />
            Back to runs
          </Link>
          <div className={styles.headerMain}>
            <div>
              <p className={styles.eyebrow}>Run inspection / sample</p>
              <div className={styles.titleLine}>
                <h1>Run {SAMPLE_RUN_ID}</h1>
                <button
                  type="button"
                  onClick={copySampleId}
                  aria-label="Copy sample run ID"
                  title="Copy run ID"
                >
                  <FiCopy aria-hidden="true" />
                </button>
              </div>
            </div>
            <span className={styles.status}>
              <span className={styles.statusDot} />
              Running
            </span>
          </div>
          <div className={styles.identity}>
            <span>Corpus: NimbusForge</span>
            <span>Index: nimbus_forge_index</span>
            <span>Dataset: NimbusForge questions</span>
            <span>Created: Sep 29, 2026 · 14:22</span>
          </div>
        </header>

        {/* A compact metrics rail summarizes execution without implying answer quality. */}
        <section className={styles.metrics} aria-label="Run summary">
          <article className={styles.metricCard}>
            <span className={styles.eyebrow}>Question progress</span>
            <strong>{SAMPLE_COMPLETED} <small>/ {SAMPLE_TOTAL} completed</small></strong>
            <progress max={SAMPLE_TOTAL} value={SAMPLE_COMPLETED}>
              {SAMPLE_COMPLETED} of {SAMPLE_TOTAL}
            </progress>
            <p>1 running · 0 failed · 2 pending</p>
          </article>
          <article className={styles.metricCard}>
            <span className={styles.eyebrow}>Stage duration</span>
            <div className={styles.pairedMetrics}>
              <div><strong>126 ms</strong><span>Retrieval avg · 3 results</span></div>
              <div><strong>977 ms</strong><span>Generation avg · 2 results</span></div>
            </div>
          </article>
          <article className={styles.metricCard}>
            <span className={styles.eyebrow}>Reported token usage</span>
            <strong>941 <small>in</small> / 93 <small>out</small></strong>
            <p>Reported by 2 of 2 saved generations</p>
          </article>
          <article className={styles.metricCard}>
            <span className={styles.eyebrow}>Evaluation status</span>
            <strong className={styles.evaluationState}>Not evaluated</strong>
            <p>Execution status does not measure answer quality.</p>
          </article>
        </section>

        {/* An expandable snapshot follows the pipeline's real stage order. */}
        <details className={styles.configuration} open>
          <summary>
            <FiDatabase aria-hidden="true" />
            Saved configuration and provenance
            <span>Immutable snapshot</span>
          </summary>
          <div className={styles.configurationGrid}>
            <div><b>Preparation</b><span>Recursive · 800 tokens · 100 overlap</span>
              <span>Ollama / nomic-embed-text · cosine</span></div>
            <div><b>Retrieval</b><span>Vector search · top K 5</span>
              <span>Raw cosine distance</span></div>
            <div><b>Generation</b><span>Groq / qwen-qwen3-32b</span>
              <span>Temperature 0.2 · max output 1000</span></div>
            <div><b>Evaluation targets</b><span>Hit rate, MRR, groundedness</span>
              <span>Configured; not evaluated</span></div>
          </div>
          <p className={styles.configIds}>
            Corpus: corpus-nimbusforge · Prepared index: prepared-nimbusforge ·
            Vector index: vector-nimbusforge · Dataset: dataset-nimbusforge
          </p>
        </details>

        {/* The question rail is the page's primary interaction and mirrors the Stitch layout. */}
        <div className={styles.questionGrid}>
          <section className={styles.navigator} aria-label="Execution questions">
            <div className={styles.navigatorHead}>
              <div className={styles.navigatorTitle}>
                <h2>Execution questions</h2>
                <span>{SAMPLE_TOTAL} total</span>
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
                  const count = status === "all" ? SAMPLE_TOTAL :
                    PREVIEW_QUESTIONS.filter((question) => question.status === status).length;

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
                  data-selected={selected.id === question.id}
                  key={question.id}
                  type="button"
                  onClick={() => setSelectedId(question.id)}
                >
                  <span className={styles.questionItemTop}>
                    <b>Q{question.ordinal + 1}</b>
                    <span>{formatDuration(question.durationMs)}</span>
                    <span className={styles.questionStatus} data-status={question.status}>
                      {statusLabel(question.status)}
                    </span>
                  </span>
                  <span className={styles.questionItemText}>{question.question}</span>
                  <span className={styles.questionItemFoot}>
                    {question.stage ? `Stage: ${question.stage}` :
                      question.status === "pending" ? "Waiting to start" : "Execution saved"}
                    {question.retrievalMs !== null &&
                      ` · ${question.chunks.length} retrieved`}
                    {question.answer && " · Answer saved"}
                  </span>
                </button>
              ))}
              {visibleQuestions.length === 0 && (
                <p className={styles.noMatch}>No questions match this search and status.</p>
              )}
            </div>
          </section>
          <QuestionInspector question={selected} />
        </div>
        <p className={styles.srNotice} role="status">
          {notice && <><FiCheck aria-hidden="true" />{notice}</>}
        </p>
        <footer className={styles.footer}>
          <FiClock aria-hidden="true" />
          Preview timestamps and results are illustrative.
        </footer>
      </WorkbenchGridCanvas>
    </main>
  );
}
