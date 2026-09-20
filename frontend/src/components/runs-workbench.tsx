"use client";

import Link from "next/link";
import { useState } from "react";
import { FiAlertCircle, FiCheckCircle, FiCopy, FiDownload, FiInfo } from "react-icons/fi";
import { FiActivity, FiSearch, FiTerminal, FiArrowRight } from "react-icons/fi";
import WorkbenchSidebar from "@/components/workbench-sidebar";
import { PREVIEW_DATE, RUN_PREVIEWS, type RunPreview } from "@/lib/run-preview-data";
import styles from "./runs-workbench.module.css";

// Four entries preserve the reference's readable row density and demonstrate pagination.
const PAGE_SIZE = 4;

/** Format nullable seconds for an execution duration; returns a human-readable label. */
function duration(seconds: number | null): string {
  // A queued run has no execution duration, rather than a duration of zero.
  if (seconds === null) return "Waiting to start";
  return seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

/** Format a nullable token count; missing provider usage remains unavailable. */
function tokens(value: number | null): string {
  return value === null ? "—" : new Intl.NumberFormat("en", {
    notation: "compact", maximumFractionDigits: 1,
  }).format(value);
}

/** Render one fixture and call onCopy with its full ID; returns an execution ledger row. */
function RunRow({ run, onCopy }: { run: RunPreview; onCopy: (id: string) => void }) {
  const percent = Math.round(run.completed / run.total * 100);
  return (
    <article className={styles.row} data-failed={run.status === "failed"}>
      {/* Identity ties this execution to its saved index and dataset. */}
      <div className={styles.identity}>
        <div className={styles.idLine}>
          <strong title={run.id}>
            {run.id.slice(0, 8)}
          </strong>
          <button aria-label={`Copy run ID ${run.id}`} onClick={() => onCopy(run.id)}>
            <FiCopy aria-hidden="true" />
          </button>
        </div>
        <time dateTime={run.created}>
          {run.created.slice(0, 10)} · {run.created.slice(11, 16)} UTC
        </time>
        <p>
          Index: <span className={styles.chip}>{run.index}</span>
        </p>
        <p>
          Dataset: <span>{run.dataset}</span>
        </p>
      </div>
      {/* The configuration records the generation model and preparation parameters. */}
      <div className={styles.config}>
        <strong>
          {run.provider} / {run.model}
        </strong>
        <p>
          Top K: {run.topK} · Temp: {run.temperature} · Chunk: {run.chunkSize}
        </p>
        <p>
          Evaluation: Not evaluated
        </p>
      </div>
      {/* Progress counts completed questions, never an invented time estimate. */}
      <div>
        <div className={styles.progressHeading}>
          <span className={styles.badge} data-status={run.status}>
            {run.status === "completed" ? "✓ " : ""}{run.status}
          </span>
          <span>
            {run.completed} / {run.total} questions ({percent}%)
          </span>
        </div>
        <progress aria-label={`Completed questions for ${run.id}`} max={run.total}
          value={run.completed} />
        <p className={styles.timing}>
          {run.status === "running" ? "Elapsed: " : run.seconds !== null ? "Duration: " : ""}
          <strong>{duration(run.seconds)}</strong>
        </p>
        <p>
          {run.status === "running" ? "Generating answer" :
            run.status === "queued" ? "Execution not started" :
              run.status === "failed" ? "Stopped during generation" : "All questions completed"}
        </p>
      </div>
      {/* Stage averages and token usage cover only recorded fixture results. */}
      <div className={styles.metrics}>
        <p>
          Avg retrieval: {run.retrievalMs === null ? "—" : `${run.retrievalMs}ms`}
        </p>
        <p>
          Avg generation: {run.generationMs === null ? "—" : `${run.generationMs / 1000}s`}
        </p>
        <p>
          Tokens: {tokens(run.inputTokens)} in / {tokens(run.outputTokens)} out
        </p>
        <small>
          {run.status === "running" || run.status === "failed" ? "Recorded results only" : " "}
        </small>
        <button disabled title="Run details will be available in a later update">
          View run <FiArrowRight aria-hidden="true" />
        </button>
      </div>
      {/* Failure remains attached to its run, with the unexecuted count made explicit. */}
      {run.status === "failed" && (
        <div className={styles.failure}>
          <FiAlertCircle aria-hidden="true" />
          <span>
            Generation failed: the provider request timed out. Partial results saved.
            {" "}{run.total - run.completed - 1} questions were not executed after one failed.
          </span>
        </div>
      )}
    </article>
  );
}

/** Render the local runs preview with filtering and export; takes no parameters. */
export default function RunsWorkbench() {
  // Search matches user-visible identity and target fields without a network request.
  const [search, setSearch] = useState("");

  // Filter values are local presentation state, not backend configuration.
  const [filters, setFilters] = useState({ status: "", index: "", dataset: "", days: "" });

  // Sorting applies to all matching fixtures before pagination.
  const [sort, setSort] = useState("newest");

  // Pagination keeps the ledger short while preserving all matching runs.
  const [page, setPage] = useState(1);

  // Clipboard and export feedback is announced to assistive technology.
  const [notice, setNotice] = useState("");

  const filtered = RUN_PREVIEWS.filter((run) => {
    // Combine every selected filter; fixture dates are relative to the preview reference day.
    const age = (Date.parse(PREVIEW_DATE) - Date.parse(run.created.slice(0, 10))) / 86400000;
    return `${run.id} ${run.index} ${run.dataset}`.toLowerCase().includes(search.toLowerCase())
      && (!filters.status || run.status === filters.status)
      && (!filters.index || run.index === filters.index)
      && (!filters.dataset || run.dataset === filters.dataset)
      && (!filters.days || age < Number(filters.days));
  }).sort((a, b) => {
    // Unknown durations stay last regardless of the selected duration ordering.
    if (sort === "duration") return (b.seconds ?? -1) - (a.seconds ?? -1);
    return sort === "oldest" ? a.created.localeCompare(b.created) :
      b.created.localeCompare(a.created);
  });
  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const visible = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
  const running = RUN_PREVIEWS.filter((run) => run.status === "running").length;
  const queued = RUN_PREVIEWS.filter((run) => run.status === "queued").length;
  const summaries = [
    { label: "Total runs", count: RUN_PREVIEWS.length, note: "All preview benchmarks",
      icon: FiTerminal },
    { label: "Active runs", count: running + queued, note: `${running} running · ${queued} queued`,
      icon: FiActivity },
    { label: "Completed runs", count: RUN_PREVIEWS.filter((r) => r.status === "completed").length,
      note: "Finished without execution errors", icon: FiCheckCircle },
    { label: "Failed runs", count: RUN_PREVIEWS.filter((r) => r.status === "failed").length,
      note: "Stopped before all questions finished", icon: FiAlertCircle },
  ];

  /** Reset all filter inputs and pagination; returns nothing. */
  function clearFilters() {
    setSearch("");
    setFilters({ status: "", index: "", dataset: "", days: "" });
    setPage(1);
  }

  /** Copy the supplied full run ID; resolves after reporting success or clipboard failure. */
  async function copyId(id: string) {
    try {
      await navigator.clipboard.writeText(id);
      setNotice("Run ID copied.");
    } catch {
      // Clipboard access can be denied even when the rest of the preview works.
      setNotice(`Unable to copy. Run ID: ${id}`);
    }
  }

  /** Download filtered sample runs as JSON; returns nothing and never contacts the backend. */
  function exportRuns() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(filtered, null, 2)], {
      type: "application/json",
    }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "runs-preview.json";
    link.click();
    // Release the object URL after the browser has started the download.
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setNotice(`Exported ${filtered.length} sample runs.`);
  }

  return (
    <main className={styles.shell}>
      {/* Reuse the application's navigation rather than the exported Stitch shell. */}
      <WorkbenchSidebar activeLabel="Runs" />
      <section className={styles.workspace}>
        {/* Page identity and actions remain available above the execution history. */}
        <header className={styles.header}>
          <div>
            <h1>
              Runs
            </h1>
            <p>
              Monitor benchmark progress and inspect past execution results.
            </p>
          </div>
          <div className={styles.actions}>
            <button onClick={exportRuns}>
              <FiDownload aria-hidden="true" /> Export
            </button>
            <Link href="/experiments" className={styles.primary}>
              New experiment
            </Link>
          </div>
        </header>
        {/* Preview disclosure prevents fixture status from appearing to be live execution. */}
        <div className={styles.preview}>
          <FiInfo aria-hidden="true" />
          Sample data · Preview as of September 20, 2026. Backend and run details are not connected.
        </div>
        {/* Summary counts always cover the complete sample inventory, independent of filters. */}
        <div className={styles.summaries}>
          {summaries.map(({ label, count, note, icon: Icon }) => (
            <section key={label} className={styles.summary} data-failed={label === "Failed runs"}>
              <div>
                <h2>
                  {label}
                </h2>
                <Icon aria-hidden="true" />
              </div>
              <strong>
                {count}
              </strong>
              <p>
                {note}
              </p>
            </section>
          ))}
        </div>
        <p className={styles.explanation}>
          <FiInfo aria-hidden="true" />
          Counts reflect execution status, not answer quality.
          Summary counts include all sample runs.
        </p>
        {/* Search, independent filters, and sorting operate on local fixtures only. */}
        <section className={styles.filters} aria-label="Filter runs">
          <div className={styles.filterControls}>
            <label className={styles.search}>
              <FiSearch aria-hidden="true" />
              <input aria-label="Search runs" value={search}
                placeholder="Search ID, index, dataset…"
                onChange={(event) => { setSearch(event.target.value); setPage(1); }} />
            </label>
            {(["status", "index", "dataset", "days"] as const).map((key) => (
              <select key={key} aria-label={`Filter by ${key === "days" ? "date" : key}`}
                value={filters[key]} onChange={(event) => {
                  // Return to the first page whenever the matching inventory changes.
                  setFilters({ ...filters, [key]: event.target.value });
                  setPage(1);
                }}>
                <option value="">
                  {{ days: "All dates", status: "All statuses",
                    index: "All indexes", dataset: "All datasets" }[key]}
                </option>
                {(key === "status" ? ["running", "queued", "completed", "failed"] :
                  key === "days" ? ["1", "7", "30"] :
                    [...new Set(RUN_PREVIEWS.map((run) => run[key]))]).map((value) => (
                  <option key={value} value={value}>
                    {key === "days" ? `Last ${value} days` : value}
                  </option>
                ))}
              </select>
            ))}
            <button className={styles.clear} onClick={clearFilters}>
              Clear filters
            </button>
          </div>
          <div className={styles.filterFooter}>
            <span aria-live="polite">
              Showing {visible.length} of {filtered.length} matching runs
            </span>
            <label>
              Sort by: <select value={sort} onChange={(event) => {
                setSort(event.target.value);
                setPage(1);
              }}>
                <option value="newest">Creation time (newest)</option>
                <option value="oldest">Creation time (oldest)</option>
                <option value="duration">Duration (longest)</option>
              </select>
            </label>
          </div>
        </section>
        {/* Responsive ledger groups each run's identity, configuration, progress, and metrics. */}
        <section className={styles.ledger} aria-label="Run history">
          <div className={styles.columns} aria-hidden="true">
            <span>
              Run identity & target
            </span>
            <span>
              Model & pipeline config
            </span>
            <span>
              Execution state & progress
            </span>
            <span>
              Metrics & action
            </span>
          </div>
          {visible.map((run) => (
            <RunRow key={run.id} run={run} onCopy={copyId} />
          ))}
          {visible.length === 0 && (
            <div className={styles.empty}>
              <FiSearch aria-hidden="true" />
              <h2>
                No matching runs
              </h2>
              <p>
                Try another search or clear your filters to see all sample runs.
              </p>
              <button onClick={clearFilters}>
                Clear filters
              </button>
            </div>
          )}
          <footer className={styles.pagination}>
            <span>
              Page {page} of {pageCount}
            </span>
            <div className={styles.actions}>
              <button disabled={page === 1} onClick={() => setPage(page - 1)}>
                Previous
              </button>
              <button disabled={page >= pageCount} onClick={() => setPage(page + 1)}>
                Next
              </button>
            </div>
          </footer>
        </section>
        <p role="status" className={styles.notice}>
          {notice}
        </p>
      </section>
    </main>
  );
}
