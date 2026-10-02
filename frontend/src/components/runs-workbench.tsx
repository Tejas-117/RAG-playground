"use client";

import Link from "next/link";
import { useState } from "react";
import { FiAlertCircle, FiCheckCircle, FiCopy, FiDownload, FiInfo } from "react-icons/fi";
import { FiActivity, FiSearch, FiTerminal, FiArrowRight } from "react-icons/fi";
import WorkbenchSidebar from "@/components/workbench-sidebar";
import WorkbenchGridCanvas from "@/components/workbench-grid-canvas";
import { presentRun, type RunPresentation } from "@/lib/run-presentation";
import { useRunHistory } from "@/lib/use-run-history";
import styles from "./runs-workbench.module.css";

// Four entries preserve the reference's readable row density on each page.
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

/** Render a validated run and call onCopy with its full ID; returns an execution ledger row. */
function RunRow({ run, onCopy }: { run: RunPresentation; onCopy: (id: string) => void }) {
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
          {run.status === "running" ? (run.source.current_stage ?? "Starting execution") :
            run.status === "queued" ? "Execution not started" :
              run.status === "failed" ?
                `Stopped: ${run.source.error?.stage ?? "execution"}` : "All questions completed"}
        </p>
      </div>
      {/* Stage averages exclude failed attempts; counts expose coverage of saved results. */}
      <div className={styles.metrics}>
        <p>
          Avg retrieval: {run.retrievalMs === null ? "—" : `${run.retrievalMs.toFixed(1)}ms`}
          {" "}({run.source.metrics.retrieval_result_count} results)
        </p>
        <p>
          Avg generation: {run.generationMs === null ? "—" :
            `${(run.generationMs / 1000).toFixed(2)}s`}
          {" "}({run.source.metrics.generation_result_count} results)
        </p>
        <p>
          Tokens: {tokens(run.inputTokens)} in / {tokens(run.outputTokens)} out
        </p>
        <Link href={`/runs/${encodeURIComponent(run.id)}`}>
          View run <FiArrowRight aria-hidden="true" />
        </Link>
      </div>
      {/* Failure remains attached to its run, with the unexecuted count made explicit. */}
      {run.status === "failed" && (
        <div className={styles.failure}>
          <FiAlertCircle aria-hidden="true" />
          <span>
            {run.source.error?.message ?? "Execution failed."}
            {" "}({run.source.error?.code ?? "unknown_error"}). Partial results saved.
            {" "}{run.source.failed_examples} failed; {run.source.pending_examples} not executed.
          </span>
        </div>
      )}
    </article>
  );
}

/** Render API-backed history with local filters and export; takes no parameters. */
export default function RunsWorkbench() {
  // Own live inventory, refresh feedback and a local elapsed-time clock.
  const { runs: snapshots, loading, error, now, retry } = useRunHistory();

  // Search matches user-visible identity and target fields without a network request.
  const [search, setSearch] = useState("");

  // Filter values are local presentation state, not backend configuration.
  const [filters, setFilters] = useState({ status: "", index: "", dataset: "", days: "" });

  // Sorting applies to all matching runs before pagination.
  const [sort, setSort] = useState("newest");

  // Pagination keeps the ledger short while preserving all matching runs.
  const [page, setPage] = useState(1);

  // Clipboard and export feedback is announced to assistive technology.
  const [notice, setNotice] = useState("");

  // Convert each API snapshot into row labels, including elapsed time using the local `now` clock.
  const runs = snapshots.map((run) => presentRun(run, now));

  // Keep only rows matching every filter, then sort that new array without modifying `runs`.
  const filtered = runs
    .filter((run) => {
      // Age is fractional days since creation: 86,400,000 milliseconds equals one 24-hour day.
      const age = (now - Date.parse(run.created)) / 86400000;

      return (
        // Search across the run ID, index name and dataset name, ignoring letter case.
        `${run.id} ${run.index} ${run.dataset}`.toLowerCase().includes(search.toLowerCase())
        // An empty status means any status; otherwise require the selected display status.
        && (!filters.status || run.status === filters.status)
        // Match the stable index ID, since different indexes can share the same name.
        && (!filters.index || run.indexId === filters.index)
        // Match the stable dataset ID for the same reason; an empty value allows all datasets.
        && (!filters.dataset || run.datasetId === filters.dataset)
        // Convert the selected day count from its select-input string before comparing ages.
        && (!filters.days || age < Number(filters.days))
      );
    })
    .sort((a, b) => {
      // `a` and `b` are two candidate rows. A negative result places `a` before `b`.
      // Subtract in reverse order for longest-first; null becomes -1, below even zero seconds.
      if (sort === "duration") {
        return (b.seconds ?? -1) - (a.seconds ?? -1);
      }

      // Compare backend UTC timestamps oldest-first, or reverse them for newest-first.
      return sort === "oldest"
        ? a.created.localeCompare(b.created)
        : b.created.localeCompare(a.created);
    });

  // Round up to include a partially filled last page; show page 1 even when no rows match.
  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));

  // Clamp the selected page when refreshed data or filters leave fewer pages available.
  const currentPage = Math.min(page, pageCount);

  // Convert the one-based page to zero-based slice bounds; the ending bound is exclusive.
  const visible = filtered.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);

  // Count executing runs across the full inventory, not just filtered or visible rows.
  const running = runs.filter((run) => run.status === "running").length;

  // Count runs waiting for execution; `presentRun` maps backend `pending` to UI `queued`.
  const queued = runs.filter((run) => run.status === "queued").length;

  // Card descriptors: `label` is the heading, `count` the value, `note` the explanation,
  // and `icon` the React icon component. All counts describe the full loaded inventory.
  const summaries = [
    // Include every saved run regardless of its lifecycle state.
    {
      label: "Total runs",
      count: runs.length,
      note: "All saved benchmarks",
      icon: FiTerminal,
    },
    // Active includes both currently executing and waiting work; the note separates them.
    {
      label: "Active runs",
      count: running + queued,
      note: `${running} running · ${queued} queued`,
      icon: FiActivity,
    },
    // Completed measures successful execution, not evaluated answer quality.
    {
      label: "Completed runs",
      count: runs.filter((run) => run.status === "completed").length,
      note: "Finished without execution errors",
      icon: FiCheckCircle,
    },
    // Failed runs can still contain saved results from questions completed before the error.
    {
      label: "Failed runs",
      count: runs.filter((run) => run.status === "failed").length,
      note: "Stopped before all questions finished",
      icon: FiAlertCircle,
    },
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
      // Clipboard access can be denied independently of run-history access.
      setNotice(`Unable to copy. Run ID: ${id}`);
    }
  }

  /** Download filtered API summaries as JSON; returns nothing without fetching detail payloads. */
  function exportRuns() {
    const data = filtered.map((run) => run.source);
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "runs.json";
    link.click();
    // Release the object URL after the browser has started the download.
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setNotice(`Exported ${filtered.length} run summaries.`);
  }

  return (
    <main className={styles.shell}>
      {/* Reuse the application's navigation rather than the exported Stitch shell. */}
      <WorkbenchSidebar activeLabel="Runs" />
      <WorkbenchGridCanvas className={styles.workspace}>
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
            <button onClick={exportRuns} disabled={loading || filtered.length === 0}>
              <FiDownload aria-hidden="true" /> Export
            </button>
            <Link href="/experiments" className={styles.primary}>
              New experiment
            </Link>
          </div>
        </header>
        {/* Refresh failures preserve prior rows and offer an immediate retry. */}
        {error && (
          <div className={styles.preview} role="alert">
            <FiAlertCircle aria-hidden="true" />
            {error}
            <button onClick={retry}>
              Retry
            </button>
          </div>
        )}
        {/* Summary counts cover the complete loaded inventory, independent of filters. */}
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
                {loading || (error && snapshots.length === 0) ? "—" : count}
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
          Summary counts include all saved runs. History refreshes automatically.
        </p>
        {/* Search, independent filters, and sorting operate on the loaded summaries. */}
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
                    [...new Set(runs.map((run) =>
                      key === "index" ? run.indexId : run.datasetId))]).map((value) => (
                  <option key={value} value={value}>
                    {key === "days" ? `Last ${value} days` : key === "index" ?
                      `${runs.find((r) => r.indexId === value)?.index} (${value.slice(0, 8)})` :
                      key === "dataset" ?
                        `${runs.find((r) => r.datasetId === value)?.dataset}` +
                          ` (${value.slice(0, 8)})` :
                        value}
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
                {loading ? "Loading runs…" : error && !runs.length ? "History unavailable" :
                  runs.length === 0 ? "No runs yet" : "No matching runs"}
              </h2>
              <p>
                {loading ? "Fetching saved benchmarks." : error && !runs.length ?
                  "Use Retry above to load your history." : runs.length === 0 ?
                  "Launch a benchmark from Experiments to see it here." :
                  "Try another search or clear your filters."}
              </p>
              <button onClick={clearFilters}>
                Clear filters
              </button>
            </div>
          )}
          <footer className={styles.pagination}>
            <span>
              Page {currentPage} of {pageCount}
            </span>
            <div className={styles.actions}>
              <button disabled={currentPage === 1} onClick={() => setPage(currentPage - 1)}>
                Previous
              </button>
              <button disabled={currentPage >= pageCount} onClick={() => setPage(currentPage + 1)}>
                Next
              </button>
            </div>
          </footer>
        </section>
        <p role="status" className={styles.notice}>
          {notice}
        </p>
      </WorkbenchGridCanvas>
    </main>
  );
}
