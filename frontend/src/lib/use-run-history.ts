/**
 * Keeps the Runs page synchronized with the backend's saved benchmark history.
 *
 * useRunHistory() takes no arguments. On mount, it calls GET /runs through the
 * API client, which validates the response before this hook stores it in React state.
 * This only observes execution: it does not start jobs or poll SQLite directly.
 *
 * After each request finishes, the hook schedules the next request. The delay is
 * 2 seconds when the latest successful snapshot contains pending/running work,
 * otherwise 15 seconds so newly created runs can still be discovered while idle.
 * Scheduling after completion avoids overlapping requests during normal polling.
 *
 * Hiding the tab cancels its request and scheduled refresh. Returning to the tab
 * fetches immediately. Unmounting or retrying cleans up the old request, timer and
 * visibility listener; late responses from that lifecycle cannot overwrite state.
 *
 * A failed refresh keeps previously loaded runs visible and exposes a safe error.
 * Retry starts a new polling lifecycle without clearing those runs. `loading`
 * describes the initial fetch, not every background refresh. A successful fetch
 * replaces the snapshot and clears the error.
 *
 * A separate one-second browser clock updates elapsed-time labels while visible.
 * It makes no API calls and does not change any persisted run duration or status.
 *
 * Returns: `runs` (validated summaries), `loading` (initial-load state), `error`
 * (refresh feedback), `now` (local milliseconds), and `retry` (manual refresh).
 */
"use client";

import { useEffect, useState } from "react";
import { listBenchmarkRuns } from "@/lib/benchmark-run-api";
import type { BenchmarkRunSummary } from "@/validation/benchmark-runs";

// Active work needs responsive progress; idle history needs only discovery of new runs.
const ACTIVE_REFRESH_MS = 2000;
const IDLE_REFRESH_MS = 15000;

/** Load history with visibility-aware polling; returns data, errors, clock and retry. */
export function useRunHistory() {
  // Preserve the last valid snapshot if a refresh fails.
  const [runs, setRuns] = useState<BenchmarkRunSummary[]>([]);

  // Separate the initial load from background refreshes so rows never flicker away.
  const [loading, setLoading] = useState(true);

  // Safe request feedback is independent from run-level execution errors.
  const [error, setError] = useState("");

  // A retry revision restarts the request lifecycle without discarding the snapshot.
  const [revision, setRevision] = useState(0);

  // Advance elapsed labels locally instead of issuing extra API requests.
  const [now, setNow] = useState(() => Date.now());

  // Own each request and timeout; cancellation prevents stale writes after navigation.
  useEffect(() => {
    // stores if the current lifecycle has been cleaned up
    let disposed = false;

    let controller: AbortController | undefined;

    // identifies scheduled refresh so it can be cancelled
    let timer: ReturnType<typeof setTimeout> | undefined;

    // track whether the latest successful response contains pending or running jobs
    let active = false;

    /** Fetch one snapshot; schedules the next request only after this one finishes. */
    async function refresh() {
      // Hidden pages make no requests; the visibility listener restarts them later.
      if (disposed || document.hidden) return;
      
      const request = new AbortController();
      controller = request;

      try {
        const snapshot = await listBenchmarkRuns(request.signal);
        // Ignore responses that raced with hide, retry or unmount.
        if (disposed || request.signal.aborted) return;

        setRuns(snapshot);
        setError("");
        active = snapshot.some((run) => ["pending", "running"].includes(run.status));
      } catch {
        // Aborts are expected lifecycle events, not user-facing failures.
        if (disposed || request.signal.aborted) return;
        
        setError("Unable to refresh run history. Previously loaded results may be outdated.");
      } finally {
        // Only the current visible lifecycle is allowed to enqueue another request.
        if (!disposed && !request.signal.aborted) {
          setLoading(false);
          timer = setTimeout(refresh, active ? ACTIVE_REFRESH_MS : IDLE_REFRESH_MS);
        }
      }
    }

    /** Stop hidden work or immediately fetch when visible; takes no arguments. */
    function visibilityChanged() {
      clearTimeout(timer);
      controller?.abort();

      // Refresh on return, even when the previous snapshot had no active jobs.
      if (!document.hidden) void refresh();
    }

    void refresh();
    document.addEventListener("visibilitychange", visibilityChanged);
    
    return () => {
      disposed = true;
      clearTimeout(timer);
      controller?.abort();
      document.removeEventListener("visibilitychange", visibilityChanged);
    };
  }, [revision]);

  // The one-second display clock does not poll the backend and skips hidden tabs.
  useEffect(() => {
    const timer = setInterval(() => {
      // Hidden pages do not need display updates.
      if (!document.hidden) setNow(Date.now());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  return { runs, loading, error, now, retry: () => setRevision((value) => value + 1) };
}
