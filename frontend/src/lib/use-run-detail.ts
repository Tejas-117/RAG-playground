"use client";

import { useEffect, useState } from "react";
import apiClient, { isAxiosError } from "@/lib/axios";
import { getBenchmarkRun } from "@/lib/benchmark-run-api";
import { parseCorpora } from "@/validation/corpora";
import type { BenchmarkRunDetail } from "@/validation/benchmark-runs";

// Active runs refresh promptly; completed and failed runs are immutable snapshots.
const ACTIVE_REFRESH_MS = 2000;

/** Load one run, poll only while active, and retain valid data on refresh failure. */
export function useRunDetail(runId: string) {
  // Last validated response remains visible if a later request fails.
  const [run, setRun] = useState<BenchmarkRunDetail | null>(null);

  // Initial loading is separate from background refreshes.
  const [loading, setLoading] = useState(true);

  // Transport and contract errors never overwrite a run's saved error.
  const [error, setError] = useState("");

  // Incrementing revision restarts the request lifecycle for manual retry.
  const [revision, setRevision] = useState(0);

  // The run contract stores a corpus ID; corpus names come from the corpus inventory.
  const [corpusNames, setCorpusNames] = useState<Record<string, string>>({});

  // A separate lookup decorates the run without holding up its execution details.
  useEffect(() => {
    const controller = new AbortController();

    /** Load validated corpus labels; missing names fall back to the stable ID. */
    async function loadNames() {
      try {
        const response = await apiClient.get<unknown>("/corpora/", {
          signal: controller.signal,
        });
        const names = Object.fromEntries(parseCorpora(response.data).map((corpus) => [
          corpus.id,
          corpus.name,
        ]));
        if (!controller.signal.aborted) setCorpusNames(names);
      } catch {
        // A lookup failure must not hide the benchmark result itself.
      }
    }

    void loadNames();
    return () => controller.abort();
  }, []);

  // One lifecycle owns its request, timer, and tab visibility listener.
  useEffect(() => {
    let disposed = false;
    let controller: AbortController | undefined;
    let timer: ReturnType<typeof setTimeout> | undefined;

    /** Fetch a snapshot and schedule only active work; returns a promise. */
    async function refresh() {
      // A hidden tab resumes immediately through the visibility listener.
      if (disposed || document.hidden) return;

      const request = new AbortController();
      controller = request;

      try {
        const snapshot = await getBenchmarkRun(runId, request.signal);
        // An old request must not overwrite a newer lifecycle's result.
        if (disposed || request.signal.aborted) return;

        setRun(snapshot);
        setError("");

        // Continue while benchmark execution or its latest independent evaluation is active.
        const evaluationActive = snapshot.latest_evaluation?.status === "pending"
          || snapshot.latest_evaluation?.status === "running";
        if (
          snapshot.status === "pending"
          || snapshot.status === "running"
          || evaluationActive
        ) {
          timer = setTimeout(refresh, ACTIVE_REFRESH_MS);
        }
      } catch (cause) {
        // An expected cancellation does not indicate a failed load.
        if (disposed || request.signal.aborted) return;

        setError(isAxiosError(cause) && cause.response?.status === 404
          ? "This run could not be found."
          : "Unable to load run details. Previously loaded results may be outdated.");
      } finally {
        // Only the live request may end the initial loading state.
        if (!disposed && !request.signal.aborted) setLoading(false);
      }
    }

    /** Cancel hidden work or fetch immediately when the tab becomes visible. */
    function visibilityChanged() {
      clearTimeout(timer);
      controller?.abort();
      if (!document.hidden) void refresh();
    }

    void refresh();
    document.addEventListener("visibilitychange", visibilityChanged);

    // Removing listeners and aborting requests prevents updates after unmount.
    return () => {
      disposed = true;
      clearTimeout(timer);
      controller?.abort();
      document.removeEventListener("visibilitychange", visibilityChanged);
    };
  }, [runId, revision]);

  return {
    run,
    corpusName: run ? corpusNames[run.corpus_id] ?? run.corpus_id : null,
    loading,
    error,
    retry: () => setRevision((value) => value + 1),
  };
}
