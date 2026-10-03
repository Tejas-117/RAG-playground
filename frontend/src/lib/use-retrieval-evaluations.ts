"use client";

import { useEffect, useState } from "react";
import {
  createEvaluation,
  getEvaluation,
  listEvaluations,
} from "@/lib/benchmark-run-api";
import { isAxiosError } from "@/lib/axios";
import { parseBenchmarkRunApiError } from "@/validation/benchmark-runs";
import type {
  RetrievalEvaluationDetail,
  RetrievalEvaluationSummary,
  AnswerMetric,
  RetrievalMetric,
} from "@/validation/benchmark-runs";

// Active scoring is local and deterministic, so a two-second refresh keeps progress responsive.
const ACTIVE_REFRESH_MS = 2000;

/**
 * Load evaluation history, own the selected attempt, and queue independent reevaluations.
 *
 * @param runId - Stable benchmark identifier whose attempts should be managed.
 * @param enabled - Whether the parent run has loaded and requests may begin.
 * @param latestMarker - Latest attempt identity and status from the parent run snapshot.
 * @returns Evaluation state, selection controls, retry, and the reevaluation action.
 */
export function useEvaluations(
  runId: string,
  enabled: boolean,
  latestMarker: string,
) {
  // History powers the attempt ledger and identifies any work still being processed.
  const [attempts, setAttempts] = useState<RetrievalEvaluationSummary[]>([]);

  // Selection remains stable while history refreshes so older attempts can be inspected.
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Detail contains aggregates plus ordered evidence for the currently selected attempt.
  const [detail, setDetail] = useState<RetrievalEvaluationDetail | null>(null);

  // Initial loading is separate from background polling to prevent content flicker.
  const [loading, setLoading] = useState(true);

  // Read failures retain earlier valid history and expose a manual retry.
  const [error, setError] = useState("");

  // Submission feedback is separate because creating an attempt does not invalidate history.
  const [createError, setCreateError] = useState("");

  // The creating flag prevents duplicate requests from repeated button activation.
  const [creating, setCreating] = useState(false);

  // Incrementing this value restarts the complete request lifecycle after a failure.
  const [revision, setRevision] = useState(0);

  // One lifecycle owns its request, timer, and visibility listener.
  useEffect(() => {
    let disposed = false;
    let controller: AbortController | undefined;
    let timer: ReturnType<typeof setTimeout> | undefined;

    /**
     * Fetch current history and the selected attempt.
     *
     * @returns A promise resolved after data is stored and active work is scheduled.
     */
    async function refresh(): Promise<void> {
      // Disabled and hidden views defer work until their parent or tab becomes active.
      if (!enabled || disposed || document.hidden) {
        return;
      }

      const request = new AbortController();
      controller = request;

      try {
        const history = await listEvaluations(runId, request.signal);

        // Ignore a response from an obsolete lifecycle.
        if (disposed || request.signal.aborted) {
          return;
        }

        const selectedExists = history.some((attempt) => attempt.id === selectedId);
        const targetId = selectedExists ? selectedId : history[0]?.id ?? null;
        setAttempts(history);

        // Empty history has no detail resource to request.
        if (targetId === null) {
          setSelectedId(null);
          setDetail(null);
        } else {
          const selectedDetail = await getEvaluation(
            runId,
            targetId,
            request.signal,
          );

          // A hidden or replaced request cannot update the selected evidence.
          if (disposed || request.signal.aborted) {
            return;
          }

          setSelectedId(targetId);
          setDetail(selectedDetail);
        }

        setError("");

        // Continue only while at least one durable evaluation attempt is active.
        const hasActiveAttempt = history.some((attempt) =>
          attempt.status === "pending" || attempt.status === "running"
        );
        if (hasActiveAttempt) {
          timer = setTimeout(refresh, ACTIVE_REFRESH_MS);
        }
      } catch {
        // Cancellation is expected during navigation, selection, and tab hiding.
        if (disposed || request.signal.aborted) {
          return;
        }

        setError("Unable to load evaluation history. Saved run results remain available.");
      } finally {
        // Only the current lifecycle may finish the initial loading state.
        if (!disposed && !request.signal.aborted) {
          setLoading(false);
        }
      }
    }

    /**
     * Cancel hidden requests and refresh immediately when the document becomes visible.
     *
     * @returns Nothing; the current request lifecycle is updated in place.
     */
    function visibilityChanged(): void {
      clearTimeout(timer);
      controller?.abort();

      // Returning to the page should show the newest durable attempt state.
      if (!document.hidden) {
        void refresh();
      }
    }

    void refresh();
    document.addEventListener("visibilitychange", visibilityChanged);

    // Cleanup prevents late network responses from changing unmounted state.
    return () => {
      disposed = true;
      clearTimeout(timer);
      controller?.abort();
      document.removeEventListener("visibilitychange", visibilityChanged);
    };
  }, [enabled, latestMarker, revision, runId, selectedId]);

  /**
   * Queue a new evaluation and select it immediately.
   *
   * @param retrievalMetrics - Retrieval metric selection for the new attempt.
   * @param answerMetrics - Answer metric selection for the new attempt.
   * @returns The created attempt, or null when the request fails.
   */
  async function evaluate(
    retrievalMetrics: RetrievalMetric[],
    answerMetrics: AnswerMetric[],
  ): Promise<RetrievalEvaluationDetail | null> {
    setCreating(true);
    setCreateError("");

    try {
      const created = await createEvaluation(
        runId,
        retrievalMetrics,
        answerMetrics,
      );

      // Place the new attempt first without waiting for the next history refresh.
      setAttempts((current) => [
        created,
        ...current.filter((attempt) => attempt.id !== created.id),
      ]);
      setSelectedId(created.id);
      setDetail(created);
      setRevision((value) => value + 1);
      return created;
    } catch (cause) {
      // Prefer the backend's structured message and fall back to a stable local explanation.
      const message = isAxiosError(cause)
        ? parseBenchmarkRunApiError(cause.response?.data)
        : null;
      setCreateError(message ?? "The evaluation could not be queued.");
      return null;
    } finally {
      setCreating(false);
    }
  }

  /**
   * Select one durable attempt and restart detail loading.
   *
   * @param evaluationId - Stable attempt identifier selected from history.
   * @returns Nothing; selection state triggers a new detail request.
   */
  function selectAttempt(evaluationId: string): void {
    // Re-selecting the loaded attempt must preserve its detail because the ID will not change.
    if (evaluationId === selectedId) {
      return;
    }

    setSelectedId(evaluationId);
    setDetail(null);
  }

  /**
   * Restart evaluation loading after a transport or validation failure.
   *
   * @returns Nothing; incrementing the revision starts a new request lifecycle.
   */
  function retry(): void {
    setRevision((value) => value + 1);
  }

  return {
    attempts,
    selectedId,
    detail,
    loading,
    error,
    createError,
    creating,
    selectAttempt,
    evaluate,
    retry,
  };
}

/** Public return shape shared by the evaluation workspace and run-detail page. */
export type EvaluationsState = ReturnType<typeof useEvaluations>;
