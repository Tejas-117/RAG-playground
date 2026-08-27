"""Local SQLite-backed worker for persisted pipeline runs."""

import asyncio
import logging
from collections.abc import Callable

from backend.db.repositories.runs import claim_next_pending_run
from backend.pipeline.execution import (
    PipelineExecutor,
    PipelineRunExecutionError,
    get_pipeline_executor,
)

# Use the module name to isolate queue and worker lifecycle records.
logger = logging.getLogger(__name__)

# A short idle delay keeps local UI latency low without continuously polling SQLite.
RUN_QUEUE_POLL_INTERVAL_SECONDS = 0.5


class PipelineRunWorker:
    """Claim persisted runs sequentially and execute them outside the event loop."""

    def __init__(
        self,
        executor_factory: Callable[[], PipelineExecutor] = get_pipeline_executor,
    ) -> None:
        """Configure the stateless executor factory used for each claimed run.

        Args:
            executor_factory: Callable returning an independently testable executor.

        Returns:
            None. Queue and run state remain in SQLite.
        """
        self._executor_factory = executor_factory

    async def run_forever(self) -> None:
        """Poll the persisted queue until the application cancels the worker.

        Args:
            None.

        Returns:
            Never returns normally; application shutdown cancels the coroutine.
        """
        logger.info("pipeline_worker_started")

        # Claim and execute one run at a time for the single-user local MVP.
        while True:
            # Queue claims are short SQLite transactions; avoid cancellable thread leakage.
            claimed_run = claim_next_pending_run()

            # Sleep only while the persisted queue has no work.
            if claimed_run is None:
                await asyncio.sleep(RUN_QUEUE_POLL_INTERVAL_SECONDS)
                continue

            run_id = claimed_run["id"]
            logger.info("pipeline_worker_run_claimed run_id=%s", run_id)

            try:
                # All parsing, provider HTTP, and persistence work is synchronous.
                executor = self._executor_factory()
                await asyncio.to_thread(executor.execute, run_id)
            except PipelineRunExecutionError as error:
                # The executor already persisted this expected terminal failure.
                logger.warning(
                    "pipeline_worker_run_failed run_id=%s stage=%s error_code=%s",
                    run_id,
                    error.stage,
                    error.code,
                )
            except Exception:
                # Preserve the worker loop if an unexpected boundary escapes the executor.
                logger.exception("pipeline_worker_unhandled_failure run_id=%s", run_id)
