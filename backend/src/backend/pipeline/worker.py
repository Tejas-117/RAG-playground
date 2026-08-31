"""Local SQLite-backed worker for persisted pipeline runs."""

import asyncio
import logging
from collections.abc import Callable

from backend.db.repositories.work_queue import claim_next_pending_work_item
from backend.pipeline.execution import (
    PipelineExecutor,
    PipelineRunExecutionError,
    get_pipeline_executor,
)
from backend.pipeline.preparation import (
    PreparedIndexExecutionError,
    PreparedIndexExecutor,
    get_prepared_index_executor,
)

# Use the module name to isolate queue and worker lifecycle records.
logger = logging.getLogger(__name__)

# A short idle delay keeps local UI latency low without continuously polling SQLite.
RUN_QUEUE_POLL_INTERVAL_SECONDS = 0.5


class PipelineRunWorker:
    """Claim all persisted pipeline jobs sequentially outside the event loop."""

    def __init__(
        self,
        executor_factory: Callable[[], PipelineExecutor] = get_pipeline_executor,
        prepared_index_executor_factory: Callable[[], PreparedIndexExecutor] = (
            get_prepared_index_executor
        ),
    ) -> None:
        """Configure the stateless executor factory used for each claimed run.

        Args:
            executor_factory: Callable returning a full-run executor.
            prepared_index_executor_factory: Callable returning a preparation executor.

        Returns:
            None. Queue and run state remain in SQLite.
        """
        self._executor_factory = executor_factory
        self._prepared_index_executor_factory = prepared_index_executor_factory

    async def run_forever(self) -> None:
        """Poll the persisted queue until the application cancels the worker.

        Args:
            None.

        Returns:
            Never returns normally; application shutdown cancels the coroutine.
        """
        logger.info("pipeline_worker_started")

        # Claim one globally oldest job at a time to avoid competing embedding workloads.
        while True:
            # The shared claim orders prepared indexes and legacy runs together.
            claimed_work = claim_next_pending_work_item()

            # Sleep only while the persisted queue has no work.
            if claimed_work is None:
                await asyncio.sleep(RUN_QUEUE_POLL_INTERVAL_SECONDS)
                continue

            work_id = claimed_work["id"]
            work_kind = claimed_work["kind"]
            logger.info(
                "pipeline_worker_work_claimed work_kind=%s work_id=%s",
                work_kind,
                work_id,
            )

            try:
                # Route the claimed durable row to its independently testable executor.
                if work_kind == "prepared_index":
                    executor = self._prepared_index_executor_factory()
                else:
                    executor = self._executor_factory()

                # Parsing, provider HTTP, Chroma, and persistence are synchronous.
                await asyncio.to_thread(executor.execute, work_id)
            except PreparedIndexExecutionError as error:
                # The preparation executor already persisted this expected failure.
                logger.warning(
                    "pipeline_worker_prepared_index_failed "
                    "prepared_index_id=%s stage=%s error_code=%s",
                    work_id,
                    error.stage,
                    error.code,
                )
            except PipelineRunExecutionError as error:
                # The executor already persisted this expected terminal failure.
                logger.warning(
                    "pipeline_worker_run_failed run_id=%s stage=%s error_code=%s",
                    work_id,
                    error.stage,
                    error.code,
                )
            except Exception:
                # Preserve the worker loop if an unexpected boundary escapes the executor.
                logger.exception(
                    "pipeline_worker_unhandled_failure work_kind=%s work_id=%s",
                    work_kind,
                    work_id,
                )
