"""Execute one dataset benchmark against an already prepared vector index."""

import logging
from time import perf_counter
from typing import Any

from backend.db.repositories.benchmark_runs import (
    complete_benchmark_example,
    complete_benchmark_run,
    fail_benchmark_run,
    get_benchmark_execution_input,
    record_benchmark_retrieval,
    start_benchmark_example,
)
from backend.embedding.models import EmbeddingProvider, VectorStore
from backend.generation.models import GenerationProvider
from backend.generation.service import generate_answer
from backend.ingestion.chunkers.models import ChunkingTokenizer
from backend.pipeline.execution import (
    AnswerGenerator,
    ChunkRetriever,
    _elapsed_milliseconds,
    _map_execution_error,
)
from backend.retrieval.service import retrieve_chunks

# Use this module name to keep benchmark events separate from legacy pipeline logs.
logger = logging.getLogger(__name__)


class BenchmarkExecutor:
    """Run ordered dataset examples through retrieval and generation only."""

    def __init__(
        self,
        chunk_retriever: ChunkRetriever = retrieve_chunks,
        answer_generator: AnswerGenerator = generate_answer,
        tokenizer: ChunkingTokenizer | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        generation_provider: GenerationProvider | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        """Configure independently replaceable query-time stage dependencies.

        Args:
            chunk_retriever: Service that embeds a question and hydrates search hits.
            answer_generator: Service that generates from one ranked context.
            tokenizer: Optional deterministic prompt-budget tokenizer override.
            embedding_provider: Optional query-embedding adapter override.
            generation_provider: Optional answer-provider adapter override.
            vector_store: Optional vector-search adapter override.
        """
        # Retain stateless adapters so one executor can process all ordered examples.
        self._chunk_retriever = chunk_retriever
        self._answer_generator = answer_generator
        self._tokenizer = tokenizer
        self._embedding_provider = embedding_provider
        self._generation_provider = generation_provider
        self._vector_store = vector_store

    def execute(self, benchmark_run_id: str) -> dict[str, Any]:
        """Execute every dataset example sequentially against one ready index.

        Args:
            benchmark_run_id: Stable identifier of an already claimed benchmark.

        Returns:
            Completed benchmark with every question-level result.

        Raises:
            PipelineRunExecutionError: If one example fails after safe persistence.
        """
        execution_input = get_benchmark_execution_input(benchmark_run_id)
        configuration = execution_input["configuration"]
        vector_index = execution_input["vector_index"]
        run_started_counter = perf_counter()
        active_example_run_id: str | None = None
        example_started_counter: float | None = None
        current_stage = "retrieval"
        logger.info(
            "benchmark_run_started run_id=%s example_count=%d",
            benchmark_run_id,
            len(execution_input["examples"]),
        )

        try:
            # Dataset order is intentional and produces stable provider request order.
            for example in execution_input["examples"]:
                active_example_run_id = example["example_run_id"]
                example_started_counter = perf_counter()
                start_benchmark_example(
                    benchmark_run_id,
                    active_example_run_id,
                    example["example_id"],
                )

                # Search only the exact technical index linked by the prepared index.
                current_stage = "retrieval"
                retrieval_started_counter = perf_counter()
                hits = self._chunk_retriever(
                    example["question"],
                    configuration.retrieval,
                    configuration.embedding,
                    vector_index,
                    self._embedding_provider,
                    self._vector_store,
                )
                retrieval_duration_ms = _elapsed_milliseconds(retrieval_started_counter)
                retrieval_result_id = record_benchmark_retrieval(
                    benchmark_run_id,
                    active_example_run_id,
                    vector_index["id"],
                    configuration.retrieval.top_k,
                    configuration.embedding.distance_metric.value,
                    hits,
                    retrieval_duration_ms,
                )

                # Generate one answer from the exact ranking persisted above.
                current_stage = "generation"
                generation_started_counter = perf_counter()
                answer = self._answer_generator(
                    example["question"],
                    configuration.generation,
                    hits,
                    self._generation_provider,
                    self._tokenizer,
                )
                generation_duration_ms = _elapsed_milliseconds(
                    generation_started_counter
                )
                complete_benchmark_example(
                    benchmark_run_id,
                    active_example_run_id,
                    retrieval_result_id,
                    configuration,
                    answer,
                    generation_duration_ms,
                    _elapsed_milliseconds(example_started_counter),
                )
                logger.info(
                    "benchmark_example_completed run_id=%s example_id=%s "
                    "ordinal=%d retrieval_duration_ms=%d generation_duration_ms=%d",
                    benchmark_run_id,
                    example["example_id"],
                    example["ordinal"],
                    retrieval_duration_ms,
                    generation_duration_ms,
                )

            completed_run = complete_benchmark_run(
                benchmark_run_id,
                _elapsed_milliseconds(run_started_counter),
            )
            logger.info("benchmark_run_completed run_id=%s", benchmark_run_id)
            return completed_run
        except Exception as error:
            # Reuse provider-neutral error categories already established by each stage.
            execution_error = _map_execution_error(
                benchmark_run_id,
                current_stage,
                error,
            )
            duration_ms = _elapsed_milliseconds(run_started_counter)
            example_duration_ms = (
                _elapsed_milliseconds(example_started_counter)
                if example_started_counter is not None
                else 0
            )
            fail_benchmark_run(
                benchmark_run_id,
                active_example_run_id,
                execution_error.code,
                {
                    "stage": current_stage,
                    "message": execution_error.message,
                    **execution_error.details,
                },
                duration_ms,
                example_duration_ms,
            )
            logger.exception(
                "benchmark_run_failed run_id=%s stage=%s error_code=%s",
                benchmark_run_id,
                current_stage,
                execution_error.code,
            )
            raise execution_error from error


def get_benchmark_executor() -> BenchmarkExecutor:
    """Return a fresh production benchmark executor with lazy default adapters."""
    # The executor stores dependencies but no mutable run-specific lifecycle state.
    return BenchmarkExecutor()
