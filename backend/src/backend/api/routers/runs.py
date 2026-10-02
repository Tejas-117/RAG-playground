"""HTTP routes for enqueueing and inspecting dataset benchmark runs."""

import logging
import sqlite3
from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.api.routers.pipeline_options import _load_pipeline_options
from backend.db.repositories.benchmark_runs import (
    BenchmarkInputMismatchError,
    BenchmarkInputNotFoundError,
    BenchmarkRunNotFoundError,
    create_pending_benchmark_run,
    get_benchmark_run,
    list_benchmark_runs,
    resolve_benchmark_configuration,
)
from backend.db.repositories.retrieval_evaluations import (
    EvaluationNotFoundError,
    EvaluationRunStateError,
    create_evaluation,
    get_evaluation,
    list_evaluations,
)
from backend.pipeline.compatibility import (
    InvalidPipelineConfigurationError,
    validate_pipeline_config,
)
from backend.pipeline.configs import ExperimentConfig, PipelineConfig

router = APIRouter()

# Use the module name so API-boundary run records remain distinguishable.
logger = logging.getLogger(__name__)

# Expose persisted lifecycle values as a closed API contract.
RunStatus = Literal["pending", "running", "completed", "failed"]

# Each stage has its own state so the UI never invents progress percentages.
StageStatus = Literal["pending", "running", "completed", "failed"]


class BenchmarkRunCreateRequest(BaseModel):
    """Represent saved artifacts and query-time settings for one benchmark.

    Attributes:
        prepared_index_id: Ready named index reused without preparation work.
        dataset_id: Immutable dataset supplying all ordered questions.
        configuration: Retrieval, generation, and future evaluation settings.
    """

    # Reject removed ad-hoc fields instead of silently ignoring stale clients.
    model_config = ConfigDict(extra="forbid")

    prepared_index_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    configuration: ExperimentConfig


class RunRetrievedChunkResponse(BaseModel):
    """Expose one ranked retrieved chunk with source and score provenance.

    Attributes:
        rank: One-based nearest-neighbor position.
        chunk_id: Stable application chunk identifier.
        raw_distance: Unmodified vector-store distance.
        source_document_id: Stable source document identifier.
        original_filename: User-visible source filename.
        ordinal: Zero-based position within the source document's chunk set.
        text: Exact persisted chunk text supplied as possible generation context.
        character_start_offset: Inclusive canonical-text character offset.
        character_end_offset: Exclusive canonical-text character offset.
        token_start_offset: Inclusive canonical-text token offset when exact.
        token_end_offset: Exclusive canonical-text token offset when exact.
        page_start: First intersected physical page when available.
        page_end: Last intersected physical page when available.
        section_path: Optional logical heading hierarchy.
        source_metadata: Parser and source-block provenance.
    """

    rank: int = Field(gt=0)
    chunk_id: str = Field(min_length=1)
    raw_distance: float
    source_document_id: str = Field(min_length=1)
    original_filename: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    text: str
    character_start_offset: int | None = Field(default=None, ge=0)
    character_end_offset: int | None = Field(default=None, ge=0)
    token_start_offset: int | None = Field(default=None, ge=0)
    token_end_offset: int | None = Field(default=None, ge=0)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section_path: list[str] | None = None
    source_metadata: dict[str, object] = Field(default_factory=dict)


class RunRetrievalResponse(BaseModel):
    """Describe retrieval state and its immutable ranked result when available.

    Attributes:
        status: Current lifecycle state of retrieval.
        result_id: Persisted retrieval-result identifier after success.
        requested_top_k: Configured maximum number of returned chunks.
        returned_count: Actual number of persisted ranked chunks.
        distance_metric: Raw-distance semantics of every returned score.
        duration_ms: Retrieval-stage wall-clock duration.
        chunks: Ranked hydrated result chunks after retrieval succeeds.
    """

    status: StageStatus
    result_id: str | None = None
    requested_top_k: int = Field(gt=0)
    returned_count: int | None = Field(default=None, ge=0)
    distance_metric: Literal["cosine", "dot_product", "euclidean"]
    duration_ms: int | None = Field(default=None, ge=0)
    chunks: list[RunRetrievedChunkResponse] = Field(default_factory=list)


class RunGenerationContextResponse(BaseModel):
    """Identify one retrieval rank included in the generated-answer prompt.

    Attributes:
        ordinal: One-based position within the prompt context.
        retrieval_rank: Original one-based retrieval-result rank.
        chunk_id: Stable chunk identifier included in the prompt.
    """

    ordinal: int = Field(gt=0)
    retrieval_rank: int = Field(gt=0)
    chunk_id: str = Field(min_length=1)


class RunGenerationResponse(BaseModel):
    """Describe generation state, answer, usage, and prompt provenance.

    Attributes:
        status: Current lifecycle state of generation.
        result_id: Persisted generation-result identifier after success.
        retrieval_result_id: Exact retrieval output used to construct the prompt.
        provider: Backend-registered generation provider identifier.
        model: Requested provider model identifier.
        provider_model: Provider-reported model identifier when available.
        answer: Generated answer text after success.
        finish_reason: Provider reason for ending the completion.
        prompt_template_version: Versioned backend prompt policy.
        provider_policy_version: Versioned provider-request policy.
        prompt_tokens: Provider-reported input token count.
        completion_tokens: Provider-reported output token count.
        total_tokens: Provider-reported combined token count.
        provider_called: Whether generation required a remote API call.
        context_chunks: Exact retrieval ranks included in the prompt.
        duration_ms: Generation-stage wall-clock duration.
    """

    status: StageStatus
    result_id: str | None = None
    retrieval_result_id: str | None = None
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    provider_model: str | None = None
    answer: str | None = None
    finish_reason: str | None = None
    prompt_template_version: str | None = None
    provider_policy_version: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    provider_called: bool | None = None
    context_chunks: list[RunGenerationContextResponse] = Field(default_factory=list)
    duration_ms: int | None = Field(default=None, ge=0)


class RunErrorResponse(BaseModel):
    """Expose a safe terminal pipeline failure to polling clients.

    Attributes:
        code: Stable machine-readable failure identifier.
        message: Safe user-readable explanation.
        stage: Pipeline stage that failed when one had started.
        details: Additional safe structured provider or validation context.
    """

    code: str
    message: str
    stage: Literal["chunking", "embedding", "retrieval", "generation"] | None = None
    details: dict[str, object] = Field(default_factory=dict)


class BenchmarkExampleResponse(BaseModel):
    """Expose one internal example execution beneath its parent benchmark.

    Attributes:
        id: Stable internal execution identifier.
        example_id: Immutable evaluation-example identifier.
        ordinal: Zero-based dataset order.
        question: Exact immutable dataset question.
        reference_answer: Optional ground-truth answer retained for evaluation.
        status: Independent child lifecycle state.
        current_stage: Active query-time stage when running.
        started_at: Worker start timestamp when available.
        completed_at: Terminal timestamp when available.
        duration_ms: Total child execution duration when terminal.
        retrieval: Persisted ranked output after retrieval succeeds.
        generation: Persisted answer after generation succeeds.
        error: Safe child failure when this example terminates the benchmark.
    """

    id: str = Field(min_length=1)
    example_id: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    question: str = Field(min_length=1)
    reference_answer: str | None = None
    status: RunStatus
    current_stage: Literal["retrieval", "generation"] | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    retrieval: RunRetrievalResponse | None = None
    generation: RunGenerationResponse | None = None
    error: RunErrorResponse | None = None


class BenchmarkRunMetricsResponse(BaseModel):
    """Expose successful stage averages and nullable usage with coverage counts."""

    retrieval_result_count: int = Field(ge=0)
    generation_result_count: int = Field(ge=0)
    average_retrieval_duration_ms: float | None = Field(ge=0)
    average_generation_duration_ms: float | None = Field(ge=0)
    prompt_tokens: int | None = Field(ge=0)
    completion_tokens: int | None = Field(ge=0)
    prompt_token_result_count: int = Field(ge=0)
    completion_token_result_count: int = Field(ge=0)


class EvaluationRequest(BaseModel):
    """Select metric categories, or use each run snapshot category when omitted.

    Attributes:
        retrieval_metrics: Optional metric override for the new scoring attempt.
        answer_metrics: Optional answer metric override for the new scoring attempt.
    """

    model_config = ConfigDict(extra="forbid")
    retrieval_metrics: list[Literal["hit_rate_at_k", "recall_at_k", "mrr"]] | None = (
        None
    )
    answer_metrics: (
        list[Literal["groundedness", "answer_relevance", "answer_correctness"]] | None
    ) = None


class EvaluationSummaryResponse(BaseModel):
    """Describe one durable scoring attempt and its aggregate coverage.

    Attributes:
        id: Stable evaluation-attempt identifier.
        benchmark_run_id: Completed benchmark whose saved outputs are scored.
        status: Independent lifecycle state of this evaluation attempt.
        configuration: Saved metric selections and evaluator provenance.
        aggregates: Selected metric averages across successfully scored questions.
        coverage: Per-metric eligible, scored, skipped, and error counts.
        has_errors: Whether some question-level work failed while scores were retained.
        eligible_count: Compatibility count; detailed eligibility is in coverage.
        total_count: Total benchmark questions considered by the attempt.
        error: Safe structured terminal error when evaluation fails.
        created_at: Timestamp at which the attempt was queued.
        started_at: Timestamp at which a worker claimed the attempt.
        completed_at: Timestamp at which the attempt reached a terminal state.
    """

    id: str
    benchmark_run_id: str
    status: RunStatus
    configuration: dict[str, object]
    aggregates: dict[str, float | None]
    coverage: dict[str, dict[str, int]]
    has_errors: bool
    eligible_count: int | None
    total_count: int | None
    error: dict[str, str] | None
    created_at: str
    started_at: str | None
    completed_at: str | None


class EvaluationDetailResponse(EvaluationSummaryResponse):
    """Include per-question scores, skip reasons, and ranked document evidence.

    Attributes:
        questions: Ordered question-level outcomes and source-document evidence.
    """

    questions: list[dict[str, object]]


class BenchmarkRunSummaryResponse(BaseModel):
    """Represent one dataset-wide run without loading question-level output."""

    id: str = Field(min_length=1)
    prepared_index_id: str = Field(min_length=1)
    prepared_index_name: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    vector_index_id: str = Field(min_length=1)
    status: RunStatus
    current_stage: Literal["retrieval", "generation"] | None = None
    current_example_id: str | None = None
    total_examples: int = Field(gt=0)
    completed_examples: int = Field(ge=0)
    failed_examples: int = Field(ge=0)
    pending_examples: int = Field(ge=0)
    running_examples: int = Field(ge=0)
    configuration: PipelineConfig
    error: RunErrorResponse | None = None
    metrics: BenchmarkRunMetricsResponse
    latest_evaluation: EvaluationSummaryResponse | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = Field(default=None, ge=0)


class BenchmarkRunResponse(BenchmarkRunSummaryResponse):
    """Represent a benchmark configuration and all ordered example executions."""

    configuration: PipelineConfig
    examples: list[BenchmarkExampleResponse]
    error: RunErrorResponse | None = None


@router.post(
    "/runs",
    response_model=BenchmarkRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_benchmark_run(
    payload: BenchmarkRunCreateRequest,
) -> BenchmarkRunResponse:
    """Validate and enqueue one dataset benchmark without waiting for execution.

    Args:
        payload: Prepared index, dataset, and query-time configuration.

    Returns:
        Persisted pending benchmark suitable for polling through ``GET /runs/{id}``.

    Raises:
        HTTPException: If request validation or pending-run persistence fails.
    """
    # Benchmarks reuse prepared artifacts and execute every immutable dataset example.
    try:
        # Merge preparation settings from the ready index into one immutable snapshot.
        _, _, effective_configuration = resolve_benchmark_configuration(
            payload.prepared_index_id,
            payload.dataset_id,
            payload.configuration,
        )
        options = _load_pipeline_options()
        validate_pipeline_config(
            effective_configuration,
            options,
            has_evaluation_dataset=True,
        )
        benchmark = create_pending_benchmark_run(
            payload.prepared_index_id,
            payload.dataset_id,
            effective_configuration,
        )
    except BenchmarkInputNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "benchmark_input_not_found",
                "message": "The selected prepared index or dataset does not exist.",
            },
        ) from error
    except BenchmarkInputMismatchError as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "incompatible_benchmark_inputs",
                "message": str(error),
            },
        ) from error
    except InvalidPipelineConfigurationError as error:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_experiment_configuration",
                "message": error.message,
                "field": error.field,
            },
        ) from error
    except (OSError, ValidationError) as error:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "pipeline_options_unavailable",
                "message": "The experiment options could not be loaded.",
            },
        ) from error
    except sqlite3.Error as error:
        logger.exception("benchmark_run_create_failed")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "persistence_error",
                "message": "The benchmark run could not be saved.",
            },
        ) from error

    logger.info(
        "benchmark_run_enqueued run_id=%s dataset_id=%s prepared_index_id=%s",
        benchmark["id"],
        payload.dataset_id,
        payload.prepared_index_id,
    )
    return BenchmarkRunResponse.model_validate(_add_benchmark_stage_statuses(benchmark))


@router.get("/runs", response_model=list[BenchmarkRunSummaryResponse])
async def read_benchmark_runs() -> list[BenchmarkRunSummaryResponse]:
    """List user-visible dataset benchmark runs newest first.

    Returns:
        Compact benchmark summaries suitable for the Runs inventory page.

    Raises:
        HTTPException: If persisted benchmark history cannot be read.
    """
    try:
        # Summary loading intentionally excludes potentially large question results.
        runs = list_benchmark_runs()
    except sqlite3.Error as error:
        logger.exception("benchmark_run_list_failed")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "persistence_error",
                "message": "Benchmark runs could not be loaded.",
            },
        ) from error

    return [BenchmarkRunSummaryResponse.model_validate(run) for run in runs]


@router.get(
    "/runs/{run_id}",
    response_model=BenchmarkRunResponse,
)
async def read_benchmark_run(run_id: str) -> BenchmarkRunResponse:
    """Return the latest persisted state of one dataset benchmark.

    Args:
        run_id: Stable identifier returned by ``POST /runs``.

    Returns:
        Current pending, running, completed, or failed benchmark representation.

    Raises:
        HTTPException: If the run is unknown or cannot be read.
    """
    try:
        # Materialize the parent and ordered child results in one detail response.
        benchmark = get_benchmark_run(run_id)
        return BenchmarkRunResponse.model_validate(
            _add_benchmark_stage_statuses(benchmark)
        )
    except BenchmarkRunNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "run_not_found",
                "message": "The selected benchmark run does not exist.",
            },
        ) from error
    except sqlite3.Error as error:
        logger.exception(
            "benchmark_run_read_failed run_id=%s error_code=persistence_error",
            run_id,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "persistence_error",
                "message": "The benchmark run could not be read.",
            },
        ) from error


def _add_benchmark_stage_statuses(benchmark: dict[str, object]) -> dict[str, object]:
    """Add response-only lifecycle states to persisted example results.

    Args:
        benchmark: Materialized benchmark containing ordered example dictionaries.

    Returns:
        Shallow benchmark copy whose result objects satisfy the shared API schemas.
    """
    response = dict(benchmark)
    examples: list[dict[str, object]] = []

    # Derive stage state from each child lifecycle without persisting redundant values.
    for raw_example in benchmark.get("examples", []):
        example = dict(raw_example)
        retrieval = example.get("retrieval")
        generation = example.get("generation")

        # A persisted result is complete; otherwise child state determines visibility.
        if retrieval is not None:
            example["retrieval"] = {"status": "completed", **retrieval}

        if generation is not None:
            example["generation"] = {"status": "completed", **generation}

        examples.append(example)

    response["examples"] = examples
    return response


@router.post(
    "/runs/{run_id}/evaluations",
    response_model=EvaluationDetailResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_run_evaluation(
    run_id: str, payload: EvaluationRequest | None = None
) -> EvaluationDetailResponse:
    """Queue independent scoring of a completed benchmark's saved outputs.

    Args:
        run_id: Stable identifier of the completed benchmark to score.
        payload: Optional retrieval and answer metric overrides for this attempt.

    Returns:
        The newly queued evaluation attempt.

    Raises:
        HTTPException: If the run is missing, incomplete, invalid, or cannot be saved.
    """
    # Convert repository failures into stable HTTP errors for API clients.
    try:
        selected = payload.retrieval_metrics if payload else None
        selected_answers = payload.answer_metrics if payload else None
        return EvaluationDetailResponse.model_validate(
            create_evaluation(run_id, selected, selected_answers)
        )
    except EvaluationNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "run_not_found",
                "message": "The selected benchmark run does not exist.",
            },
        ) from error
    except EvaluationRunStateError as error:
        raise HTTPException(
            status_code=409, detail={"code": "run_not_completed", "message": str(error)}
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_evaluation_metrics", "message": str(error)},
        ) from error
    except sqlite3.Error as error:
        logger.exception("evaluation_create_failed run_id=%s", run_id)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "persistence_error",
                "message": "The evaluation could not be queued.",
            },
        ) from error


@router.get(
    "/runs/{run_id}/evaluations", response_model=list[EvaluationSummaryResponse]
)
async def read_run_evaluations(run_id: str) -> list[EvaluationSummaryResponse]:
    """Return all attempts for one benchmark, newest first.

    Args:
        run_id: Stable benchmark identifier whose evaluation history is requested.

    Returns:
        Compact evaluation attempts ordered newest first.

    Raises:
        HTTPException: If the run is missing or its history cannot be read.
    """
    # Validate each repository dictionary against the public response contract.
    try:
        return [
            EvaluationSummaryResponse.model_validate(item)
            for item in list_evaluations(run_id)
        ]
    except EvaluationNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "run_not_found",
                "message": "The selected benchmark run does not exist.",
            },
        ) from error
    except sqlite3.Error as error:
        logger.exception("evaluation_list_failed run_id=%s", run_id)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "persistence_error",
                "message": "Evaluations could not be loaded.",
            },
        ) from error


@router.get(
    "/runs/{run_id}/evaluations/{evaluation_id}",
    response_model=EvaluationDetailResponse,
)
async def read_run_evaluation(
    run_id: str, evaluation_id: str
) -> EvaluationDetailResponse:
    """Return one attempt and its saved per-question scoring evidence.

    Args:
        run_id: Stable identifier of the parent benchmark.
        evaluation_id: Stable identifier of the requested scoring attempt.

    Returns:
        Evaluation detail with ordered question-level evidence.

    Raises:
        HTTPException: If the attempt is missing or cannot be read.
    """
    # Keep an attempt scoped to its parent run when resolving the resource.
    try:
        return EvaluationDetailResponse.model_validate(
            get_evaluation(run_id, evaluation_id)
        )
    except EvaluationNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "evaluation_not_found",
                "message": "The selected evaluation does not exist.",
            },
        ) from error
    except sqlite3.Error as error:
        logger.exception("evaluation_read_failed evaluation_id=%s", evaluation_id)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "persistence_error",
                "message": "The evaluation could not be loaded.",
            },
        ) from error
