"""HTTP routes for enqueueing and polling immutable pipeline runs."""

import logging
import sqlite3
from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, ValidationError

from backend.api.routers.pipeline_options import _load_pipeline_options
from backend.db.repositories.runs import (
    CorpusNotFoundError,
    RunNotFoundError,
    create_pending_run,
    get_run,
)
from backend.pipeline.compatibility import (
    InvalidPipelineConfigurationError,
    validate_pipeline_config,
)
from backend.pipeline.configs import PipelineConfig

router = APIRouter()

# Use the module name so API-boundary run records remain distinguishable.
logger = logging.getLogger(__name__)

# Expose persisted lifecycle values as a closed API contract.
RunStatus = Literal["pending", "running", "completed", "failed"]

# Each stage has its own state so the UI never invents progress percentages.
StageStatus = Literal["pending", "running", "completed", "failed"]


class RunCreateRequest(BaseModel):
    """Represent the user input required to enqueue one pipeline run.

    Attributes:
        corpus_id: Stable identifier of the selected immutable corpus.
        question: Ad hoc question retained for later retrieval and generation.
        configuration: Complete typed pipeline configuration for the run.
    """

    corpus_id: str = Field(min_length=1)
    question: str
    configuration: PipelineConfig


class RunChunkingResponse(BaseModel):
    """Describe chunking state and its reusable artifact when available.

    Attributes:
        status: Current lifecycle state of the chunking stage.
        chunk_set_id: Ready chunk artifact identifier after chunking succeeds.
        chunk_count: Number of chunks in the ready artifact.
        reused: Whether this run reused an existing compatible artifact.
        duration_ms: Time this run spent resolving the chunking stage.
    """

    status: StageStatus
    chunk_set_id: str | None = None
    chunk_count: int | None = Field(default=None, ge=0)
    reused: bool | None = None
    duration_ms: int | None = Field(default=None, ge=0)


class RunEmbeddingResponse(BaseModel):
    """Describe embedding state, configuration, and artifact when available.

    Attributes:
        status: Current lifecycle state of the embedding stage.
        vector_index_id: Ready vector-index identifier after embedding succeeds.
        vector_count: Number of vectors stored in the ready index.
        dimensions: Width of every vector in the index.
        provider: Backend-registered provider from the immutable run snapshot.
        model: Provider model identifier from the immutable run snapshot.
        distance_metric: Distance space used by the vector collection.
        reused: Whether this run reused a compatible ready vector index.
        duration_ms: Time this run spent resolving the embedding stage.
    """

    status: StageStatus
    vector_index_id: str | None = None
    vector_count: int | None = Field(default=None, ge=0)
    dimensions: int | None = Field(default=None, gt=0)
    provider: str
    model: str
    distance_metric: Literal["cosine", "dot_product", "euclidean"]
    reused: bool | None = None
    duration_ms: int | None = Field(default=None, ge=0)


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


class RunResponse(BaseModel):
    """Represent one persisted run at any queue or execution state.

    Attributes:
        id: Stable application-generated run identifier.
        corpus_id: Stable identifier of the selected immutable corpus.
        question: Normalized question saved for the run.
        configuration: Resolved immutable configuration snapshot.
        status: Overall persisted lifecycle state.
        current_stage: Stage currently executing, or ``None`` when inactive.
        created_at: UTC timestamp when the run was enqueued.
        started_at: UTC timestamp when a worker claimed the run.
        completed_at: UTC timestamp when the run reached a terminal state.
        duration_ms: Total execution duration for a terminal run.
        chunking: Current chunking state and optional artifact summary.
        embedding: Current embedding state and optional artifact summary.
        retrieval: Current retrieval state and ranked result when available.
        generation: Current generation state and answer when available.
        error: Safe structured failure for a failed run.
    """

    id: str
    corpus_id: str
    question: str
    configuration: PipelineConfig
    status: RunStatus
    current_stage: (
        Literal[
            "chunking",
            "embedding",
            "retrieval",
            "generation",
        ]
        | None
    ) = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    chunking: RunChunkingResponse
    embedding: RunEmbeddingResponse
    retrieval: RunRetrievalResponse
    generation: RunGenerationResponse
    error: RunErrorResponse | None = None


@router.post(
    "/runs",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_pipeline_run(payload: RunCreateRequest) -> RunResponse:
    """Validate and enqueue one run without waiting for provider execution.

    Args:
        payload: Selected corpus, question, and complete pipeline configuration.

    Returns:
        Persisted pending run suitable for polling through ``GET /runs/{id}``.

    Raises:
        HTTPException: If request validation or pending-run persistence fails.
    """
    normalized_question = payload.question.strip()
    logger.info(
        "pipeline_run_requested corpus_id=%s chunking_strategy=%s "
        "embedding_provider=%s embedding_model=%s",
        payload.corpus_id,
        payload.configuration.chunking.strategy.value,
        payload.configuration.embedding.provider,
        payload.configuration.embedding.model,
    )

    # Reject blank questions before creating an immutable queue record.
    if not normalized_question:
        logger.warning(
            "pipeline_run_rejected corpus_id=%s error_code=invalid_question",
            payload.corpus_id,
        )
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_question",
                "message": "Question must not be blank.",
            },
        )

    try:
        # Resolve catalog compatibility before enqueueing an immutable snapshot.
        options = _load_pipeline_options()
        validate_pipeline_config(payload.configuration, options)
        persisted_run = create_pending_run(
            payload.corpus_id,
            normalized_question,
            payload.configuration,
        )
    except CorpusNotFoundError as error:
        logger.warning(
            "pipeline_run_rejected corpus_id=%s error_code=corpus_not_found",
            payload.corpus_id,
        )
        raise HTTPException(
            status_code=404,
            detail={
                "code": "corpus_not_found",
                "message": "The selected corpus does not exist.",
            },
        ) from error
    except InvalidPipelineConfigurationError as error:
        logger.warning(
            "pipeline_run_rejected corpus_id=%s "
            "error_code=invalid_pipeline_configuration field=%s",
            payload.corpus_id,
            error.field,
        )
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_pipeline_configuration",
                "message": error.message,
                "field": error.field,
            },
        ) from error
    except (OSError, ValidationError) as error:
        logger.exception(
            "pipeline_run_rejected corpus_id=%s "
            "error_code=pipeline_options_unavailable",
            payload.corpus_id,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "pipeline_options_unavailable",
                "message": "The pipeline configuration options could not be loaded.",
            },
        ) from error
    except sqlite3.Error as error:
        logger.exception(
            "pipeline_run_rejected corpus_id=%s error_code=persistence_error",
            payload.corpus_id,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "persistence_error",
                "message": "The pipeline run could not be saved.",
            },
        ) from error

    logger.info(
        "pipeline_run_enqueued run_id=%s corpus_id=%s",
        persisted_run["id"],
        payload.corpus_id,
    )
    return RunResponse.model_validate(persisted_run)


@router.get("/runs/{run_id}", response_model=RunResponse)
async def read_pipeline_run(run_id: str) -> RunResponse:
    """Return the latest persisted state of one pipeline run.

    Args:
        run_id: Stable identifier returned by ``POST /runs``.

    Returns:
        Current pending, running, completed, or failed run representation.

    Raises:
        HTTPException: If the run is unknown or cannot be read.
    """
    try:
        # This short local SQLite read contains no parsing or provider work.
        persisted_run = get_run(run_id)
    except RunNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "run_not_found",
                "message": "The selected pipeline run does not exist.",
            },
        ) from error
    except sqlite3.Error as error:
        logger.exception(
            "pipeline_run_read_failed run_id=%s error_code=persistence_error",
            run_id,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "persistence_error",
                "message": "The pipeline run could not be read.",
            },
        ) from error

    return RunResponse.model_validate(persisted_run)
