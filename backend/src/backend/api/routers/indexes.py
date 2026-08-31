"""HTTP routes for creating and polling named prepared indexes."""

import logging
import sqlite3
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field, ValidationError

from backend.api.routers.pipeline_options import _load_pipeline_options
from backend.db.repositories.prepared_indexes import (
    PreparedIndexCorpusNotFoundError,
    PreparedIndexNotFoundError,
    create_pending_prepared_index,
    get_prepared_index,
    list_prepared_indexes,
)
from backend.pipeline.compatibility import (
    InvalidPipelineConfigurationError,
    validate_preparation_config,
)
from backend.pipeline.configs import PreparationConfig

# Keep all user-facing named-index endpoints under one resource path.
router = APIRouter(prefix="/indexes", tags=["indexes"])

# Use the module name to distinguish API events from executor lifecycle logs.
logger = logging.getLogger(__name__)

# Prepared indexes have a durable lifecycle independent of technical artifacts.
PreparedIndexStatus = Literal["pending", "running", "ready", "failed"]

# Each visible stage reports its own lifecycle without invented percentages.
PreparedIndexStageStatus = Literal["pending", "running", "completed", "failed"]


class PreparedIndexCreateRequest(BaseModel):
    """Represent the inputs required to enqueue one named preparation request.

    Attributes:
        name: User-facing label; duplicate labels are intentionally permitted.
        corpus_id: Stable immutable corpus selected for preparation.
        configuration: Complete chunking and embedding configuration.
    """

    name: str = Field(max_length=100)
    corpus_id: str = Field(min_length=1)
    configuration: PreparationConfig


class PreparedIndexChunkingResponse(BaseModel):
    """Describe the chunking stage and its artifact when available.

    Attributes:
        status: Derived lifecycle state of the chunking stage.
        chunk_set_id: Exact reusable chunk artifact after stage success.
        chunk_count: Number of chunks in the ready artifact.
        reused: Whether this request reused an existing compatible artifact.
        duration_ms: Time spent resolving the chunking stage.
    """

    status: PreparedIndexStageStatus
    chunk_set_id: str | None = None
    chunk_count: int | None = Field(default=None, ge=0)
    reused: bool | None = None
    duration_ms: int | None = Field(default=None, ge=0)


class PreparedIndexEmbeddingResponse(BaseModel):
    """Describe the embedding stage and exact vector artifact when available.

    Attributes:
        status: Derived lifecycle state of the embedding stage.
        vector_index_id: Exact reusable vector artifact after stage success.
        vector_count: Number of vectors in the ready Chroma collection.
        dimensions: Width of every vector in the ready index.
        provider: Provider identifier from the immutable request snapshot.
        model: Model identifier from the immutable request snapshot.
        distance_metric: Distance semantics configured for the index.
        reused: Whether this request reused an existing compatible artifact.
        duration_ms: Time spent resolving the embedding stage.
    """

    status: PreparedIndexStageStatus
    vector_index_id: str | None = None
    vector_count: int | None = Field(default=None, ge=0)
    dimensions: int | None = Field(default=None, gt=0)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    distance_metric: Literal["cosine", "dot_product", "euclidean"]
    reused: bool | None = None
    duration_ms: int | None = Field(default=None, ge=0)


class PreparedIndexErrorResponse(BaseModel):
    """Expose a safe terminal preparation failure to polling clients.

    Attributes:
        code: Stable machine-readable failure category.
        message: Safe user-readable failure explanation.
        stage: Preparation stage that failed.
        details: Additional safe structured context.
    """

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    stage: Literal["chunking", "embedding"] | None = None
    details: dict[str, object] = Field(default_factory=dict)


class PreparedIndexResponse(BaseModel):
    """Represent one named prepared index at any lifecycle state.

    Attributes:
        id: Stable identifier used by later experiment selection.
        name: User-facing label; it is not the compatibility identity.
        corpus_id: Immutable corpus used by this preparation request.
        configuration: Resolved immutable chunking and embedding snapshot.
        status: Overall durable lifecycle state.
        current_stage: Currently executing stage, or null while inactive.
        created_at: UTC enqueue timestamp.
        started_at: UTC worker-claim timestamp when available.
        completed_at: UTC terminal timestamp when available.
        duration_ms: Total preparation duration for a terminal request.
        chunking: Chunking stage state and reusable artifact summary.
        embedding: Embedding stage state and reusable artifact summary.
        error: Safe terminal failure details for failed requests.
    """

    id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=100)
    corpus_id: str = Field(min_length=1)
    configuration: PreparationConfig
    status: PreparedIndexStatus
    current_stage: Literal["chunking", "embedding"] | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    chunking: PreparedIndexChunkingResponse
    embedding: PreparedIndexEmbeddingResponse
    error: PreparedIndexErrorResponse | None = None


@router.post(
    "",
    response_model=PreparedIndexResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_index(payload: PreparedIndexCreateRequest) -> PreparedIndexResponse:
    """Validate and enqueue one named index without waiting for model work.

    Args:
        payload: Name, corpus, and complete preparation configuration.

    Returns:
        Persisted pending prepared index suitable for polling.

    Raises:
        HTTPException: If validation, catalog loading, or persistence fails.
    """
    normalized_name = payload.name.strip()
    logger.info(
        "prepared_index_requested corpus_id=%s chunking_strategy=%s "
        "embedding_provider=%s embedding_model=%s",
        payload.corpus_id,
        payload.configuration.chunking.strategy.value,
        payload.configuration.embedding.provider,
        payload.configuration.embedding.model,
    )

    # Whitespace-only labels pass string length checks but are not meaningful names.
    if not normalized_name:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_index_name",
                "message": "Index name must not be blank.",
            },
        )

    try:
        # Resolve backend-owned compatibility before saving an immutable request.
        options = _load_pipeline_options()
        validate_preparation_config(payload.configuration, options)
        prepared_index = create_pending_prepared_index(
            normalized_name,
            payload.corpus_id,
            payload.configuration,
        )
    except PreparedIndexCorpusNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "corpus_not_found",
                "message": "The selected corpus does not exist.",
            },
        ) from error
    except InvalidPipelineConfigurationError as error:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_preparation_configuration",
                "message": error.message,
                "field": error.field,
            },
        ) from error
    except (OSError, ValidationError) as error:
        logger.exception(
            "prepared_index_rejected corpus_id=%s "
            "error_code=pipeline_options_unavailable",
            payload.corpus_id,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "pipeline_options_unavailable",
                "message": "The preparation options could not be loaded.",
            },
        ) from error
    except sqlite3.Error as error:
        logger.exception(
            "prepared_index_rejected corpus_id=%s error_code=persistence_error",
            payload.corpus_id,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "persistence_error",
                "message": "The prepared index request could not be saved.",
            },
        ) from error

    logger.info(
        "prepared_index_enqueued prepared_index_id=%s corpus_id=%s",
        prepared_index["id"],
        payload.corpus_id,
    )
    return PreparedIndexResponse.model_validate(_add_stage_statuses(prepared_index))


@router.get("", response_model=list[PreparedIndexResponse])
async def read_indexes(
    lifecycle_status: Annotated[
        PreparedIndexStatus | None,
        Query(alias="status"),
    ] = None,
) -> list[PreparedIndexResponse]:
    """List named prepared indexes newest first with an optional status filter.

    Args:
        lifecycle_status: Optional lifecycle status supplied as ``status``.

    Returns:
        Ordered prepared-index summaries including failed and active requests.

    Raises:
        HTTPException: If persisted indexes cannot be read.
    """
    try:
        # This local SQLite query does not load chunk text or vector coordinates.
        prepared_indexes = list_prepared_indexes(lifecycle_status)
    except sqlite3.Error as error:
        logger.exception("prepared_index_list_failed error_code=persistence_error")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "persistence_error",
                "message": "Prepared indexes could not be loaded.",
            },
        ) from error

    # Derive presentation-neutral stage states for every persisted lifecycle row.
    return [
        PreparedIndexResponse.model_validate(_add_stage_statuses(prepared_index))
        for prepared_index in prepared_indexes
    ]


@router.get("/{prepared_index_id}", response_model=PreparedIndexResponse)
async def read_index(prepared_index_id: str) -> PreparedIndexResponse:
    """Return the latest persisted state of one named prepared index.

    Args:
        prepared_index_id: Stable identifier returned by ``POST /indexes``.

    Returns:
        Current lifecycle state with configuration and artifact summaries.

    Raises:
        HTTPException: If the identifier is unknown or persistence fails.
    """
    try:
        # Polling performs one local relational read with no model or Chroma call.
        prepared_index = get_prepared_index(prepared_index_id)
    except PreparedIndexNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "prepared_index_not_found",
                "message": "The selected prepared index does not exist.",
            },
        ) from error
    except sqlite3.Error as error:
        logger.exception(
            "prepared_index_read_failed prepared_index_id=%s",
            prepared_index_id,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "persistence_error",
                "message": "The prepared index could not be loaded.",
            },
        ) from error

    return PreparedIndexResponse.model_validate(_add_stage_statuses(prepared_index))


def _add_stage_statuses(prepared_index: dict[str, Any]) -> dict[str, Any]:
    """Add derived chunking and embedding states to a materialized row.

    Args:
        prepared_index: Repository dictionary without derived stage statuses.

    Returns:
        Shallow copied response dictionary containing both derived statuses.
    """
    response = {
        **prepared_index,
        "chunking": dict(prepared_index["chunking"]),
        "embedding": dict(prepared_index["embedding"]),
    }
    response["chunking"]["status"] = _chunking_stage_status(prepared_index)
    response["embedding"]["status"] = _embedding_stage_status(prepared_index)
    embedding_configuration = prepared_index["configuration"]["embedding"]
    response["embedding"].update(
        {
            "provider": embedding_configuration["provider"],
            "model": embedding_configuration["model"],
            "distance_metric": embedding_configuration["distance_metric"],
        }
    )
    return response


def _chunking_stage_status(
    prepared_index: dict[str, Any],
) -> PreparedIndexStageStatus:
    """Derive the chunking-stage lifecycle from durable preparation fields.

    Args:
        prepared_index: Materialized prepared-index lifecycle record.

    Returns:
        Pending, running, completed, or failed chunking state.
    """
    # An attached ready chunk artifact proves that chunking completed.
    if prepared_index["chunking"]["chunk_set_id"] is not None:
        return "completed"

    # A claimed request begins in chunking until that artifact is attached.
    if (
        prepared_index["status"] == "running"
        and prepared_index["current_stage"] == "chunking"
    ):
        return "running"

    # Terminal failure without an attached artifact occurred during chunking.
    if prepared_index["status"] == "failed":
        return "failed"

    return "pending"


def _embedding_stage_status(
    prepared_index: dict[str, Any],
) -> PreparedIndexStageStatus:
    """Derive the embedding-stage lifecycle from durable preparation fields.

    Args:
        prepared_index: Materialized prepared-index lifecycle record.

    Returns:
        Pending, running, completed, or failed embedding state.
    """
    # An attached ready vector artifact proves that embedding completed.
    if prepared_index["embedding"]["vector_index_id"] is not None:
        return "completed"

    # The worker explicitly advances to embedding after chunking persistence.
    if (
        prepared_index["status"] == "running"
        and prepared_index["current_stage"] == "embedding"
    ):
        return "running"

    # A failed request with a chunk artifact failed in embedding.
    if (
        prepared_index["status"] == "failed"
        and prepared_index["chunking"]["chunk_set_id"] is not None
    ):
        return "failed"

    return "pending"
