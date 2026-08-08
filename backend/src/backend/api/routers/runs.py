"""HTTP route for persisting immutable single-question pipeline runs."""

import sqlite3

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, ValidationError

from backend.api.routers.pipeline_options import _load_pipeline_options
from backend.db.repositories.runs import CorpusNotFoundError, create_run
from backend.pipeline.compatibility import (
    InvalidPipelineConfigurationError,
    validate_pipeline_config,
)
from backend.pipeline.configs import PipelineConfig

router = APIRouter()


class RunCreateRequest(BaseModel):
    """Represent the user input required to persist one pipeline run.

    Attributes:
        corpus_id: Stable identifier of the selected immutable corpus.
        question: Ad hoc question to answer when execution is implemented.
        configuration: Complete typed pipeline configuration for the run.
    """

    corpus_id: str = Field(min_length=1)
    question: str
    configuration: PipelineConfig


class RunResponse(BaseModel):
    """Represent one persisted immutable single-question run.

    Attributes:
        id: Stable application-generated run identifier.
        corpus_id: Stable identifier of the selected immutable corpus.
        question: Normalized question saved for the run.
        configuration: Resolved configuration snapshot saved with the run.
        created_at: UTC timestamp when the run was persisted.
    """

    id: str
    corpus_id: str
    question: str
    configuration: PipelineConfig
    created_at: str


@router.post(
    "/runs",
    response_model=RunResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_pipeline_run(payload: RunCreateRequest) -> RunResponse:
    """Validate and persist one single-question run without executing it.

    Args:
        payload: Selected corpus, question, and complete pipeline configuration.

    Returns:
        The newly persisted immutable run.

    Raises:
        HTTPException: If the question, corpus, configuration, or persistence fails.
    """
    normalized_question = payload.question.strip()

    # Reject blank questions after normalization with a stable API error code.
    if not normalized_question:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_question",
                "message": "Question must not be blank.",
            },
        )

    try:
        # Validate semantic compatibility against the same catalog served to the UI.
        options = _load_pipeline_options()
        validate_pipeline_config(payload.configuration, options)
        persisted_run = create_run(
            payload.corpus_id,
            normalized_question,
            payload.configuration,
        )
    except CorpusNotFoundError as error:
        # Distinguish an unknown corpus from a general database failure.
        raise HTTPException(
            status_code=404,
            detail={
                "code": "corpus_not_found",
                "message": "The selected corpus does not exist.",
            },
        ) from error
    except InvalidPipelineConfigurationError as error:
        # Return the incompatible field without exposing internal adapter details.
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_pipeline_configuration",
                "message": error.message,
                "field": error.field,
            },
        ) from error
    except (OSError, ValidationError) as error:
        # Match the options endpoint when its version-controlled catalog is unavailable.
        raise HTTPException(
            status_code=500,
            detail={
                "code": "pipeline_options_unavailable",
                "message": "The pipeline configuration options could not be loaded.",
            },
        ) from error
    except sqlite3.Error as error:
        # Hide database internals behind a stable persistence error response.
        raise HTTPException(
            status_code=500,
            detail={
                "code": "persistence_error",
                "message": "The pipeline run could not be saved.",
            },
        ) from error

    return RunResponse.model_validate(persisted_run)
