"""HTTP route for executing immutable single-question pipeline runs."""

import sqlite3
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from backend.api.routers.pipeline_options import _load_pipeline_options
from backend.db.repositories.runs import CorpusNotFoundError
from backend.pipeline.compatibility import (
    InvalidPipelineConfigurationError,
    validate_pipeline_config,
)
from backend.pipeline.configs import PipelineConfig
from backend.pipeline.execution import (
    PipelineExecutor,
    PipelineRunExecutionError,
    get_pipeline_executor,
)

router = APIRouter()


class RunCreateRequest(BaseModel):
    """Represent the user input required to execute one pipeline run.

    Attributes:
        corpus_id: Stable identifier of the selected immutable corpus.
        question: Ad hoc question retained for later retrieval and generation.
        configuration: Complete typed pipeline configuration for the run.
    """

    corpus_id: str = Field(min_length=1)
    question: str
    configuration: PipelineConfig


class RunChunkingResponse(BaseModel):
    """Describe the reusable ready chunk artifact selected by a run.

    Attributes:
        chunk_set_id: Stable identifier of the persisted chunk artifact.
        status: Ready lifecycle state required for completed runs.
        chunk_count: Number of ordered chunks in the artifact.
        reused: Whether the executor reused an existing ready artifact.
    """

    chunk_set_id: str
    status: Literal["ready"]
    chunk_count: int = Field(ge=0)
    reused: bool


class RunResponse(BaseModel):
    """Represent one completed immutable run through the chunking stage.

    Attributes:
        id: Stable application-generated run identifier.
        corpus_id: Stable identifier of the selected immutable corpus.
        question: Normalized question saved for the run.
        configuration: Resolved immutable configuration snapshot.
        status: Completed lifecycle state for the currently implemented stages.
        created_at: UTC timestamp when the pending run was persisted.
        started_at: UTC timestamp when pipeline execution began.
        completed_at: UTC timestamp when chunking completed.
        duration_ms: Total execution duration through chunking.
        chunking: Compact ready chunk-set summary without chunk bodies.
    """

    id: str
    corpus_id: str
    question: str
    configuration: PipelineConfig
    status: Literal["completed"]
    created_at: str
    started_at: str
    completed_at: str
    duration_ms: int = Field(ge=0)
    chunking: RunChunkingResponse


@router.post(
    "/runs",
    response_model=RunResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_pipeline_run(
    payload: RunCreateRequest,
    executor: Annotated[PipelineExecutor, Depends(get_pipeline_executor)],
) -> RunResponse:
    """Validate, persist, and execute one run through chunking.

    Args:
        payload: Selected corpus, question, and complete pipeline configuration.
        executor: Injected coordinator for run lifecycle and pipeline stages.

    Returns:
        Completed run linked to one ready reusable chunk set.

    Raises:
        HTTPException: If validation, chunking, or persistence fails.
    """

    normalized_question = payload.question.strip()

    print("================= /RUNS ================= ")
    print(payload)
    print(normalized_question)

    # Reject blank questions before creating an immutable run record.
    if not normalized_question:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_question",
                "message": "Question must not be blank.",
            },
        )

    try:
        # Validate semantic compatibility before any run lifecycle row is created.
        print("=====================  LOADING PIPELINE OPTIONS... ==============================")
        options = _load_pipeline_options()
        print("=====================  VALIDATING PIPELINE OPTIONS... ==============================")
        validate_pipeline_config(payload.configuration, options)

        print(options)

        print("===================== RUNNING THE EXECUTOR IN THREAD POOL ==============================")
        # Tokenization and SQLite are synchronous, so execute them off the event loop.
        persisted_run = await run_in_threadpool(
            executor.execute,
            payload.corpus_id,
            normalized_question,
            payload.configuration,
        )
    except CorpusNotFoundError as error:
        # Unknown corpora fail before pending-run creation and therefore have no run ID.
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
    except PipelineRunExecutionError as error:
        # State conflicts are repairable inputs; all other execution failures are server-side.
        failure_status = (
            409
            if error.code in {"empty_corpus", "missing_parse_artifact"}
            else 404
            if error.code == "corpus_not_found"
            else 500
        )
        detail = {
            "code": error.code,
            "message": error.message,
            "run_id": error.run_id,
            **error.details,
        }
        raise HTTPException(status_code=failure_status, detail=detail) from error
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
        print(error)

        # Hide database internals behind a stable persistence error response.
        raise HTTPException(
            status_code=500,
            detail={
                "code": "persistence_error",
                "message": "The pipeline run could not be saved.",
            },
        ) from error

    return RunResponse.model_validate(persisted_run)
