"""HTTP routes for importing and managing evaluation datasets."""

import logging
import sqlite3
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.db.repositories.evaluation_datasets import (
    EvaluationDatasetCorpusNotFoundError,
    EvaluationDatasetInUseError,
    EvaluationDatasetNotFoundError,
    delete_evaluation_dataset,
    get_evaluation_dataset,
    list_evaluation_datasets,
)
from backend.evaluation.datasets import (
    InvalidDatasetFileError,
    import_evaluation_dataset,
)

# Group immutable dataset-management operations under one public resource path.
router = APIRouter(prefix="/datasets", tags=["datasets"])

# Identify dataset import and management lifecycle events in backend logs.
logger = logging.getLogger(__name__)

# Reuse the document-upload ceiling so all user-provided files have one bounded limit.
MAX_DATASET_SIZE_BYTES = 30 * 1024 * 1024


class RelevantDocumentResponse(BaseModel):
    """Represent one successfully resolved relevance label.

    Attributes:
        id: Stable document identifier used by evaluation.
        filename: User-recognizable original document filename.
    """

    id: str = Field(min_length=1)
    filename: str = Field(min_length=1)


class DatasetImportWarningResponse(BaseModel):
    """Describe one document filename skipped during a successful import.

    Attributes:
        example_id: Stable example affected by the warning.
        example_ordinal: Zero-based source position of that example.
        document_name: External filename label that could not be resolved.
        code: Machine-readable reason the label was skipped.
        message: Safe user-readable explanation.
    """

    example_id: str = Field(min_length=1)
    example_ordinal: int = Field(ge=0)
    document_name: str
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class DatasetExampleResponse(BaseModel):
    """Represent one immutable evaluation example and resolved labels.

    Attributes:
        id: Stable example identifier used by future benchmark runs.
        ordinal: Zero-based order in the imported JSON file.
        question: Query supplied to retrieval and generation.
        reference_answer: Optional expected answer for compatible metrics.
        relevant_documents: Successfully resolved stable document identities.
    """

    id: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    question: str = Field(min_length=1)
    reference_answer: str | None = None
    relevant_documents: list[RelevantDocumentResponse]


class DatasetSummaryResponse(BaseModel):
    """Represent one dataset in collection listings.

    Attributes:
        id: Stable dataset identifier.
        name: User-provided display label.
        corpus_id: Stable corpus used for document-label resolution.
        corpus_name: User-facing corpus label.
        source_filename: Original uploaded JSON filename.
        source_sha256: Digest identifying the exact imported bytes.
        example_count: Number of imported examples, including duplicates.
        resolved_document_count: Number of stored relevance relationships.
        warning_count: Number of skipped filename labels.
        created_at: UTC import timestamp.
    """

    id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=100)
    corpus_id: str = Field(min_length=1)
    corpus_name: str = Field(min_length=1)
    source_filename: str = Field(min_length=1)
    source_sha256: str = Field(min_length=64, max_length=64)
    example_count: int = Field(ge=0)
    resolved_document_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    created_at: str


class DatasetDetailResponse(DatasetSummaryResponse):
    """Represent one dataset with its examples and persisted import warnings.

    Attributes:
        warnings: Non-fatal document-label issues found during import.
        examples: Ordered immutable examples and resolved documents.
    """

    warnings: list[DatasetImportWarningResponse]
    examples: list[DatasetExampleResponse]


async def _read_dataset_upload(file: UploadFile) -> bytes:
    """Read one dataset upload while enforcing the backend byte limit.

    Args:
        file: Multipart JSON upload supplied by the client.

    Returns:
        Exact uploaded bytes when the file is within the limit.

    Raises:
        HTTPException: If the upload exceeds the dataset size limit.
    """
    # Reading one byte beyond the limit distinguishes an exact-limit file from overflow.
    body = await file.read(MAX_DATASET_SIZE_BYTES + 1)
    if len(body) > MAX_DATASET_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "dataset_file_too_large",
                "message": "The evaluation dataset must be 30 MB or smaller.",
                "max_size_bytes": MAX_DATASET_SIZE_BYTES,
            },
        )
    return body


def _validate_dataset_filename(filename: str | None) -> str:
    """Validate that an upload has a plain JSON filename.

    Args:
        filename: Client-provided multipart filename.

    Returns:
        Validated original filename.

    Raises:
        HTTPException: If the filename is unsafe or not a JSON file.
    """
    # Reject path components because only provenance, not client paths, is accepted.
    if not filename or filename in {".", ".."} or Path(filename).name != filename:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_dataset_filename",
                "message": "The dataset upload must have a plain filename.",
            },
        )

    # The v1 import contract intentionally accepts JSON files only.
    if Path(filename).suffix.lower() != ".json":
        raise HTTPException(
            status_code=415,
            detail={
                "code": "unsupported_dataset_file_type",
                "message": "Evaluation datasets must be uploaded as JSON files.",
            },
        )
    return filename


@router.post(
    "", response_model=DatasetDetailResponse, status_code=status.HTTP_201_CREATED
)
async def create_dataset(
    name: Annotated[str, Form(...)],
    corpus_id: Annotated[str, Form(...)],
    file: Annotated[UploadFile, File(...)],
) -> DatasetDetailResponse:
    """Import one immutable evaluation dataset for a selected corpus.

    Args:
        name: Required user-facing dataset label.
        corpus_id: Corpus whose filenames should resolve relevance labels.
        file: UTF-8 JSON dataset upload.

    Returns:
        Persisted dataset detail including all non-fatal import warnings.

    Raises:
        HTTPException: If metadata, content, corpus, or persistence is invalid.
    """
    normalized_name = name.strip()
    if not normalized_name or len(normalized_name) > 100:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_dataset_name",
                "message": "Dataset name must contain 1 to 100 characters.",
            },
        )
    normalized_corpus_id = corpus_id.strip()
    if not normalized_corpus_id:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_corpus_id",
                "message": "A corpus must be selected for the dataset.",
            },
        )

    source_filename = _validate_dataset_filename(file.filename)
    source_body = await _read_dataset_upload(file)
    logger.info(
        "evaluation_dataset_import_requested corpus_id=%s filename=%r size_bytes=%d",
        normalized_corpus_id,
        source_filename,
        len(source_body),
    )

    try:
        # JSON parsing and SQLite work run outside the async event-loop thread.
        dataset = await run_in_threadpool(
            import_evaluation_dataset,
            normalized_name,
            normalized_corpus_id,
            source_filename,
            source_body,
        )
    except InvalidDatasetFileError as error:
        logger.warning(
            "evaluation_dataset_import_rejected corpus_id=%s error_code=%s",
            normalized_corpus_id,
            error.code,
        )
        raise HTTPException(
            status_code=422,
            detail={
                "code": error.code,
                "message": error.message,
                **error.details,
            },
        ) from error
    except EvaluationDatasetCorpusNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "corpus_not_found",
                "message": "The selected corpus does not exist.",
            },
        ) from error
    except sqlite3.Error as error:
        logger.exception(
            "evaluation_dataset_import_failed corpus_id=%s error_code=persistence_error",
            normalized_corpus_id,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "persistence_error",
                "message": "The evaluation dataset could not be saved.",
            },
        ) from error

    logger.info(
        "evaluation_dataset_import_completed dataset_id=%s example_count=%d "
        "warning_count=%d",
        dataset["id"],
        dataset["example_count"],
        dataset["warning_count"],
    )
    return DatasetDetailResponse.model_validate(dataset)


@router.get("", response_model=list[DatasetSummaryResponse])
async def read_datasets(
    corpus_id: Annotated[str | None, Query()] = None,
    search: Annotated[str | None, Query()] = None,
) -> list[DatasetSummaryResponse]:
    """List evaluation datasets with optional corpus and name filtering.

    Args:
        corpus_id: Optional exact corpus identifier.
        search: Optional case-insensitive dataset-name substring.

    Returns:
        Newest-first dataset summaries.

    Raises:
        HTTPException: If persistence cannot be read.
    """
    try:
        datasets = list_evaluation_datasets(corpus_id, search)
    except sqlite3.Error as error:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "persistence_error",
                "message": "Evaluation datasets could not be loaded.",
            },
        ) from error
    return [DatasetSummaryResponse.model_validate(dataset) for dataset in datasets]


@router.get("/{dataset_id}", response_model=DatasetDetailResponse)
async def read_dataset(dataset_id: str) -> DatasetDetailResponse:
    """Return one evaluation dataset with examples and import warnings.

    Args:
        dataset_id: Stable identifier returned by dataset import or listing.

    Returns:
        Complete immutable dataset detail.

    Raises:
        HTTPException: If the dataset is unknown or persistence cannot be read.
    """
    try:
        dataset = get_evaluation_dataset(dataset_id)
    except EvaluationDatasetNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "dataset_not_found",
                "message": "The selected evaluation dataset does not exist.",
            },
        ) from error
    except sqlite3.Error as error:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "persistence_error",
                "message": "The evaluation dataset could not be loaded.",
            },
        ) from error
    return DatasetDetailResponse.model_validate(dataset)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_dataset(dataset_id: str) -> Response:
    """Delete one dataset unless immutable downstream history references it.

    Args:
        dataset_id: Stable dataset identifier to remove.

    Returns:
        Empty 204 response after successful deletion.

    Raises:
        HTTPException: If the dataset is unknown, protected, or cannot be deleted.
    """
    try:
        delete_evaluation_dataset(dataset_id)
    except EvaluationDatasetNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "dataset_not_found",
                "message": "The selected evaluation dataset does not exist.",
            },
        ) from error
    except EvaluationDatasetInUseError as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "dataset_in_use",
                "message": "The dataset is referenced by benchmark history.",
            },
        ) from error
    except sqlite3.Error as error:
        logger.exception(
            "evaluation_dataset_delete_failed dataset_id=%s error_code=persistence_error",
            dataset_id,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "persistence_error",
                "message": "The evaluation dataset could not be deleted.",
            },
        ) from error

    logger.info("evaluation_dataset_deleted dataset_id=%s", dataset_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
