"""Validation and document-label resolution for imported evaluation datasets."""

import hashlib
import json
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from backend.db.repositories.evaluation_datasets import (
    create_evaluation_dataset,
    get_corpus_documents_for_dataset,
)


class InvalidDatasetFileError(ValueError):
    """Expose a safe validation failure for imported dataset content.

    Attributes:
        code: Machine-readable failure category used by the HTTP adapter.
        message: Safe user-readable explanation of the invalid input.
        details: Optional structured validation context.
    """

    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize one dataset validation failure.

        Args:
            code: Stable machine-readable error category.
            message: Safe explanation suitable for an API response.
            details: Optional field-level validation information.

        Returns:
            None.
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class DatasetExampleInput(BaseModel):
    """Represent one validated example from an imported JSON document.

    Attributes:
        question: User query that retrieval and generation will process.
        reference_answer: Optional expected answer used by compatible metrics.
        relevant_documents: Source filenames expected to contain relevant content.
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    reference_answer: str | None = None
    relevant_documents: list[str] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Strip and reject a question that contains only whitespace.

        Args:
            value: Question read from the imported example.

        Returns:
            Normalized non-blank question text.

        Raises:
            ValueError: If the supplied question is blank.
        """
        # Persist normalized questions so semantically empty input cannot enter SQLite.
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("Question must not be blank.")
        return normalized_value

    @field_validator("reference_answer")
    @classmethod
    def normalize_reference_answer(cls, value: str | None) -> str | None:
        """Normalize an optional reference answer without making it mandatory.

        Args:
            value: Optional answer read from the imported example.

        Returns:
            Stripped answer text, or ``None`` when it is absent or blank.
        """
        # Treat blank optional answers as absent so metric eligibility stays explicit.
        if value is None:
            return None
        normalized_value = value.strip()
        return normalized_value or None


class DatasetFileInput(BaseModel):
    """Represent the complete validated structure of an imported JSON file.

    Attributes:
        examples: Ordered, non-empty examples retained exactly, including duplicates.
    """

    model_config = ConfigDict(extra="forbid")

    examples: list[DatasetExampleInput] = Field(min_length=1)


def import_evaluation_dataset(
    name: str,
    corpus_id: str,
    source_filename: str,
    source_body: bytes,
) -> dict[str, Any]:
    """Validate, resolve, and atomically persist an uploaded evaluation dataset.

    Args:
        name: Normalized user-facing dataset name.
        corpus_id: Stable corpus selected for relevance-label resolution.
        source_filename: Original JSON upload filename.
        source_body: Exact bounded upload bytes.

    Returns:
        Fully materialized persisted dataset with non-fatal import warnings.

    Raises:
        InvalidDatasetFileError: If UTF-8, JSON, or dataset validation fails.
        EvaluationDatasetCorpusNotFoundError: If the selected corpus is unknown.
        sqlite3.Error: If persistence fails.
    """
    try:
        # Evaluation JSON is a textual contract and must decode deterministically.
        decoded_body = source_body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise InvalidDatasetFileError(
            "invalid_dataset_encoding",
            "The evaluation dataset must be UTF-8 encoded JSON.",
        ) from error

    try:
        # Parse JSON separately so syntax errors remain distinguishable from schema errors.
        raw_dataset = json.loads(decoded_body)
    except json.JSONDecodeError as error:
        raise InvalidDatasetFileError(
            "invalid_dataset_json",
            "The uploaded evaluation dataset is not valid JSON.",
            {"line": error.lineno, "column": error.colno},
        ) from error

    try:
        # Pydantic enforces the public file contract before any rows are written.
        dataset = DatasetFileInput.model_validate(raw_dataset)
    except ValidationError as error:
        # Round-trip Pydantic's JSON form so exception objects in contexts stay serializable.
        validation_errors = json.loads(error.json(include_url=False))
        raise InvalidDatasetFileError(
            "invalid_dataset_schema",
            "The uploaded JSON does not match the evaluation dataset schema.",
            {"validation_errors": validation_errors},
        ) from error

    corpus_documents = get_corpus_documents_for_dataset(corpus_id)
    resolved_examples, warnings = resolve_dataset_examples(dataset, corpus_documents)
    source_sha256 = hashlib.sha256(source_body).hexdigest()
    return create_evaluation_dataset(
        name,
        corpus_id,
        source_filename,
        source_sha256,
        resolved_examples,
        warnings,
    )


def resolve_dataset_examples(
    dataset: DatasetFileInput,
    corpus_documents: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve imported document filenames to unambiguous corpus document IDs.

    Args:
        dataset: Validated dataset JSON whose example order must be retained.
        corpus_documents: Stable IDs and original filenames from the selected corpus.

    Returns:
        Persistable examples and non-fatal warnings for skipped document labels.
    """
    documents_by_filename: dict[str, list[dict[str, str]]] = defaultdict(list)

    # Preserve every same-name document so ambiguity is detected instead of guessed.
    for document in corpus_documents:
        documents_by_filename[document["original_filename"]].append(document)

    resolved_examples: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    # Resolve each example independently so a bad label cannot remove its question.
    for ordinal, example in enumerate(dataset.examples):
        resolved_document_ids: list[str] = []

        # Convert each external filename label into a stable internal document ID.
        for supplied_name in example.relevant_documents:
            normalized_name = supplied_name.strip()
            matching_documents = documents_by_filename.get(normalized_name, [])

            # Empty and unknown labels are skipped with enough context for the UI.
            if not normalized_name or not matching_documents:
                warnings.append(
                    _build_document_warning(
                        ordinal,
                        supplied_name,
                        "document_not_found",
                        "No document with this filename exists in the selected corpus.",
                    )
                )
                continue

            # Multiple exact matches are unsafe because filenames are not identifiers.
            if len(matching_documents) > 1:
                warnings.append(
                    _build_document_warning(
                        ordinal,
                        supplied_name,
                        "ambiguous_document_name",
                        "Multiple corpus documents have this filename.",
                    )
                )
                continue

            document_id = matching_documents[0]["id"]

            # Repeated names in one question represent one relevance relationship.
            if document_id not in resolved_document_ids:
                resolved_document_ids.append(document_id)

        resolved_examples.append(
            {
                "ordinal": ordinal,
                "question": example.question,
                "reference_answer": example.reference_answer,
                "relevant_document_ids": resolved_document_ids,
            }
        )

    return resolved_examples, warnings


def _build_document_warning(
    example_ordinal: int,
    document_name: str,
    code: str,
    message: str,
) -> dict[str, Any]:
    """Create one stable warning for a skipped relevance label.

    Args:
        example_ordinal: Zero-based position of the affected imported example.
        document_name: Exact external label supplied by the dataset.
        code: Machine-readable reason the label was skipped.
        message: Safe explanation suitable for display.

    Returns:
        JSON-compatible warning persisted with the dataset.
    """
    return {
        "example_ordinal": example_ordinal,
        "document_name": document_name,
        "code": code,
        "message": message,
    }
