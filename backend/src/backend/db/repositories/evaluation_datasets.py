"""SQLite persistence for immutable evaluation datasets and examples."""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from backend.db.connection import connect


class EvaluationDatasetCorpusNotFoundError(LookupError):
    """Indicate that an import selected a corpus that does not exist."""


class EvaluationDatasetNotFoundError(LookupError):
    """Indicate that a requested evaluation dataset does not exist."""


class EvaluationDatasetInUseError(RuntimeError):
    """Indicate that downstream benchmark history protects a dataset."""


def _utc_timestamp() -> str:
    """Create the UTC timestamp format shared by persisted backend resources.

    Returns:
        An ISO-8601 timestamp ending in ``Z``.
    """
    # Use timezone-aware values so dataset provenance is unambiguous.
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def get_corpus_documents_for_dataset(corpus_id: str) -> list[dict[str, str]]:
    """Load the document identifiers and filenames available for label resolution.

    Args:
        corpus_id: Stable corpus selected for the dataset import.

    Returns:
        Ordered document identities belonging to the corpus.

    Raises:
        EvaluationDatasetCorpusNotFoundError: If the corpus does not exist.
    """
    with connect() as connection:
        # Check the parent independently because a valid empty corpus returns no documents.
        corpus_row = connection.execute(
            "SELECT id FROM corpus WHERE id = ?",
            (corpus_id,),
        ).fetchone()
        if corpus_row is None:
            raise EvaluationDatasetCorpusNotFoundError(corpus_id)

        document_rows = connection.execute(
            """
            SELECT id, original_filename
            FROM document
            WHERE corpus_id = ?
            ORDER BY uploaded_at, id
            """,
            (corpus_id,),
        ).fetchall()

    return [dict(row) for row in document_rows]


def create_evaluation_dataset(
    name: str,
    corpus_id: str,
    source_filename: str,
    source_sha256: str,
    examples: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Persist one dataset, all examples, and resolved labels atomically.

    Args:
        name: Normalized user-facing dataset label.
        corpus_id: Stable corpus used to resolve document labels.
        source_filename: Original JSON upload filename.
        source_sha256: SHA-256 digest of the exact uploaded bytes.
        examples: Ordered validated examples with resolved document IDs.
        warnings: Non-fatal unresolved-label diagnostics indexed by ordinal.

    Returns:
        Fully materialized immutable dataset.

    Raises:
        EvaluationDatasetCorpusNotFoundError: If the corpus disappeared before insert.
    """
    dataset_id = str(uuid4())
    created_at = _utc_timestamp()
    example_ids = {example["ordinal"]: str(uuid4()) for example in examples}

    # Add stable example IDs before persisting warnings for later inspection.
    persisted_warnings = [
        {**warning, "example_id": example_ids[warning["example_ordinal"]]}
        for warning in warnings
    ]

    with connect() as connection:
        # Recheck ownership inside the write transaction to avoid orphaned imports.
        corpus_row = connection.execute(
            "SELECT id FROM corpus WHERE id = ?",
            (corpus_id,),
        ).fetchone()
        if corpus_row is None:
            raise EvaluationDatasetCorpusNotFoundError(corpus_id)

        connection.execute(
            """
            INSERT INTO evaluation_dataset (
                id, name, corpus_id, source_filename, source_sha256,
                import_warnings_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                name,
                corpus_id,
                source_filename,
                source_sha256,
                json.dumps(persisted_warnings, separators=(",", ":")),
                created_at,
            ),
        )

        # Insert examples and their resolved relevance relationships in source order.
        for example in examples:
            example_id = example_ids[example["ordinal"]]
            connection.execute(
                """
                INSERT INTO evaluation_example (
                    id, dataset_id, ordinal, question, reference_answer
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    example_id,
                    dataset_id,
                    example["ordinal"],
                    example["question"],
                    example["reference_answer"],
                ),
            )

            # Each successfully resolved filename becomes a stable document link.
            for document_id in example["relevant_document_ids"]:
                connection.execute(
                    """
                    INSERT INTO evaluation_example_relevant_document (
                        example_id, document_id
                    ) VALUES (?, ?)
                    """,
                    (example_id, document_id),
                )

    return get_evaluation_dataset(dataset_id)


def list_evaluation_datasets(
    corpus_id: str | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    """List dataset summaries newest first with optional corpus and name filters.

    Args:
        corpus_id: Optional exact corpus identifier.
        search: Optional case-insensitive substring of the dataset name.

    Returns:
        Ordered response-ready dataset summaries.
    """
    conditions: list[str] = []
    parameters: list[str] = []

    # Add only selected filters so an omitted query returns the complete inventory.
    if corpus_id is not None:
        conditions.append("evaluation_dataset.corpus_id = ?")
        parameters.append(corpus_id)
    if search is not None and search.strip():
        conditions.append("instr(lower(evaluation_dataset.name), lower(?)) > 0")
        parameters.append(search.strip())

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with connect() as connection:
        rows = connection.execute(
            f"""
            SELECT evaluation_dataset.*, corpus.name AS corpus_name,
                   COUNT(DISTINCT evaluation_example.id) AS example_count,
                   COUNT(evaluation_example_relevant_document.document_id)
                       AS resolved_document_count
            FROM evaluation_dataset
            JOIN corpus ON corpus.id = evaluation_dataset.corpus_id
            LEFT JOIN evaluation_example
                ON evaluation_example.dataset_id = evaluation_dataset.id
            LEFT JOIN evaluation_example_relevant_document
                ON evaluation_example_relevant_document.example_id =
                   evaluation_example.id
            {where_clause}
            GROUP BY evaluation_dataset.id
            ORDER BY evaluation_dataset.created_at DESC, evaluation_dataset.id DESC
            """,
            parameters,
        ).fetchall()

    return [_dataset_summary_from_row(row) for row in rows]


def get_evaluation_dataset(dataset_id: str) -> dict[str, Any]:
    """Load one dataset with its ordered examples and resolved documents.

    Args:
        dataset_id: Stable dataset identifier.

    Returns:
        Response-ready dataset detail.

    Raises:
        EvaluationDatasetNotFoundError: If the dataset does not exist.
    """
    with connect() as connection:
        dataset_row = connection.execute(
            """
            SELECT evaluation_dataset.*, corpus.name AS corpus_name
            FROM evaluation_dataset
            JOIN corpus ON corpus.id = evaluation_dataset.corpus_id
            WHERE evaluation_dataset.id = ?
            """,
            (dataset_id,),
        ).fetchone()
        if dataset_row is None:
            raise EvaluationDatasetNotFoundError(dataset_id)

        example_rows = connection.execute(
            """
            SELECT id, ordinal, question, reference_answer
            FROM evaluation_example
            WHERE dataset_id = ?
            ORDER BY ordinal
            """,
            (dataset_id,),
        ).fetchall()
        document_rows = connection.execute(
            """
            SELECT evaluation_example_relevant_document.example_id,
                   document.id, document.original_filename
            FROM evaluation_example_relevant_document
            JOIN document
                ON document.id = evaluation_example_relevant_document.document_id
            JOIN evaluation_example
                ON evaluation_example.id =
                   evaluation_example_relevant_document.example_id
            WHERE evaluation_example.dataset_id = ?
            ORDER BY evaluation_example.ordinal, document.original_filename, document.id
            """,
            (dataset_id,),
        ).fetchall()

    documents_by_example: dict[str, list[dict[str, str]]] = {}

    # Group normalized labels beneath their owning example for the public response.
    for row in document_rows:
        documents_by_example.setdefault(row["example_id"], []).append(
            {"id": row["id"], "filename": row["original_filename"]}
        )

    examples = [
        {
            "id": row["id"],
            "ordinal": row["ordinal"],
            "question": row["question"],
            "reference_answer": row["reference_answer"],
            "relevant_documents": documents_by_example.get(row["id"], []),
        }
        for row in example_rows
    ]
    warnings = json.loads(dataset_row["import_warnings_json"])
    return {
        "id": dataset_row["id"],
        "name": dataset_row["name"],
        "corpus_id": dataset_row["corpus_id"],
        "corpus_name": dataset_row["corpus_name"],
        "source_filename": dataset_row["source_filename"],
        "source_sha256": dataset_row["source_sha256"],
        "example_count": len(examples),
        "resolved_document_count": sum(
            len(example["relevant_documents"]) for example in examples
        ),
        "warning_count": len(warnings),
        "created_at": dataset_row["created_at"],
        "warnings": warnings,
        "examples": examples,
    }


def delete_evaluation_dataset(dataset_id: str) -> None:
    """Delete one unreferenced dataset and its cascading child records.

    Args:
        dataset_id: Stable dataset identifier to remove.

    Returns:
        None.

    Raises:
        EvaluationDatasetNotFoundError: If the dataset does not exist.
        EvaluationDatasetInUseError: If downstream history references the dataset.
    """
    try:
        with connect() as connection:
            cursor = connection.execute(
                "DELETE FROM evaluation_dataset WHERE id = ?",
                (dataset_id,),
            )
            if cursor.rowcount == 0:
                raise EvaluationDatasetNotFoundError(dataset_id)
    except sqlite3.IntegrityError as error:
        # Convert future benchmark foreign-key protection into a domain-level conflict.
        raise EvaluationDatasetInUseError(dataset_id) from error


def _dataset_summary_from_row(row: sqlite3.Row) -> dict[str, Any]:
    """Convert an aggregate SQLite row into a dataset-list response.

    Args:
        row: Joined dataset, corpus, example, and relevance aggregate row.

    Returns:
        JSON-compatible dataset summary with derived warning count.
    """
    warnings = json.loads(row["import_warnings_json"])
    return {
        "id": row["id"],
        "name": row["name"],
        "corpus_id": row["corpus_id"],
        "corpus_name": row["corpus_name"],
        "source_filename": row["source_filename"],
        "source_sha256": row["source_sha256"],
        "example_count": row["example_count"],
        "resolved_document_count": row["resolved_document_count"],
        "warning_count": len(warnings),
        "created_at": row["created_at"],
    }
