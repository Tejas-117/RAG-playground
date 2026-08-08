"""SQLite persistence helpers for immutable single-question pipeline runs."""

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from backend.db.connection import connect
from backend.pipeline.configs import PipelineConfig


class CorpusNotFoundError(LookupError):
    """Report that a run references a corpus that does not exist."""


def _utc_timestamp() -> str:
    """Create the UTC timestamp format used by the SQLite schema.

    Args:
        None.

    Returns:
        An ISO-8601 UTC timestamp ending in ``Z``.
    """
    # Use timezone-aware values so saved run creation times are unambiguous.
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def create_run(
    corpus_id: str,
    question: str,
    configuration: PipelineConfig,
) -> dict[str, Any]:
    """Persist one immutable single-question run.

    Args:
        corpus_id: Stable identifier of the immutable corpus selected for the run.
        question: Normalized non-empty question submitted by the user.
        configuration: Typed configuration with all backend defaults resolved.

    Returns:
        A response-ready dictionary containing the newly persisted run.

    Raises:
        CorpusNotFoundError: If the selected corpus does not exist.
        sqlite3.Error: If SQLite cannot read or persist the run.
    """
    run_id = str(uuid4())
    created_at = _utc_timestamp()
    effective_configuration = configuration.model_dump(mode="json")
    effective_config_json = json.dumps(
        effective_configuration,
        sort_keys=True,
        separators=(",", ":"),
    )

    # Verify corpus existence and insert the run within one database transaction.
    with connect() as connection:
        corpus = connection.execute(
            "SELECT id FROM corpus WHERE id = ?",
            (corpus_id,),
        ).fetchone()

        # Return a domain-specific error instead of exposing a foreign-key failure.
        if corpus is None:
            raise CorpusNotFoundError(corpus_id)

        connection.execute(
            """
            INSERT INTO pipeline_run (
                id, corpus_id, question, effective_config_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                run_id,
                corpus_id,
                question,
                effective_config_json,
                created_at,
            ),
        )

    # Return the parsed configuration rather than leaking its SQLite serialization.
    return {
        "id": run_id,
        "corpus_id": corpus_id,
        "question": question,
        "configuration": effective_configuration,
        "created_at": created_at,
    }
