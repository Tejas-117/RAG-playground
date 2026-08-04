"""HTTP routes for reading corpora and their uploaded documents."""

import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.db.repositories.corpora import list_corpora

router = APIRouter()


@router.get("/corpora/")
async def get_corpora() -> dict[str, list[dict[str, Any]]]:
    """Return all corpora and their persisted document details.

    Args:
        None.

    Returns:
        A response containing corpus records with nested document metadata.

    Raises:
        HTTPException: If the SQLite database cannot be read.
    """
    try:
        # Read the current corpus inventory for frontend display.
        return {"corpora": list_corpora()}
    except sqlite3.Error as error:
        # Return a stable machine-readable error without exposing database internals.
        raise HTTPException(
            status_code=500,
            detail={
                "code": "persistence_error",
                "message": "The uploaded corpus details could not be read.",
            },
        ) from error
