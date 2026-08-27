"""Destructive HTTP utilities intended for local backend testing."""

import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException, Response

from backend.api.routers.uploads import UPLOADS_DIRECTORY
from backend.embedding.models import VectorStoreError
from backend.maintenance import (
    clear_database_data,
    clear_uploads_directory,
    clear_vector_store_data,
)

router = APIRouter()


@router.get("/testing/reset")
def reset_testing_data(response: Response) -> dict[str, Any]:
    """Clear all database records and locally stored uploads.

    Args:
        response: FastAPI response used to prevent caching of this destructive request.

    Returns:
        A confirmation message with deleted database-row and upload-file counts.

    Raises:
        HTTPException: If database deletion or upload cleanup fails.
    """
    # Prevent clients and intermediaries from caching or replaying a reset response.
    response.headers["Cache-Control"] = "no-store"

    try:
        # Remove external collections before deleting their relational provenance.
        deleted_vector_collections = clear_vector_store_data()
    except VectorStoreError as error:
        # Leave SQLite references intact when their external collections remain uncertain.
        raise HTTPException(
            status_code=500,
            detail={
                "code": "vector_store_reset_failed",
                "message": "The testing vector-store data could not be cleared.",
            },
            headers={"Cache-Control": "no-store"},
        ) from error

    try:
        # Clear relational data after its external vector collections are gone.
        deleted_database_rows = clear_database_data()
    except sqlite3.Error as error:
        # Return a stable error code without exposing SQLite details.
        raise HTTPException(
            status_code=500,
            detail={
                "code": "database_reset_failed",
                "message": "The testing database data could not be cleared.",
                "vector_store_cleared": True,
            },
            headers={"Cache-Control": "no-store"},
        ) from error

    try:
        # Clear stored sources only after the database transaction has committed.
        deleted_upload_files = clear_uploads_directory(UPLOADS_DIRECTORY)
    except OSError as error:
        # Explain the partial outcome so another reset can clean remaining files.
        raise HTTPException(
            status_code=500,
            detail={
                "code": "upload_cleanup_failed",
                "message": (
                    "Database data was cleared, but uploaded files could not be "
                    "fully removed."
                ),
                "database_cleared": True,
            },
            headers={"Cache-Control": "no-store"},
        ) from error

    return {
        "message": "Testing data cleared successfully.",
        "deleted_database_rows": deleted_database_rows,
        "deleted_vector_collections": deleted_vector_collections,
        "deleted_upload_files": deleted_upload_files,
    }
