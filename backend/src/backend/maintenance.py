"""Destructive local-maintenance operations used during backend testing."""

from pathlib import Path

from backend.db.connection import connect
from backend.embedding.vector_store import get_vector_store

# Delete dependent records before the immutable corpus and document roots.
DATABASE_DELETE_ORDER = (
    "generation_context_chunk",
    "generation_result",
    "retrieved_chunk",
    "retrieval_result",
    "pipeline_run",
    "evaluation_example_relevant_document",
    "evaluation_example",
    "evaluation_dataset",
    "prepared_index",
    "vector_index",
    "chunk",
    "chunk_set",
    "parsed_block",
    "parsed_page",
    "document_parse",
    "document",
    "corpus",
)


def clear_vector_store_data() -> int:
    """Delete every application-owned vector collection from local Chroma.

    Args:
        None.

    Returns:
        Number of managed vector collections deleted.

    Raises:
        VectorStoreError: If Chroma cannot complete the reset.
    """
    # Delegate ownership filtering to the adapter that defines collection naming.
    return get_vector_store().delete_managed_collections()


def clear_database_data() -> int:
    """Delete every persisted application record in one transaction.

    Args:
        None.

    Returns:
        The total number of rows deleted across all application tables.
    """
    deleted_rows = 0

    # Use one transaction so a database failure cannot leave partially cleared tables.
    with connect() as connection:
        # Follow the explicit foreign-key-safe order defined for the current schema.
        for table_name in DATABASE_DELETE_ORDER:
            cursor = connection.execute(f'DELETE FROM "{table_name}"')
            deleted_rows += max(cursor.rowcount, 0)

    return deleted_rows


def _delete_upload_entry(entry: Path) -> int:
    """Delete one upload entry without following symbolic links.

    Args:
        entry: File, directory, or symbolic link located under the uploads directory.

    Returns:
        The number of files or symbolic links removed from the entry tree.
    """
    # Unlink files and symbolic links directly so links to directories are not traversed.
    if entry.is_symlink() or entry.is_file():
        entry.unlink()
        return 1

    # Recursively empty real directories before removing the directory itself.
    if entry.is_dir():
        deleted_files = 0

        # Remove every child while keeping file deletion counts for the API response.
        for child in entry.iterdir():
            deleted_files += _delete_upload_entry(child)

        entry.rmdir()
        return deleted_files

    # Remove unusual filesystem entries using unlink without attempting traversal.
    entry.unlink()
    return 1


def clear_uploads_directory(uploads_directory: Path) -> int:
    """Remove all upload contents while preserving the configured root directory.

    Args:
        uploads_directory: Exact backend directory containing stored source files.

    Returns:
        The number of uploaded files or symbolic links removed.

    Raises:
        OSError: If the root is unsafe or any filesystem operation fails.
    """
    # Reject a symbolic-link root so cleanup cannot escape the configured location.
    if uploads_directory.is_symlink():
        raise OSError("The uploads directory must not be a symbolic link.")

    # Create a missing directory so the backend remains ready for the next upload.
    if not uploads_directory.exists():
        uploads_directory.mkdir(parents=True)
        return 0

    # Reject a non-directory root rather than deleting an unexpected filesystem entry.
    if not uploads_directory.is_dir():
        raise OSError("The configured uploads path is not a directory.")

    deleted_files = 0

    # Delete each child but preserve the root directory used by the upload route.
    for entry in uploads_directory.iterdir():
        deleted_files += _delete_upload_entry(entry)

    return deleted_files
