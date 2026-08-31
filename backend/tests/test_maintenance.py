"""Tests for foreign-key-safe local development reset ordering."""

from backend.maintenance import DATABASE_DELETE_ORDER


def test_retrieval_rows_are_deleted_before_their_parent_artifacts() -> None:
    """Verify reset removes ranked results before runs, indexes, and chunks.

    Args:
        None.

    Returns:
        None. Assertions verify every retrieval dependency is ordered safely.
    """
    # Child ranking rows must disappear before their result parent and chunk targets.
    assert DATABASE_DELETE_ORDER.index("retrieved_chunk") < (
        DATABASE_DELETE_ORDER.index("retrieval_result")
    )
    assert DATABASE_DELETE_ORDER.index("retrieved_chunk") < (
        DATABASE_DELETE_ORDER.index("chunk")
    )

    # Result parents must disappear before their owning run and searched index.
    assert DATABASE_DELETE_ORDER.index("retrieval_result") < (
        DATABASE_DELETE_ORDER.index("pipeline_run")
    )
    assert DATABASE_DELETE_ORDER.index("retrieval_result") < (
        DATABASE_DELETE_ORDER.index("vector_index")
    )


def test_generation_rows_are_deleted_before_retrieval_and_runs() -> None:
    """Verify reset removes generated answers before their upstream artifacts.

    Args:
        None.

    Returns:
        None. Assertions verify generation foreign-key dependencies are ordered.
    """
    # Prompt context links depend on both answer and ranked retrieval rows.
    assert DATABASE_DELETE_ORDER.index("generation_context_chunk") < (
        DATABASE_DELETE_ORDER.index("generation_result")
    )
    assert DATABASE_DELETE_ORDER.index("generation_context_chunk") < (
        DATABASE_DELETE_ORDER.index("retrieved_chunk")
    )

    # Generated-answer parents depend on both their run and retrieval result.
    assert DATABASE_DELETE_ORDER.index("generation_result") < (
        DATABASE_DELETE_ORDER.index("retrieval_result")
    )
    assert DATABASE_DELETE_ORDER.index("generation_result") < (
        DATABASE_DELETE_ORDER.index("pipeline_run")
    )


def test_prepared_indexes_are_deleted_before_their_reusable_artifacts() -> None:
    """Verify reset removes prepared-index references before their targets.

    Args:
        None.

    Returns:
        None. Assertions verify prepared-index foreign keys are ordered safely.
    """
    # Prepared indexes may reference both reusable artifact types after a build.
    assert DATABASE_DELETE_ORDER.index("prepared_index") < (
        DATABASE_DELETE_ORDER.index("vector_index")
    )
    assert DATABASE_DELETE_ORDER.index("prepared_index") < (
        DATABASE_DELETE_ORDER.index("chunk_set")
    )

    # Every prepared index belongs to a corpus, including pending and failed builds.
    assert DATABASE_DELETE_ORDER.index("prepared_index") < (
        DATABASE_DELETE_ORDER.index("corpus")
    )
