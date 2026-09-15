"""Tests for foreign-key-safe local development reset ordering."""

from backend.maintenance import DATABASE_DELETE_ORDER


def test_benchmark_rows_are_deleted_before_selected_inputs() -> None:
    """Verify benchmark dependencies are cleared before datasets and indexes.

    Returns:
        None. Assertions cover every benchmark parent and selected artifact.
    """
    # Child results must be removed before their example-execution parent.
    assert DATABASE_DELETE_ORDER.index("benchmark_generation_result") < (
        DATABASE_DELETE_ORDER.index("benchmark_example_run")
    )
    assert DATABASE_DELETE_ORDER.index("benchmark_retrieval_result") < (
        DATABASE_DELETE_ORDER.index("benchmark_example_run")
    )

    # The user-visible run protects both immutable selected resources.
    assert DATABASE_DELETE_ORDER.index("benchmark_run") < (
        DATABASE_DELETE_ORDER.index("evaluation_dataset")
    )
    assert DATABASE_DELETE_ORDER.index("benchmark_run") < (
        DATABASE_DELETE_ORDER.index("prepared_index")
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


def test_evaluation_datasets_are_deleted_in_foreign_key_order() -> None:
    """Verify reset removes relevance links and examples before dataset parents.

    Args:
        None.

    Returns:
        None. Assertions verify every evaluation-dataset dependency is safe.
    """
    # Relevance links depend on both their owning example and a source document.
    assert DATABASE_DELETE_ORDER.index(
        "evaluation_example_relevant_document"
    ) < DATABASE_DELETE_ORDER.index("evaluation_example")
    assert DATABASE_DELETE_ORDER.index(
        "evaluation_example_relevant_document"
    ) < DATABASE_DELETE_ORDER.index("document")

    # Examples cascade from datasets, while datasets themselves belong to corpora.
    assert DATABASE_DELETE_ORDER.index("evaluation_example") < (
        DATABASE_DELETE_ORDER.index("evaluation_dataset")
    )
    assert DATABASE_DELETE_ORDER.index("evaluation_dataset") < (
        DATABASE_DELETE_ORDER.index("corpus")
    )
