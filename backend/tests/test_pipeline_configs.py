"""Tests for typed chunking and embedding pipeline configuration."""

import pytest
from pydantic import ValidationError

from backend.pipeline.configs import (
    ChunkingConfig,
    ChunkingStrategy,
    DistanceMetric,
    EmbeddingConfig,
    EvaluationConfig,
)


def test_chunking_config_uses_mvp_defaults() -> None:
    """Verify the default configuration represents the documented MVP baseline.

    Parameters:
        None.
    Returns:
        None. Assertions verify the resolved default configuration values.
    """
    # Build the configuration without overrides to resolve the backend defaults.
    configuration = ChunkingConfig()

    # Keep the code defaults aligned with the published MVP parameters.
    assert configuration.strategy is ChunkingStrategy.RECURSIVE
    assert configuration.chunk_size_tokens == 800
    assert configuration.chunk_overlap_tokens == 100


def test_chunking_config_rejects_invalid_overlap() -> None:
    """Verify overlap cannot prevent an advancing chunking window.

    Parameters:
        None.
    Returns:
        None. A validation exception confirms invalid configuration is rejected.
    """
    # Configure an overlap equal to the chunk size, which would produce no stride.
    with pytest.raises(ValidationError, match="smaller than chunk_size_tokens"):
        ChunkingConfig(chunk_size_tokens=100, chunk_overlap_tokens=100)


def test_paragraph_config_requires_zero_overlap() -> None:
    """Verify paragraph chunking does not accept unused overlap settings.

    Parameters:
        None.
    Returns:
        None. Assertions verify valid structure-aware configuration parsing.
    """
    # Resolve the zero-overlap default because this strategy does not slide a window.
    default_configuration = ChunkingConfig(strategy=ChunkingStrategy.PARAGRAPH)
    assert default_configuration.chunk_overlap_tokens == 0

    # Reject an explicit overlap because paragraph/section chunking does not slide.
    with pytest.raises(ValidationError, match="must be 0"):
        ChunkingConfig(
            strategy=ChunkingStrategy.PARAGRAPH,
            chunk_overlap_tokens=10,
        )

    # Accept the explicit non-overlapping configuration for the structure-aware strategy.
    configuration = ChunkingConfig(
        strategy=ChunkingStrategy.PARAGRAPH,
        chunk_overlap_tokens=0,
    )
    assert configuration.chunk_overlap_tokens == 0


def test_chunking_config_rejects_removed_paragraph_section_name() -> None:
    """Verify new configurations cannot request the removed hybrid strategy.

    Parameters:
        None.
    Returns:
        None. A validation exception confirms the breaking rename is enforced.
    """
    # Keep section-aware semantics out of newly persisted immutable run snapshots.
    with pytest.raises(ValidationError, match="paragraph_section"):
        ChunkingConfig(strategy="paragraph_section")


def test_embedding_config_validates_identifiers_and_metric() -> None:
    """Verify embedding configuration preserves a valid model-space identity.

    Parameters:
        None.
    Returns:
        None. Assertions verify valid parsing and invalid identifier rejection.
    """
    # Create a compatible embedding configuration with a non-default metric.
    configuration = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        distance_metric=DistanceMetric.DOT_PRODUCT,
    )
    assert configuration.distance_metric is DistanceMetric.DOT_PRODUCT

    # Reject whitespace-only provider identifiers before a vector index is created.
    with pytest.raises(ValidationError, match="provider must not be blank"):
        EmbeddingConfig(provider="   ", model="text-embedding-3-small")


def test_evaluation_config_allows_multiple_optional_metrics() -> None:
    """Verify evaluation accepts multiple metrics and permits skipping the stage.

    Parameters:
        None.
    Returns:
        None. Assertions verify selected and empty metric lists.
    """
    # Preserve both selected answer metrics in their submitted display order.
    selected_configuration = EvaluationConfig(
        answer_metrics=["groundedness", "answer_relevance"]
    )
    assert selected_configuration.answer_metrics == [
        "groundedness",
        "answer_relevance",
    ]

    # Empty lists explicitly represent a run that skips evaluation.
    skipped_configuration = EvaluationConfig()
    assert skipped_configuration.retrieval_metrics == []
    assert skipped_configuration.answer_metrics == []


def test_evaluation_config_rejects_duplicate_metrics() -> None:
    """Verify one metric cannot be selected twice in the same category.

    Parameters:
        None.
    Returns:
        None. A validation exception confirms duplicate work is rejected.
    """
    # Reject duplicates while retaining lists for stable serialized ordering.
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        EvaluationConfig(answer_metrics=["groundedness", "groundedness"])
