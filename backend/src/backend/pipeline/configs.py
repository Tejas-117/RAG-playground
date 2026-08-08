"""Typed, serializable configuration for reusable ingestion pipeline stages."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class ChunkingStrategy(str, Enum):
    """Chunking strategies supported by the first RAG Playground MVP."""

    RECURSIVE = "recursive"
    FIXED_SIZE = "fixed_size"
    PARAGRAPH_SECTION = "paragraph_section"


class DistanceMetric(str, Enum):
    """Vector distance or similarity metrics supported by the MVP."""

    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


class ChunkingConfig(BaseModel):
    """Configure how normalized document text is split into chunks.

    Attributes:
        strategy: The chunking implementation selected for the run.
        chunk_size_tokens: Maximum target size of each chunk in tokens.
        chunk_overlap_tokens: Shared tokens between adjacent chunks where supported.
    """

    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size_tokens: int = Field(default=800, gt=0)
    chunk_overlap_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        """Validate chunk-overlap rules for the selected chunking strategy.

        Args:
            None. Validation reads this model's already-parsed fields.

        Returns:
            The validated configuration instance.

        Raises:
            ValueError: If overlap is not meaningful or cannot form a stride.
        """
        # Resolve a strategy-specific default so serialized run configurations are explicit.
        if self.chunk_overlap_tokens is None:
            self.chunk_overlap_tokens = (
                0 if self.strategy is ChunkingStrategy.PARAGRAPH_SECTION else 100
            )

        # Paragraph/section chunking preserves structure and does not use overlap.
        if (
            self.strategy is ChunkingStrategy.PARAGRAPH_SECTION
            and self.chunk_overlap_tokens != 0
        ):
            raise ValueError(
                "chunk_overlap_tokens must be 0 for paragraph_section chunking"
            )

        # Ensure sliding-window strategies always advance through the source text.
        if (
            self.strategy in {ChunkingStrategy.RECURSIVE, ChunkingStrategy.FIXED_SIZE}
            and self.chunk_overlap_tokens >= self.chunk_size_tokens
        ):
            raise ValueError(
                "chunk_overlap_tokens must be smaller than chunk_size_tokens"
            )

        return self


class EmbeddingConfig(BaseModel):
    """Configure the embedding space used to build a compatible vector index.

    Attributes:
        provider: Backend-registered embedding provider identifier.
        model: Provider model identifier used to create vectors.
        distance_metric: Metric used by the vector index for retrieval.
    """

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    distance_metric: DistanceMetric = DistanceMetric.COSINE

    @model_validator(mode="after")
    def validate_identifiers(self) -> "EmbeddingConfig":
        """Reject provider and model identifiers that contain only whitespace.

        Args:
            None. Validation reads this model's already-parsed fields.

        Returns:
            The validated embedding configuration instance.

        Raises:
            ValueError: If a required identifier is blank after trimming.
        """
        # Require meaningful identifiers while preserving the submitted identifier text.
        if not self.provider.strip():
            raise ValueError("provider must not be blank")

        if not self.model.strip():
            raise ValueError("model must not be blank")

        return self


class RetrievalConfig(BaseModel):
    """Configure nearest-neighbor retrieval for one pipeline run.

    Attributes:
        top_k: Maximum number of chunks returned for the question.
    """

    top_k: int = Field(default=10, gt=0)


class GenerationConfig(BaseModel):
    """Configure answer generation for one pipeline run.

    Attributes:
        provider: Backend-registered generation provider identifier.
        model: Provider model identifier used to generate the answer.
        temperature: Sampling temperature sent to the generation provider.
        max_output_tokens: Maximum number of tokens requested for the answer.
    """

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.2, ge=0)
    max_output_tokens: int = Field(default=1000, gt=0)

    @model_validator(mode="after")
    def validate_identifiers(self) -> "GenerationConfig":
        """Reject provider and model identifiers that contain only whitespace.

        Args:
            None. Validation reads this model's already-parsed fields.

        Returns:
            The validated generation configuration instance.

        Raises:
            ValueError: If a required identifier is blank after trimming.
        """
        # Require provider and model names that can be matched to the option catalog.
        if not self.provider.strip():
            raise ValueError("provider must not be blank")

        # Reject blank model identifiers independently for precise validation feedback.
        if not self.model.strip():
            raise ValueError("model must not be blank")

        return self


class EvaluationConfig(BaseModel):
    """Configure the optional metrics calculated for one pipeline run.

    Attributes:
        retrieval_metrics: Retrieval metric identifiers selected for the run.
        answer_metrics: Answer metric identifiers selected for the run.
    """

    retrieval_metrics: list[str] = Field(default_factory=list)
    answer_metrics: list[str] = Field(default_factory=list)

    @field_validator("retrieval_metrics", "answer_metrics")
    @classmethod
    def validate_metric_identifiers(cls, metrics: list[str]) -> list[str]:
        """Reject blank or duplicate metric identifiers.

        Args:
            metrics: Metric identifiers submitted for one evaluation category.

        Returns:
            The unchanged ordered metric identifiers when they are valid.

        Raises:
            ValueError: If a metric is blank or selected more than once.
        """
        # Reject identifiers that cannot be matched to the backend option catalog.
        if any(not metric.strip() for metric in metrics):
            raise ValueError("metric identifiers must not be blank")

        # Keep list ordering stable while preventing duplicate evaluation work.
        if len(metrics) != len(set(metrics)):
            raise ValueError("metric identifiers must not contain duplicates")

        return metrics


class PipelineConfig(BaseModel):
    """Capture the complete effective configuration saved with one run.

    Attributes:
        chunking: Settings used to create or reuse a chunk set.
        embedding: Settings used to create or reuse a compatible vector index.
        retrieval: Settings used to retrieve context for the question.
        generation: Settings used to generate an answer from retrieved context.
        evaluation: Optional metrics selected for the generated run output.
    """

    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
