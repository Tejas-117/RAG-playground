"""HTTP route for reading the backend-owned pipeline option catalog."""

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError

router = APIRouter()

# Resolve the version-controlled catalog relative to the installed backend package.
PIPELINE_OPTIONS_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "pipeline_options.json"
)


class Option(BaseModel):
    """Describe one selectable pipeline option.

    Attributes:
        value: Stable identifier submitted in a run configuration.
        label: Human-readable name intended for presentation.
        description: Optional explanation of the option's behavior.
    """

    value: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str | None = None


class ChunkingStrategyOption(Option):
    """Describe one chunker and whether it accepts overlap.

    Attributes:
        supports_overlap: Whether the frontend may expose overlap for this strategy.
    """

    supports_overlap: bool


class IntegerSettingOption(BaseModel):
    """Describe the default and accepted range of an integer setting.

    Attributes:
        default: Backend-recommended starting value.
        minimum: Smallest selectable value.
        maximum: Optional largest selectable value.
    """

    default: int
    minimum: int
    maximum: int | None = None


class FloatSettingOption(BaseModel):
    """Describe the default and accepted range of a floating-point setting.

    Attributes:
        default: Backend-recommended starting value.
        minimum: Smallest selectable value.
        maximum: Optional largest selectable value.
    """

    default: float
    minimum: float
    maximum: float | None = None


class ProviderOption(BaseModel):
    """Describe one provider and the model identifiers exposed for it.

    Attributes:
        value: Stable provider identifier submitted to the backend.
        label: Human-readable provider name.
        models: Models available from this provider in the MVP catalog.
    """

    value: str = Field(min_length=1)
    label: str = Field(min_length=1)
    models: list[Option] = Field(min_length=1)


class GenerationModelCapabilities(BaseModel):
    """Describe the token limits advertised for a generation model.

    Attributes:
        context_window_tokens: Maximum combined input and output token capacity.
        max_output_tokens: Optional independent output limit published by the provider.
    """

    context_window_tokens: int = Field(gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)


class GenerationModelOption(Option):
    """Describe a generation model and its model-specific capabilities.

    Attributes:
        capabilities: Token limits used to validate generation configurations.
    """

    capabilities: GenerationModelCapabilities


class GenerationProviderOption(BaseModel):
    """Describe a generation provider and its capability-aware models.

    Attributes:
        value: Stable provider identifier submitted to the backend.
        label: Human-readable provider name.
        models: Generation models and their provider-advertised capabilities.
    """

    value: str = Field(min_length=1)
    label: str = Field(min_length=1)
    models: list[GenerationModelOption] = Field(min_length=1)


class ChunkingOptions(BaseModel):
    """Group the user-selectable chunking configuration options.

    Attributes:
        strategies: Implemented chunking strategies.
        chunk_size_tokens: Chunk-size defaults and limits.
        chunk_overlap_tokens: Chunk-overlap defaults and limits.
    """

    strategies: list[ChunkingStrategyOption] = Field(min_length=1)
    chunk_size_tokens: IntegerSettingOption
    chunk_overlap_tokens: IntegerSettingOption


class EmbeddingOptions(BaseModel):
    """Group available embedding providers, models, and distance metrics.

    Attributes:
        providers: Backend-supported embedding provider catalogs.
        distance_metrics: Metrics supported by the vector index.
    """

    providers: list[ProviderOption] = Field(min_length=1)
    distance_metrics: list[Option] = Field(min_length=1)


class RetrievalOptions(BaseModel):
    """Group the user-selectable retrieval configuration options.

    Attributes:
        top_k: Default and accepted range for retrieved chunk count.
    """

    top_k: IntegerSettingOption


class GenerationOptions(BaseModel):
    """Group available generation providers, models, and sampling settings.

    Attributes:
        providers: Backend-supported generation provider catalogs.
        temperature: Temperature defaults and limits.
        max_output_tokens: Output-token defaults and limits.
    """

    providers: list[GenerationProviderOption] = Field(min_length=1)
    temperature: FloatSettingOption
    max_output_tokens: IntegerSettingOption


class EvaluationMetricOption(Option):
    """Describe an evaluation metric and its input requirements.

    Attributes:
        requires_reference_answer: Whether the metric needs a reference answer.
        selected_by_default: Whether a new experiment initially selects the metric.
    """

    requires_reference_answer: bool = False
    selected_by_default: bool = False


class EvaluationOptions(BaseModel):
    """Group the retrieval and answer metrics supported by evaluation.

    Attributes:
        retrieval_metrics: Metrics calculated from relevant-document labels.
        answer_metrics: Metrics calculated for generated answers.
    """

    retrieval_metrics: list[EvaluationMetricOption] = Field(min_length=1)
    answer_metrics: list[EvaluationMetricOption] = Field(min_length=1)


class PipelineOptionsResponse(BaseModel):
    """Represent the complete configuration catalog returned to API clients.

    Attributes:
        chunking: Chunking strategy and numeric setting options.
        embedding: Embedding provider, model, and metric options.
        retrieval: Retrieval setting options.
        generation: Generation provider, model, and sampling options.
        evaluation: Evaluation metrics supported by the MVP.
    """

    chunking: ChunkingOptions
    embedding: EmbeddingOptions
    retrieval: RetrievalOptions
    generation: GenerationOptions
    evaluation: EvaluationOptions


@lru_cache(maxsize=1)
def _load_pipeline_options() -> PipelineOptionsResponse:
    """Load and validate the backend pipeline option catalog once.

    Args:
        None. The catalog path is resolved from the backend package.

    Returns:
        The validated pipeline options served to API clients.

    Raises:
        OSError: If the catalog cannot be read.
        ValidationError: If the JSON does not match the response schema.
    """
    # Validate the complete file before allowing any client to consume its values.
    return PipelineOptionsResponse.model_validate_json(
        PIPELINE_OPTIONS_PATH.read_text(encoding="utf-8")
    )


@router.get("/pipeline/options", response_model=PipelineOptionsResponse)
async def get_pipeline_options() -> PipelineOptionsResponse:
    """Return the configuration choices supported by the backend.

    Args:
        None.

    Returns:
        The validated, backend-owned pipeline option catalog.

    Raises:
        HTTPException: If the catalog is missing, unreadable, or invalid.
    """
    try:
        # Serve the cached typed catalog without reading the file for every request.
        return _load_pipeline_options()
    except (OSError, ValidationError) as error:
        # Hide filesystem and validation internals behind a stable API error.
        raise HTTPException(
            status_code=500,
            detail={
                "code": "pipeline_options_unavailable",
                "message": "The pipeline configuration options could not be loaded.",
            },
        ) from error
