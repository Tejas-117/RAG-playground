"""Compatibility validation for effective pipeline run configurations."""

from typing import TYPE_CHECKING, Any

from backend.pipeline.configs import PipelineConfig, PreparationConfig

if TYPE_CHECKING:
    from backend.api.routers.pipeline_options import (
        FloatSettingOption,
        IntegerSettingOption,
        PipelineOptionsResponse,
    )


class InvalidPipelineConfigurationError(ValueError):
    """Report a configuration value that is not supported by the option catalog.

    Attributes:
        field: Dot-separated request field containing the invalid value.
        message: User-readable explanation of the compatibility failure.
    """

    def __init__(self, field: str, message: str) -> None:
        """Create a pipeline compatibility error.

        Args:
            field: Dot-separated request field containing the invalid value.
            message: User-readable explanation of the compatibility failure.

        Returns:
            None. The initialized exception carries structured error details.
        """
        # Initialize ValueError for conventional exception logging and handling.
        super().__init__(message)
        self.field = field
        self.message = message


def _validate_numeric_setting(
    field: str,
    value: float,
    setting: "IntegerSettingOption | FloatSettingOption",
) -> None:
    """Validate a numeric value against catalog-owned inclusive bounds.

    Args:
        field: Dot-separated configuration field being validated.
        value: Submitted numeric value.
        setting: Catalog setting containing minimum and optional maximum values.

    Returns:
        None. Successful validation has no transformed result.

    Raises:
        InvalidPipelineConfigurationError: If the value is outside catalog bounds.
    """
    # Enforce the backend catalog's lower bound even when the Pydantic model is broader.
    if value < setting.minimum:
        raise InvalidPipelineConfigurationError(
            field,
            f"Value must be at least {setting.minimum}.",
        )

    # Enforce an upper bound only when the catalog publishes one.
    if setting.maximum is not None and value > setting.maximum:
        raise InvalidPipelineConfigurationError(
            field,
            f"Value must be at most {setting.maximum}.",
        )


def _find_provider(
    providers: list[Any],
    provider_value: str,
    field: str,
) -> Any:
    """Find a provider in a backend-owned option list.

    Args:
        providers: Provider options exposed by the backend catalog.
        provider_value: Stable provider identifier from the run configuration.
        field: Dot-separated provider field used in structured errors.

    Returns:
        The matching provider option.

    Raises:
        InvalidPipelineConfigurationError: If the provider is not available.
    """
    # Search the small version-controlled provider catalog by stable identifier.
    for provider in providers:
        if provider.value == provider_value:
            return provider

    # Reject identifiers that the backend cannot execute rather than storing invalid runs.
    raise InvalidPipelineConfigurationError(
        field,
        f"Provider '{provider_value}' is not supported.",
    )


def _validate_model(provider: Any, model_value: str, field: str) -> Any:
    """Validate that a model belongs to its selected provider.

    Args:
        provider: Selected provider option containing its supported models.
        model_value: Stable model identifier from the run configuration.
        field: Dot-separated model field used in structured errors.

    Returns:
        The matching model option.

    Raises:
        InvalidPipelineConfigurationError: If the provider does not expose the model.
    """
    # Match within the selected provider so cross-provider model combinations fail.
    for model in provider.models:
        if model.value == model_value:
            return model

    # Explain the incompatible pair without exposing internal adapter details.
    raise InvalidPipelineConfigurationError(
        field,
        f"Model '{model_value}' is not available for provider '{provider.value}'.",
    )


def _validate_metrics(
    selected_metrics: list[str],
    available_metrics: list[Any],
    field: str,
) -> None:
    """Validate selected metrics against one backend-owned metric catalog.

    Args:
        selected_metrics: Stable metric identifiers selected for the run.
        available_metrics: Metric options exposed by the backend catalog.
        field: Dot-separated configuration field used in structured errors.

    Returns:
        None. Successful validation leaves the selected list unchanged.

    Raises:
        InvalidPipelineConfigurationError: If a selected metric is unavailable.
    """
    available_values = {metric.value for metric in available_metrics}

    # Reject the first identifier that the current backend cannot evaluate.
    for metric in selected_metrics:
        if metric not in available_values:
            raise InvalidPipelineConfigurationError(
                field,
                f"Evaluation metric '{metric}' is not supported.",
            )


def validate_pipeline_config(
    configuration: PipelineConfig,
    options: "PipelineOptionsResponse",
    *,
    has_evaluation_dataset: bool = False,
) -> None:
    """Validate one resolved configuration against the current option catalog.

    Args:
        configuration: Typed, default-resolved configuration submitted for a run.
        options: Validated backend-owned pipeline option catalog.
        has_evaluation_dataset: Whether stable labels and references are available.

    Returns:
        None. The configuration is unchanged when it is compatible.

    Raises:
        InvalidPipelineConfigurationError: If any option or combination is unsupported.
    """
    # Validate the shared preparation stages before checking query-specific stages.
    validate_preparation_config(configuration, options)

    _validate_numeric_setting(
        "configuration.retrieval.top_k",
        configuration.retrieval.top_k,
        options.retrieval.top_k,
    )

    generation_provider = _find_provider(
        options.generation.providers,
        configuration.generation.provider,
        "configuration.generation.provider",
    )
    generation_model = _validate_model(
        generation_provider,
        configuration.generation.model,
        "configuration.generation.model",
    )
    _validate_numeric_setting(
        "configuration.generation.temperature",
        configuration.generation.temperature,
        options.generation.temperature,
    )
    _validate_numeric_setting(
        "configuration.generation.max_output_tokens",
        configuration.generation.max_output_tokens,
        options.generation.max_output_tokens,
    )

    # A requested output cannot exceed the model's entire combined context window.
    if (
        configuration.generation.max_output_tokens
        > generation_model.capabilities.context_window_tokens
    ):
        raise InvalidPipelineConfigurationError(
            "configuration.generation.max_output_tokens",
            "Maximum output tokens exceed the selected model's context window.",
        )

    model_output_limit = generation_model.capabilities.max_output_tokens

    # Enforce a separate provider-published output limit when the model declares one.
    if (
        model_output_limit is not None
        and configuration.generation.max_output_tokens > model_output_limit
    ):
        raise InvalidPipelineConfigurationError(
            "configuration.generation.max_output_tokens",
            "Maximum output tokens exceed the selected model's output limit.",
        )

    _validate_metrics(
        configuration.evaluation.retrieval_metrics,
        options.evaluation.retrieval_metrics,
        "configuration.evaluation.retrieval_metrics",
    )
    _validate_metrics(
        configuration.evaluation.answer_metrics,
        options.evaluation.answer_metrics,
        "configuration.evaluation.answer_metrics",
    )

    # Ad hoc runs have no labelled relevant documents for retrieval metrics.
    if not has_evaluation_dataset and configuration.evaluation.retrieval_metrics:
        raise InvalidPipelineConfigurationError(
            "configuration.evaluation.retrieval_metrics",
            "Retrieval metrics require an evaluation dataset with relevance labels.",
        )

    answer_options = {
        metric.value: metric for metric in options.evaluation.answer_metrics
    }

    # Ad hoc runs cannot use metrics that compare against a reference answer.
    for metric in configuration.evaluation.answer_metrics:
        if (
            not has_evaluation_dataset
            and answer_options[metric].requires_reference_answer
        ):
            raise InvalidPipelineConfigurationError(
                "configuration.evaluation.answer_metrics",
                f"Evaluation metric '{metric}' requires a reference answer.",
            )


def validate_preparation_config(
    configuration: PreparationConfig | PipelineConfig,
    options: "PipelineOptionsResponse",
) -> None:
    """Validate chunking and embedding settings against the option catalog.

    Args:
        configuration: Configuration containing chunking and embedding stages.
        options: Validated backend-owned pipeline option catalog.

    Returns:
        None. The configuration is unchanged when it is compatible.

    Raises:
        InvalidPipelineConfigurationError: If a preparation option is unsupported.
    """
    # Resolve the selected chunker so strategy-specific catalog capabilities are enforced.
    selected_strategy = next(
        (
            strategy
            for strategy in options.chunking.strategies
            if strategy.value == configuration.chunking.strategy.value
        ),
        None,
    )

    # Reject code-level strategies that are not currently exposed for execution.
    if selected_strategy is None:
        raise InvalidPipelineConfigurationError(
            "configuration.chunking.strategy",
            f"Chunking strategy '{configuration.chunking.strategy.value}' is not supported.",
        )

    _validate_numeric_setting(
        "configuration.chunking.chunk_size_tokens",
        configuration.chunking.chunk_size_tokens,
        options.chunking.chunk_size_tokens,
    )
    _validate_numeric_setting(
        "configuration.chunking.chunk_overlap_tokens",
        configuration.chunking.chunk_overlap_tokens or 0,
        options.chunking.chunk_overlap_tokens,
    )

    # Guard against overlap if a future catalog disables it for another strategy.
    if (
        not selected_strategy.supports_overlap
        and configuration.chunking.chunk_overlap_tokens != 0
    ):
        raise InvalidPipelineConfigurationError(
            "configuration.chunking.chunk_overlap_tokens",
            "The selected chunking strategy does not support overlap.",
        )

    embedding_provider = _find_provider(
        options.embedding.providers,
        configuration.embedding.provider,
        "configuration.embedding.provider",
    )
    _validate_model(
        embedding_provider,
        configuration.embedding.model,
        "configuration.embedding.model",
    )

    # Ensure the requested vector score semantics are implemented by the backend.
    if configuration.embedding.distance_metric.value not in {
        metric.value for metric in options.embedding.distance_metrics
    }:
        raise InvalidPipelineConfigurationError(
            "configuration.embedding.distance_metric",
            "The selected distance metric is not supported.",
        )
