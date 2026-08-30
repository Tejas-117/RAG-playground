"""Backend-owned capabilities for generation models exposed by the API catalog."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationModelCapabilities:
    """Describe input and output limits used while assembling generation prompts.

    Attributes:
        context_window_tokens: Provider-advertised combined input/output limit.
        max_output_tokens: Provider-advertised completion limit.
    """

    context_window_tokens: int
    max_output_tokens: int


# Keep runtime prompt budgeting aligned with the version-controlled public catalog.
GROQ_MODEL_CAPABILITIES: dict[str, GenerationModelCapabilities] = {
    "openai/gpt-oss-20b": GenerationModelCapabilities(131_072, 65_536),
    "openai/gpt-oss-120b": GenerationModelCapabilities(131_072, 65_536),
    "llama-3.2-3b-preview": GenerationModelCapabilities(128_000, 8_000),
}


def get_generation_model_capabilities(
    provider: str,
    model: str,
) -> GenerationModelCapabilities:
    """Return trusted prompt limits for one configured provider/model pair.

    Args:
        provider: Backend-registered generation provider identifier.
        model: Provider model identifier from the immutable run configuration.

    Returns:
        Provider-advertised context and completion limits.

    Raises:
        LookupError: If the selected provider or model has no registered limits.
    """
    # Groq is the only executable generation provider in the current backend.
    if provider != "groq":
        raise LookupError(f"Generation provider '{provider}' is not registered.")

    # A model without trusted limits cannot be packed safely into a request.
    try:
        return GROQ_MODEL_CAPABILITIES[model]
    except KeyError as error:
        raise LookupError(f"Generation model '{model}' is not registered.") from error
