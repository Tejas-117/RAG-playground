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
    "qwen/qwen3.6-27b": GenerationModelCapabilities(131_072, 16_384),
    "qwen/qwen3.8-27b": GenerationModelCapabilities(131_042, 16_384),
}

# Conservative local limits avoid assuming that Ollama was started with 128K context.
OLLAMA_MODEL_CAPABILITIES: dict[str, GenerationModelCapabilities] = {
    "llama3.2:1b": GenerationModelCapabilities(8_192, 2_048),
    "llama3.2:3b": GenerationModelCapabilities(8_192, 2_048),
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
    # Resolve limits through the provider namespace so identical tags cannot collide.
    provider_catalogs = {
        "groq": GROQ_MODEL_CAPABILITIES,
        "ollama": OLLAMA_MODEL_CAPABILITIES,
    }

    # An unknown provider cannot be packed safely or resolved to an adapter.
    if provider not in provider_catalogs:
        raise LookupError(f"Generation provider '{provider}' is not registered.")

    # A model without trusted limits cannot be packed safely into a request.
    try:
        return provider_catalogs[provider][model]
    except KeyError as error:
        raise LookupError(f"Generation model '{model}' is not registered.") from error
