"""Provider-neutral answer-generation services and contracts."""

from backend.generation.models import (
    GenerationProvider,
    GenerationProviderResponse,
    GenerationServiceResult,
)
from backend.generation.service import generate_answer

__all__ = [
    "GenerationProvider",
    "GenerationProviderResponse",
    "GenerationServiceResult",
    "generate_answer",
]
