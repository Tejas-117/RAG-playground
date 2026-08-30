"""Immutable provider-neutral contracts for answer generation."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GenerationMessage:
    """Represent one ordered chat message sent to a generation provider.

    Attributes:
        role: Chat role understood by the provider adapter.
        content: Complete instruction, question, or source-context text.
    """

    role: str
    content: str


@dataclass(frozen=True)
class GenerationProviderResponse:
    """Return one validated answer and optional provider provenance.

    Attributes:
        answer_text: Non-empty assistant answer returned by the provider.
        provider_model: Provider-reported model identifier when available.
        finish_reason: Provider reason for ending the completion.
        prompt_tokens: Provider-reported input token count when available.
        completion_tokens: Provider-reported output token count when available.
        total_tokens: Provider-reported combined token count when available.
        provider_request_id: Provider request identifier useful for diagnostics.
        system_fingerprint: Provider backend fingerprint when available.
    """

    answer_text: str
    provider_model: str | None
    finish_reason: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    provider_request_id: str | None = None
    system_fingerprint: str | None = None


@dataclass(frozen=True)
class GenerationServiceResult:
    """Combine provider output with exact prompt and context provenance.

    Attributes:
        response: Validated answer and provider metadata.
        context_chunk_ids: Exact ranked chunk IDs included in the prompt.
        prompt_template_version: Stable prompt-content policy identifier.
        provider_policy_version: Stable provider request policy identifier.
        provider_called: Whether a remote request was necessary.
    """

    response: GenerationProviderResponse
    context_chunk_ids: tuple[str, ...]
    prompt_template_version: str
    provider_policy_version: str
    provider_called: bool


class GenerationProviderError(RuntimeError):
    """Base class for safe provider-neutral generation failures."""


class GenerationProviderUnavailableError(GenerationProviderError):
    """Report that the configured generation provider cannot complete a request."""


class GenerationRequestTimeoutError(GenerationProviderError):
    """Report that a generation request exceeded its configured timeout."""


class GenerationAuthenticationError(GenerationProviderError):
    """Report missing or rejected backend generation credentials."""


class GenerationRateLimitError(GenerationProviderError):
    """Report temporary rejection caused by provider request or token limits."""


class GenerationRequestRejectedError(GenerationProviderError):
    """Report a provider-rejected model or request without leaking raw details."""


class GenerationInputTooLargeError(GenerationRequestRejectedError):
    """Report a question or context that cannot fit the configured model budget."""


class InvalidGenerationResponseError(GenerationProviderError):
    """Report a provider response that cannot represent one generated answer."""


class GenerationProvider(Protocol):
    """Define the behavior required from every generation provider adapter."""

    identifier: str
    version: str

    def policy_version(self, model: str) -> str:
        """Return the stable request policy applied to one provider model.

        Args:
            model: Provider model identifier selected by the run.

        Returns:
            Versioned provider policy stored with generation provenance.
        """
        ...

    def generate(
        self,
        model: str,
        messages: tuple[GenerationMessage, ...],
        temperature: float,
        max_output_tokens: int,
    ) -> GenerationProviderResponse:
        """Generate one answer from ordered messages and resolved settings.

        Args:
            model: Provider model identifier selected by the run.
            messages: Ordered prompt messages assembled by the generation service.
            temperature: Sampling temperature from the immutable run snapshot.
            max_output_tokens: Maximum completion tokens requested from the provider.

        Returns:
            Validated answer and provider-reported provenance.
        """
        ...
