"""Groq HTTP-SDK adapter behind the provider-neutral generation contract."""

import os
from pathlib import Path
from typing import Any

import groq
from dotenv import load_dotenv

from backend.generation.models import (
    GenerationAuthenticationError,
    GenerationInputTooLargeError,
    GenerationMessage,
    GenerationProvider,
    GenerationProviderResponse,
    GenerationProviderUnavailableError,
    GenerationRateLimitError,
    GenerationRequestRejectedError,
    GenerationRequestTimeoutError,
    InvalidGenerationResponseError,
)

# Local development credentials live at backend/.env and remain gitignored.
BACKEND_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"

# Bound one synchronous provider call executed inside the worker thread.
GROQ_REQUEST_TIMEOUT_SECONDS = 120.0

# Request-policy changes can alter output and must remain visible in provenance.
_GROQ_POLICY_VERSION = "groq-chat-completions-v1"


class GroqGenerationProvider:
    """Generate answers through Groq without exposing SDK objects to services."""

    identifier = "groq"
    version = "1"

    def __init__(self, client: Any | None = None) -> None:
        """Configure an injectable synchronous Groq client.

        Args:
            client: Optional fake or preconfigured client used by offline tests.

        Returns:
            None. A real client is created lazily only when generation is attempted.
        """
        # Injected clients make unit tests deterministic; production stays lazy.
        self._client = client

    def policy_version(self, model: str) -> str:
        """Return the versioned Groq request policy for one model.

        Args:
            model: Groq model identifier selected by the run.

        Returns:
            Stable policy identity persisted with the generated answer.
        """
        # The first policy uses non-streaming chat completions with no external tools.
        return _GROQ_POLICY_VERSION

    def generate(
        self,
        model: str,
        messages: tuple[GenerationMessage, ...],
        temperature: float,
        max_output_tokens: int,
    ) -> GenerationProviderResponse:
        """Request and validate one non-streaming Groq chat completion.

        Args:
            model: Groq model identifier selected by the run.
            messages: Ordered system and user messages built by the prompt service.
            temperature: Sampling temperature from the run snapshot.
            max_output_tokens: Maximum completion size requested from Groq.

        Returns:
            Provider-neutral validated answer and usage provenance.

        Raises:
            GenerationProviderError: If transport, HTTP, or validation fails.
        """
        request_arguments: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": temperature,
            "max_completion_tokens": max_output_tokens,
            "stream": False,
        }

        # GPT-OSS should return only its final answer, not internal reasoning text.
        if model.casefold().startswith("openai/gpt-oss-"):
            request_arguments["include_reasoning"] = False
            request_arguments["reasoning_effort"] = "medium"

        client = self._client or self._create_client()

        try:
            # The synchronous call runs in the pipeline worker's dedicated thread.
            response = client.chat.completions.create(**request_arguments)
        except groq.APITimeoutError as error:
            raise GenerationRequestTimeoutError(
                "The generation provider request timed out."
            ) from error
        except (groq.AuthenticationError, groq.PermissionDeniedError) as error:
            raise GenerationAuthenticationError(
                "The generation provider rejected backend authentication."
            ) from error
        except groq.RateLimitError as error:
            raise GenerationRateLimitError(
                "The generation provider rate limit was reached."
            ) from error
        except groq.APIConnectionError as error:
            raise GenerationProviderUnavailableError(
                "The generation provider could not be reached."
            ) from error
        except groq.APIStatusError as error:
            self._raise_status_error(error)
        except groq.APIResponseValidationError as error:
            raise InvalidGenerationResponseError(
                "The generation provider returned an invalid response."
            ) from error

        return self._validate_response(response)

    def _create_client(self) -> groq.Groq:
        """Create one bounded real Groq client from server-side credentials.

        Args:
            None.

        Returns:
            Authenticated synchronous Groq SDK client with retries disabled.

        Raises:
            GenerationAuthenticationError: If ``GROQ_API_KEY`` is unavailable.
        """
        # Load local values without overriding deployment environment values.
        load_dotenv(BACKEND_ENV_PATH, override=False)
        api_key = os.getenv("GROQ_API_KEY", "").strip()

        # Missing credentials fail only a run that actually needs a remote call.
        if not api_key:
            raise GenerationAuthenticationError(
                "The generation provider API key is not configured."
            )

        # Disable SDK retries so one run maps to one visible provider attempt.
        return groq.Groq(
            api_key=api_key,
            timeout=GROQ_REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )

    def _raise_status_error(self, error: groq.APIStatusError) -> None:
        """Translate one Groq HTTP failure into a safe provider-neutral category.

        Args:
            error: Groq SDK status exception containing HTTP response metadata.

        Returns:
            Never returns because a mapped generation exception is always raised.

        Raises:
            GenerationProviderError: Stable category derived from safe status data.
        """
        status_code = error.status_code

        # Capacity and server failures describe temporary provider availability.
        if status_code == 498 or status_code >= 500:
            raise GenerationProviderUnavailableError(
                "The generation provider could not complete the request."
            ) from error

        response_text = str(error).casefold()

        # Recognize common provider wording for a request exceeding model limits.
        if status_code in {400, 413, 422} and any(
            marker in response_text
            for marker in (
                "context length",
                "too long",
                "input length",
                "request too large",
            )
        ):
            raise GenerationInputTooLargeError(
                "The generation prompt exceeds the selected model's input limit."
            ) from error

        # Other client failures may be model access or request compatibility issues.
        raise GenerationRequestRejectedError(
            "The generation provider rejected the model or request."
        ) from error

    def _validate_response(self, response: Any) -> GenerationProviderResponse:
        """Validate an untrusted Groq SDK response before domain use.

        Args:
            response: SDK response object returned by chat completions.

        Returns:
            Immutable provider-neutral answer and optional usage fields.

        Raises:
            InvalidGenerationResponseError: If required answer fields are malformed.
        """
        choices = getattr(response, "choices", None)

        # The current contract requests exactly one completion choice.
        if not isinstance(choices, list) or len(choices) != 1:
            raise InvalidGenerationResponseError(
                "The generation provider returned an unexpected choice count."
            )

        choice = choices[0]
        message = getattr(choice, "message", None)
        answer_text = getattr(message, "content", None)
        finish_reason = getattr(choice, "finish_reason", None)

        # Empty or missing assistant content cannot be persisted as an answer.
        if not isinstance(answer_text, str) or not answer_text.strip():
            raise InvalidGenerationResponseError(
                "The generation provider returned an empty answer."
            )

        # Finish provenance must be meaningful even for length-limited answers.
        if not isinstance(finish_reason, str) or not finish_reason.strip():
            raise InvalidGenerationResponseError(
                "The generation provider omitted its finish reason."
            )

        usage = getattr(response, "usage", None)
        prompt_tokens = self._optional_non_negative_integer(usage, "prompt_tokens")
        completion_tokens = self._optional_non_negative_integer(
            usage,
            "completion_tokens",
        )
        total_tokens = self._optional_non_negative_integer(usage, "total_tokens")

        return GenerationProviderResponse(
            answer_text=answer_text.strip(),
            provider_model=self._optional_non_empty_string(response, "model"),
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            provider_request_id=self._optional_non_empty_string(response, "id"),
            system_fingerprint=self._optional_non_empty_string(
                response,
                "system_fingerprint",
            ),
        )

    def _optional_non_negative_integer(
        self,
        value: Any,
        attribute: str,
    ) -> int | None:
        """Read and validate one optional non-negative integer attribute.

        Args:
            value: SDK object that may contain the requested usage attribute.
            attribute: Attribute name to read from the object.

        Returns:
            Valid integer value or ``None`` when the provider omitted it.

        Raises:
            InvalidGenerationResponseError: If a present value is invalid.
        """
        # Missing usage objects or attributes are allowed by the provider-neutral model.
        if value is None:
            return None

        result = getattr(value, attribute, None)
        if result is None:
            return None

        # Booleans are not meaningful token counts despite being Python integers.
        if isinstance(result, bool) or not isinstance(result, int) or result < 0:
            raise InvalidGenerationResponseError(
                "The generation provider returned invalid token usage."
            )

        return result

    def _optional_non_empty_string(
        self,
        value: Any,
        attribute: str,
    ) -> str | None:
        """Read and validate one optional non-empty string attribute.

        Args:
            value: SDK object that may contain the requested attribute.
            attribute: Attribute name to read from the object.

        Returns:
            Stripped string or ``None`` when the provider omitted it.

        Raises:
            InvalidGenerationResponseError: If a present value is not meaningful text.
        """
        result = getattr(value, attribute, None)

        # Optional provider provenance may be absent without invalidating the answer.
        if result is None:
            return None

        if not isinstance(result, str) or not result.strip():
            raise InvalidGenerationResponseError(
                "The generation provider returned invalid provenance."
            )

        return result.strip()


def get_generation_provider(provider: str) -> GenerationProvider:
    """Resolve one backend-registered generation provider adapter.

    Args:
        provider: Provider identifier from the validated run configuration.

    Returns:
        Stateless provider adapter implementing the shared generation protocol.

    Raises:
        GenerationRequestRejectedError: If no adapter is registered.
    """
    # Provider resolution remains centralized as additional APIs are introduced.
    if provider == "groq":
        return GroqGenerationProvider()

    raise GenerationRequestRejectedError(
        "The selected generation provider is not registered."
    )
