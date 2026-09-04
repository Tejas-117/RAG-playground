"""Offline tests for the Groq generation-provider adapter."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import groq
import httpx
import pytest

from backend.generation.models import (
    GenerationAuthenticationError,
    GenerationMessage,
    GenerationProviderUnavailableError,
    GenerationRateLimitError,
    InvalidGenerationResponseError,
)
from backend.generation.providers import GroqGenerationProvider


class CapturingCompletions:
    """Capture one SDK request and return a configured response or exception."""

    def __init__(self, outcome: object) -> None:
        """Store the fake SDK outcome.

        Args:
            outcome: Response object to return or exception to raise.

        Returns:
            None. Requests begin empty for later assertions.
        """
        self.outcome = outcome
        self.requests: list[dict[str, object]] = []

    def create(self, **request: object) -> object:
        """Record one chat-completion request and resolve the configured outcome.

        Args:
            request: Keyword arguments supplied by the Groq adapter.

        Returns:
            Configured fake SDK response.

        Raises:
            Exception: Configured fake SDK failure when one was supplied.
        """
        self.requests.append(request)

        # Exception outcomes exercise provider-neutral error translation.
        if isinstance(self.outcome, Exception):
            raise self.outcome

        return self.outcome


def _client(outcome: object) -> tuple[object, CapturingCompletions]:
    """Build a minimal nested client matching the Groq SDK surface.

    Args:
        outcome: Response or exception resolved by chat completions.

    Returns:
        Fake client and its capturing completions object.
    """
    completions = CapturingCompletions(outcome)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def _response(content: str = "Answer [Source 1]") -> object:
    """Create one valid Groq-like SDK response.

    Args:
        content: Assistant answer placed in the single response choice.

    Returns:
        Namespace carrying answer, usage, and provider provenance.
    """
    return SimpleNamespace(
        id="request-1",
        model="openai/gpt-oss-20b",
        system_fingerprint="fingerprint-1",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
        ),
    )


def test_groq_provider_sends_non_streaming_gpt_oss_request() -> None:
    """Verify resolved generation settings and hidden reasoning reach Groq.

    Args:
        None.

    Returns:
        None. Assertions cover request shape and validated response provenance.
    """
    client, completions = _client(_response())
    provider = GroqGenerationProvider(client=client)
    messages = (
        GenerationMessage(role="system", content="Instructions"),
        GenerationMessage(role="user", content="Question and context"),
    )

    result = provider.generate("openai/gpt-oss-20b", messages, 0.2, 1000)

    assert completions.requests == [
        {
            "model": "openai/gpt-oss-20b",
            "messages": [
                {"role": "system", "content": "Instructions"},
                {"role": "user", "content": "Question and context"},
            ],
            "temperature": 0.2,
            "max_completion_tokens": 1000,
            "stream": False,
            "include_reasoning": False,
            "reasoning_effort": "medium",
        }
    ]
    assert result.answer_text == "Answer [Source 1]"
    assert result.total_tokens == 14
    assert result.provider_request_id == "request-1"


def test_groq_provider_does_not_send_gpt_oss_options_to_qwen() -> None:
    """Verify GPT-OSS-only fields are excluded from Qwen requests.

    Args:
        None.

    Returns:
        None. Assertions verify model-specific request compatibility.
    """
    client, completions = _client(_response())
    provider = GroqGenerationProvider(client=client)

    provider.generate(
        "qwen/qwen3.6-27b",
        (GenerationMessage(role="user", content="Question"),),
        0.2,
        100,
    )

    request = completions.requests[0]
    assert "include_reasoning" not in request
    assert "reasoning_effort" not in request


def test_groq_provider_requires_server_side_api_key() -> None:
    """Verify missing local credentials fail before creating a provider client.

    Args:
        None.

    Returns:
        None. An authentication error confirms no unauthenticated request is made.
    """
    # Prevent developer machine values or .env contents from entering the test.
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("backend.generation.providers.load_dotenv"),
        pytest.raises(GenerationAuthenticationError),
    ):
        GroqGenerationProvider().generate(
            "openai/gpt-oss-20b",
            (GenerationMessage(role="user", content="Question"),),
            0.2,
            100,
        )


@pytest.mark.parametrize(
    ("provider_error", "expected_error"),
    [
        (
            groq.RateLimitError(
                "limited",
                response=httpx.Response(
                    429,
                    request=httpx.Request("POST", "https://api.groq.test"),
                ),
                body=None,
            ),
            GenerationRateLimitError,
        ),
        (
            groq.APIStatusError(
                "capacity",
                response=httpx.Response(
                    498,
                    request=httpx.Request("POST", "https://api.groq.test"),
                ),
                body=None,
            ),
            GenerationProviderUnavailableError,
        ),
    ],
)
def test_groq_provider_maps_safe_operational_failures(
    provider_error: Exception,
    expected_error: type[Exception],
) -> None:
    """Verify rate and capacity failures become provider-neutral categories.

    Args:
        provider_error: Groq SDK exception selected by parametrization.
        expected_error: Provider-neutral exception expected from the adapter.

    Returns:
        None. An exception assertion verifies safe translation.
    """
    client, _ = _client(provider_error)
    provider = GroqGenerationProvider(client=client)

    with pytest.raises(expected_error):
        provider.generate(
            "openai/gpt-oss-20b",
            (GenerationMessage(role="user", content="Question"),),
            0.2,
            100,
        )


def test_groq_provider_rejects_empty_answer_response() -> None:
    """Verify blank assistant content cannot enter answer persistence.

    Args:
        None.

    Returns:
        None. An invalid-response error confirms boundary validation.
    """
    client, _ = _client(_response("   "))
    provider = GroqGenerationProvider(client=client)

    with pytest.raises(InvalidGenerationResponseError, match="empty answer"):
        provider.generate(
            "openai/gpt-oss-20b",
            (GenerationMessage(role="user", content="Question"),),
            0.2,
            100,
        )
