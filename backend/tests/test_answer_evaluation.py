"""Offline tests for the fixed OpenRouter answer-judge adapter."""

import json as json_module

import httpx
import pytest

from backend.evaluation.answer import (
    OPENROUTER_CHAT_COMPLETIONS_URL,
    AnswerJudgeError,
    AnswerJudgeInput,
    OpenRouterAnswerJudge,
)


class FakeHttpClient:
    """Capture one OpenRouter request and return configured JSON content."""

    def __init__(
        self,
        payload: dict[str, object],
        status_code: int = 200,
    ) -> None:
        """Configure the JSON content returned by the fake endpoint.

        Args:
            payload: Structured response object to encode as assistant content.
            status_code: HTTP status returned by the fake endpoint.

        Returns:
            None. Requests begin with an empty history.
        """
        self.payload = payload
        self.status_code = status_code
        self.requests: list[dict[str, object]] = []

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
    ) -> httpx.Response:
        """Record request arguments and return one HTTP response.

        Args:
            url: OpenRouter chat-completions endpoint.
            headers: Request authentication and content headers.
            json: OpenAI-compatible chat-completion body.

        Returns:
            Minimal successful response accepted by the production adapter.
        """
        self.requests.append({"url": url, "headers": headers, "json": json})
        request = httpx.Request("POST", url)
        return httpx.Response(
            self.status_code,
            request=request,
            json={
                "choices": [{"message": {"content": json_module.dumps(self.payload)}}],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                    "total_tokens": 28,
                },
                "id": "request-1",
                "model": "qwen/qwen3.8-27b:free",
            },
        )


def _judge_input() -> AnswerJudgeInput:
    """Create one fixed multi-metric judge request.

    Returns:
        Input containing exact context and an optional reference answer.
    """
    return AnswerJudgeInput(
        question="What is saved?",
        answer="A saved answer.",
        context_chunks=("First context", "Second context"),
        reference_answer="A saved answer.",
        metrics=("groundedness", "answer_relevance"),
    )


def test_openrouter_judge_uses_strict_schema_and_normalizes_scores() -> None:
    """Verify the fixed request policy and validated normalized result.

    Returns:
        None. Assertions cover strict output, low reasoning, usage, and scores.
    """
    client = FakeHttpClient(
        {
            "results": [
                {
                    "metric": "groundedness",
                    "score": 4,
                    "rationale": "Both claims use the first context.",
                    "evidence_ranks": [1],
                },
                {
                    "metric": "answer_relevance",
                    "score": 3,
                    "rationale": "The answer directly addresses the question.",
                    "evidence_ranks": [],
                },
            ]
        }
    )
    result = OpenRouterAnswerJudge(client, api_key="test-key").judge(_judge_input())
    captured = client.requests[0]
    request = captured["json"]

    assert captured["url"] == OPENROUTER_CHAT_COMPLETIONS_URL
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert request["model"] == "qwen/qwen3.8-27b:free"
    assert request["temperature"] == 0
    assert request["reasoning"] == {"enabled": False}
    assert request["response_format"]["json_schema"]["strict"] is True
    assert result.results["groundedness"]["score"] == 1.0
    assert result.results["answer_relevance"]["score"] == 0.75
    assert result.total_tokens == 28


def test_openrouter_judge_rejects_context_rank_outside_saved_evidence() -> None:
    """Verify evidence citations cannot reference context not used for generation.

    Returns:
        None. The adapter raises a safe structured response error.
    """
    client = FakeHttpClient(
        {
            "results": [
                {
                    "metric": "groundedness",
                    "score": 4,
                    "rationale": "Invalid citation.",
                    "evidence_ranks": [3],
                },
                {
                    "metric": "answer_relevance",
                    "score": 4,
                    "rationale": "Relevant.",
                    "evidence_ranks": [],
                },
            ]
        }
    )
    with pytest.raises(AnswerJudgeError) as error:
        OpenRouterAnswerJudge(client, api_key="test-key").judge(_judge_input())

    assert error.value.code == "invalid_evaluator_response"


def test_openrouter_judge_discards_evidence_for_non_groundedness_metrics() -> None:
    """Verify incidental citations do not invalidate non-groundedness judgments.

    Returns:
        None. The adapter retains groundedness evidence and clears irrelevant citations.
    """
    client = FakeHttpClient(
        {
            "results": [
                {
                    "metric": "groundedness",
                    "score": 4,
                    "rationale": "The first context supports the answer.",
                    "evidence_ranks": [1],
                },
                {
                    "metric": "answer_relevance",
                    "score": 4,
                    "rationale": "The answer directly addresses the question.",
                    "evidence_ranks": [1],
                },
            ]
        }
    )
    result = OpenRouterAnswerJudge(client, api_key="test-key").judge(_judge_input())

    assert result.results["groundedness"]["evidence_ranks"] == [1]
    assert result.results["answer_relevance"]["evidence_ranks"] == []


@pytest.mark.parametrize(
    ("status_code", "expected_code"),
    [
        (401, "evaluator_authentication_failed"),
        (429, "evaluator_rate_limited"),
        (503, "evaluator_unavailable"),
        (400, "evaluator_request_rejected"),
    ],
)
def test_openrouter_judge_maps_http_failures(
    status_code: int,
    expected_code: str,
) -> None:
    """Verify provider HTTP failures become stable evaluator errors.

    Args:
        status_code: Fake OpenRouter status selected by parametrization.
        expected_code: Safe application error expected for the status.

    Returns:
        None. Assertions cover the public failure classification.
    """
    client = FakeHttpClient({}, status_code=status_code)

    with pytest.raises(AnswerJudgeError) as error:
        OpenRouterAnswerJudge(client, api_key="test-key").judge(_judge_input())

    assert error.value.code == expected_code
