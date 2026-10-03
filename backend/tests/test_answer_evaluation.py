"""Offline tests for the fixed structured answer-judge adapter."""

import json
from types import SimpleNamespace

import pytest

from backend.evaluation.answer import (
    AnswerJudgeError,
    AnswerJudgeInput,
    GroqAnswerJudge,
)


class FakeCompletions:
    """Capture one Groq request and return configured structured content."""

    def __init__(self, payload: dict[str, object]) -> None:
        """Configure the JSON content returned by the fake endpoint.

        Args:
            payload: Structured response object to encode as assistant content.

        Returns:
            None. Requests begin with an empty history.
        """
        self.payload = payload
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        """Record request arguments and return one SDK-shaped completion.

        Args:
            **kwargs: Groq chat-completion request fields.

        Returns:
            Minimal response object accepted by the production adapter.
        """
        self.requests.append(kwargs)
        message = SimpleNamespace(content=json.dumps(self.payload))
        choice = SimpleNamespace(message=message)
        usage = SimpleNamespace(prompt_tokens=20, completion_tokens=8, total_tokens=28)
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            id="request-1",
            model="openai/gpt-oss-20b",
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


def test_groq_judge_uses_strict_schema_and_normalizes_scores() -> None:
    """Verify the fixed request policy and validated normalized result.

    Returns:
        None. Assertions cover strict output, low reasoning, usage, and scores.
    """
    completions = FakeCompletions(
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
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = GroqAnswerJudge(client).judge(_judge_input())
    request = completions.requests[0]

    assert request["model"] == "openai/gpt-oss-20b"
    assert request["temperature"] == 0
    assert request["reasoning_effort"] == "low"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert result.results["groundedness"]["score"] == 1.0
    assert result.results["answer_relevance"]["score"] == 0.75
    assert result.total_tokens == 28


def test_groq_judge_rejects_context_rank_outside_saved_evidence() -> None:
    """Verify evidence citations cannot reference context not used for generation.

    Returns:
        None. The adapter raises a safe structured response error.
    """
    completions = FakeCompletions(
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
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    with pytest.raises(AnswerJudgeError) as error:
        GroqAnswerJudge(client).judge(_judge_input())

    assert error.value.code == "invalid_evaluator_response"


def test_groq_judge_discards_evidence_for_non_groundedness_metrics() -> None:
    """Verify incidental citations do not invalidate non-groundedness judgments.

    Returns:
        None. The adapter retains groundedness evidence and clears irrelevant citations.
    """
    completions = FakeCompletions(
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
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = GroqAnswerJudge(client).judge(_judge_input())

    assert result.results["groundedness"]["evidence_ranks"] == [1]
    assert result.results["answer_relevance"]["evidence_ranks"] == []
