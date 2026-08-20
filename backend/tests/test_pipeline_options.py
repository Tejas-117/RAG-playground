"""Tests for the backend-owned pipeline option catalog."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient

from backend.api.routers import pipeline_options
from backend.app import app


def test_pipeline_options_expose_requested_ollama_models() -> None:
    """Verify the MVP catalog contains only the requested Ollama models.

    Args:
        None. The test reads the version-controlled backend catalog.

    Returns:
        None. Assertions verify the configured provider and model identifiers.
    """
    # Resolve the catalog from the backend source tree and parse it as JSON.
    catalog_path = (
        Path(__file__).parents[1]
        / "src"
        / "backend"
        / "config"
        / "pipeline_options.json"
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    # Confirm the embedding picker exposes exactly the locally available models.
    embedding_provider = catalog["embedding"]["providers"][0]
    assert embedding_provider["value"] == "ollama"
    assert [model["value"] for model in embedding_provider["models"]] == [
        "all-minilm",
        "nomic-embed-text",
    ]

    # Confirm generation uses the explicit Ollama identifier for Llama 3.2 3B.
    generation_provider = catalog["generation"]["providers"][0]
    assert generation_provider["value"] == "ollama"
    assert [model["value"] for model in generation_provider["models"]] == [
        "llama3.2:3b"
    ]
    assert generation_provider["models"][0]["capabilities"] == {
        "context_window_tokens": 131072,
        "max_output_tokens": None,
    }

    # Keep initial multi-selection behavior owned by the backend metric catalog.
    answer_metrics = catalog["evaluation"]["answer_metrics"]
    assert [
        metric["value"] for metric in answer_metrics if metric["selected_by_default"]
    ] == ["groundedness", "answer_relevance"]

    # Expose paragraph chunking without implying unavailable section semantics.
    chunking_values = [option["value"] for option in catalog["chunking"]["strategies"]]
    assert "paragraph" in chunking_values
    assert "paragraph_section" not in chunking_values


class PipelineOptionsRouteTestCase(unittest.IsolatedAsyncioTestCase):
    """Regression tests for the pipeline-options API endpoint."""

    def tearDown(self) -> None:
        """Clear the catalog cache after each API test.

        Returns:
            None. Later tests will load the catalog from their active path.
        """
        # Prevent one test's catalog value or error path from leaking into another.
        pipeline_options._load_pipeline_options.cache_clear()

    async def test_pipeline_options_returns_validated_catalog(self) -> None:
        """Verify the endpoint returns the configured pipeline choices.

        Returns:
            None. Assertions verify the public response contract.
        """
        transport = ASGITransport(app=app)

        # Request the catalog through the complete FastAPI application.
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get("/pipeline/options")

        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertEqual(
            [
                model["value"]
                for model in response_data["embedding"]["providers"][0]["models"]
            ],
            ["all-minilm", "nomic-embed-text"],
        )
        self.assertEqual(
            response_data["generation"]["providers"][0]["models"][0]["value"],
            "llama3.2:3b",
        )
        self.assertEqual(
            response_data["generation"]["providers"][0]["models"][0]["capabilities"],
            {
                "context_window_tokens": 131072,
                "max_output_tokens": None,
            },
        )
        self.assertEqual(
            [
                metric["value"]
                for metric in response_data["evaluation"]["answer_metrics"]
                if metric["selected_by_default"]
            ],
            ["groundedness", "answer_relevance"],
        )

    async def test_pipeline_options_returns_structured_error_for_invalid_catalog(
        self,
    ) -> None:
        """Verify invalid catalog JSON produces a stable API error.

        Returns:
            None. Assertions verify malformed configuration is not exposed.
        """
        transport = ASGITransport(app=app)

        with TemporaryDirectory() as directory:
            invalid_catalog_path = Path(directory) / "pipeline_options.json"
            invalid_catalog_path.write_text("{}", encoding="utf-8")

            # Replace the source file and clear the valid catalog cached by earlier calls.
            with patch.object(
                pipeline_options,
                "PIPELINE_OPTIONS_PATH",
                invalid_catalog_path,
            ):
                pipeline_options._load_pipeline_options.cache_clear()

                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    response = await client.get("/pipeline/options")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["detail"],
            {
                "code": "pipeline_options_unavailable",
                "message": "The pipeline configuration options could not be loaded.",
            },
        )
