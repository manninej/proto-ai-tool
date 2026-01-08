from __future__ import annotations

import json

import respx
from click.testing import CliRunner
import httpx

from saga_code.cli import main
from saga_code.model_discovery import discover_models
from saga_code.openai_client import OpenAIClient


def test_list_models_success() -> None:
    with respx.mock(base_url="https://api.example.com") as router:
        router.get("/v1/models").respond(200, json={"data": [{"id": "model-a"}]})
        client = OpenAIClient("https://api.example.com", api_key="test", timeout=5)
        assert client.list_models() == ["model-a"]


def test_discover_models_fallback_probe() -> None:
    with respx.mock(base_url="https://api.example.com") as router:
        router.get("/v1/models").respond(403, json={"error": "forbidden"})
        router.post("/v1/chat/completions", json__model="model-a").respond(200, json={"id": "ok"})
        router.post("/v1/chat/completions", json__model="model-b").respond(404, json={"error": "not found"})

        client = OpenAIClient("https://api.example.com", api_key="test", timeout=5)
        results = discover_models(client, prefer_endpoint="auto", candidates=["model-a", "model-b"])

    assert [result.status for result in results] == ["available", "unavailable"]
    assert [result.discovery_method for result in results] == ["probe", "probe"]


def test_cli_json_output() -> None:
    runner = CliRunner()
    with respx.mock(base_url="https://api.example.com") as router:
        router.get("/v1/models").respond(200, json={"data": [{"id": "model-a"}]})
        result = runner.invoke(
            main,
            ["models", "--base-url", "https://api.example.com", "--json"],
            env={"RICH_DISABLE": "1"},
        )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == [
        {
            "model_id": "model-a",
            "discovery_method": "models_endpoint",
            "status": "available",
            "details": "listed",
        }
    ]
