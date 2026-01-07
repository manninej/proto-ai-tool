from __future__ import annotations

import json
from typing import Iterable

import click
from rich.console import Console
from rich.panel import Panel

from proto_codegen import __version__
from proto_codegen.config import Config, load_config
from proto_codegen.model_discovery import ModelResult, discover_models
from proto_codegen.openai_client import ApiError, NetworkError, OpenAIClient
from proto_codegen.render import print_error_panel, print_json, print_models_table

console = Console()


@click.group()
def main() -> None:
    """proto-codegen CLI."""


@main.command()
@click.option("--base-url", envvar="OPENAI_BASE_URL", help="Override the API base URL.")
@click.option("--api-key", envvar="OPENAI_API_KEY", help="Override the API key.")
@click.option("--timeout", envvar="PROTO_CODEGEN_TIMEOUT", type=int, help="Request timeout in seconds.")
@click.option(
    "--ca-bundle",
    envvar="PROTO_CODEGEN_CA_BUNDLE",
    help="Path to a PEM-encoded CA bundle for TLS verification.",
)
def version(base_url: str | None, api_key: str | None, timeout: int | None, ca_bundle: str | None) -> None:
    """Show package version and resolved configuration."""
    config = load_config(base_url=base_url, api_key=api_key, timeout=timeout, ca_bundle=ca_bundle)
    panel = Panel(
        f"Version: {__version__}\n"
        f"Base URL: {config.base_url}\n"
        f"API key present: {'yes' if config.has_api_key else 'no'}\n"
        f"CA bundle: {config.ca_bundle or 'default'}",
        title="proto-codegen",
    )
    console.print(panel)


@main.command()
@click.option("--base-url", envvar="OPENAI_BASE_URL", help="Override the API base URL.")
@click.option("--api-key", envvar="OPENAI_API_KEY", help="Override the API key.")
@click.option("--timeout", envvar="PROTO_CODEGEN_TIMEOUT", type=int, help="Request timeout in seconds.")
@click.option(
    "--ca-bundle",
    envvar="PROTO_CODEGEN_CA_BUNDLE",
    help="Path to a PEM-encoded CA bundle for TLS verification.",
)
@click.option(
    "--prefer-endpoint",
    type=click.Choice(["models", "probe", "auto"], case_sensitive=False),
    default="auto",
    show_default=True,
)
@click.option("--candidate", multiple=True, help="Candidate model ID to probe.")
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON.")
def models(
    base_url: str | None,
    api_key: str | None,
    timeout: int | None,
    ca_bundle: str | None,
    prefer_endpoint: str,
    candidate: Iterable[str],
    as_json: bool,
) -> None:
    """Discover available models."""
    config = load_config(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        candidates=tuple(candidate),
        ca_bundle=ca_bundle,
    )
    client = OpenAIClient(config.base_url, config.api_key, config.timeout, config.ca_bundle)

    try:
        with console.status("Querying models...", spinner="dots"):
            results = discover_models(client, prefer_endpoint=prefer_endpoint, candidates=config.candidates)
    except (ApiError, NetworkError) as exc:
        print_error_panel(str(exc))
        raise SystemExit(1) from exc

    if as_json:
        print_json(_results_to_json(results))
        return

    print_models_table(results)


def _results_to_json(results: Iterable[ModelResult]) -> list[dict[str, str]]:
    return [
        {
            "model_id": result.model_id,
            "discovery_method": result.discovery_method,
            "status": result.status,
            "details": result.details,
        }
        for result in results
    ]
