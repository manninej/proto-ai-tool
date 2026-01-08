from __future__ import annotations

import json
from typing import Iterable

import click
from rich.console import Console
from rich.panel import Panel

from proto_codegen import __version__
from proto_codegen.chat import run_chat_loop
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
@click.option("--debug-http", is_flag=True, help="Print HTTP debug information.")
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
    debug_http: bool,
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
    client = OpenAIClient(
        config.base_url,
        config.api_key,
        config.timeout,
        config.ca_bundle,
        debug_http=debug_http,
        debug_sink=console.print,
    )

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


@main.command()
@click.option("--base-url", envvar="OPENAI_BASE_URL", help="Override the API base URL.")
@click.option("--api-key", envvar="OPENAI_API_KEY", help="Override the API key.")
@click.option("--timeout", envvar="PROTO_CODEGEN_TIMEOUT", type=int, help="Request timeout in seconds.")
@click.option(
    "--ca-bundle",
    envvar="PROTO_CODEGEN_CA_BUNDLE",
    help="Path to a PEM-encoded CA bundle for TLS verification.",
)
@click.option("--debug-http", is_flag=True, help="Print HTTP debug information.")
@click.option("--model", "model_override", help="Model ID to use for chat.")
@click.option("--system", "system_prompt", default="", help="Optional system prompt.")
@click.option("--temperature", type=float, default=0.0, show_default=True)
@click.option("--max-tokens", type=int, default=None)
@click.option("--no-history", is_flag=True, help="Disable conversation history.")
@click.option("--json", "as_json", is_flag=True, help="Print raw JSON responses.")
@click.option("--show-reasoning/--no-show-reasoning", default=False, show_default=True)
@click.option("--raw-response", is_flag=True, help="Print raw response JSON instead of formatted output.")
def chat(
    base_url: str | None,
    api_key: str | None,
    timeout: int | None,
    ca_bundle: str | None,
    debug_http: bool,
    model_override: str | None,
    system_prompt: str,
    temperature: float,
    max_tokens: int | None,
    no_history: bool,
    as_json: bool,
    show_reasoning: bool,
    raw_response: bool,
) -> None:
    """Start an interactive chat session."""
    config = load_config(base_url=base_url, api_key=api_key, timeout=timeout, ca_bundle=ca_bundle)
    client = OpenAIClient(
        config.base_url,
        config.api_key,
        config.timeout,
        config.ca_bundle,
        debug_http=debug_http,
        debug_sink=console.print,
    )

    model_id = model_override or _resolve_default_model(client, config)

    resolved_max_tokens = max_tokens or _resolve_max_tokens(client, model_id)
    try:
        run_chat_loop(
            client=client,
            console=console,
            model=model_id,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=resolved_max_tokens,
            no_history=no_history,
            json_output=as_json,
            show_reasoning=show_reasoning,
            raw_response=raw_response,
        )
    except KeyboardInterrupt:
        console.print("Exiting chat.")


def _resolve_default_model(client: OpenAIClient, config: Config) -> str:
    try:
        with console.status("Discovering models...", spinner="dots"):
            results = discover_models(client, prefer_endpoint="auto", candidates=config.candidates)
    except (ApiError, NetworkError):
        return "gpt-oss-120b"

    for result in results:
        if result.status == "available":
            return result.model_id
    return "gpt-oss-120b"


def _resolve_max_tokens(client: OpenAIClient, model_id: str) -> int:
    try:
        model_info = client.get_model_info(model_id)
    except (ApiError, NetworkError):
        return 2048
    if isinstance(model_info, dict):
        for key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
            value = model_info.get(key)
            if isinstance(value, int) and value > 0:
                return value
    return 2048


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
