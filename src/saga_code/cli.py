from __future__ import annotations

import json
from typing import Callable, Iterable

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from saga_code import __version__
from saga_code.chat import CommandResult, run_chat_loop
from saga_code.config import (
    Config,
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    PersistentConfig,
    config_path,
    load_config,
    load_persistent_config,
    save_persistent_config,
)
from saga_code.model_discovery import ModelResult, discover_models
from saga_code.openai_client import ApiError, NetworkError, OpenAIClient
from saga_code.render import print_error_panel, print_json, print_models_table

console = Console()


@click.group()
def main() -> None:
    """SAGA Code CLI."""


@main.command()
@click.option("--base-url", envvar="OPENAI_BASE_URL", help="Override the API base URL.")
@click.option("--api-key", envvar="OPENAI_API_KEY", help="Override the API key.")
@click.option("--timeout", envvar="SAGA_CODE_TIMEOUT", type=int, help="Request timeout in seconds.")
@click.option(
    "--ca-bundle",
    envvar="SAGA_CODE_CA_BUNDLE",
    help="Path to a PEM-encoded CA bundle for TLS verification.",
)
def version(base_url: str | None, api_key: str | None, timeout: int | None, ca_bundle: str | None) -> None:
    """Show package version and resolved configuration."""
    config = load_config(base_url=base_url, api_key=api_key, timeout=timeout, ca_bundle=ca_bundle)
    panel = Panel(
        f"Version: {__version__}\n"
        f"Base URL: {config.base_url}\n"
        f"API key present: {'yes' if config.has_api_key else 'no'}\n"
        f"CA bundle: {config.ca_bundle or 'default'}\n"
        f"Config file: {config_path()}",
        title="SAGA Code",
    )
    console.print(panel)


@main.command()
@click.option("--base-url", envvar="OPENAI_BASE_URL", help="Override the API base URL.")
@click.option("--api-key", envvar="OPENAI_API_KEY", help="Override the API key.")
@click.option("--timeout", envvar="SAGA_CODE_TIMEOUT", type=int, help="Request timeout in seconds.")
@click.option(
    "--ca-bundle",
    envvar="SAGA_CODE_CA_BUNDLE",
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
@click.option("--timeout", envvar="SAGA_CODE_TIMEOUT", type=int, help="Request timeout in seconds.")
@click.option(
    "--ca-bundle",
    envvar="SAGA_CODE_CA_BUNDLE",
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
    persistent = _ensure_persistent_config()
    config = load_config(base_url=base_url, api_key=api_key, timeout=timeout, ca_bundle=ca_bundle)
    client = OpenAIClient(
        config.base_url,
        config.api_key,
        config.timeout,
        config.ca_bundle,
        debug_http=debug_http,
        debug_sink=console.print,
    )

    model_id = model_override or config.default_model or _resolve_default_model(client, config)

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
            command_handler=_build_command_handler(
                client=client,
                config=config,
                persistent=persistent,
                model=model_id,
                max_tokens=resolved_max_tokens,
            ),
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


def _ensure_persistent_config() -> PersistentConfig:
    persistent = load_persistent_config()
    if persistent and persistent.base_url and persistent.api_key and persistent.model:
        return persistent

    base_url_default = persistent.base_url if persistent and persistent.base_url else DEFAULT_BASE_URL
    base_url = Prompt.ask("Server URL", default=base_url_default).strip() or base_url_default

    token_prompt = "Access token"
    api_key_default = persistent.api_key if persistent and persistent.api_key else None
    api_key = ""
    while not api_key:
        api_key = Prompt.ask(token_prompt, password=True, default=api_key_default or "").strip()

    ca_bundle_default = persistent.ca_bundle if persistent and persistent.ca_bundle else ""
    ca_bundle_input = Prompt.ask("PEM bundle file location (optional)", default=ca_bundle_default)
    ca_bundle = ca_bundle_input.strip() or None

    temp_config = load_config(base_url=base_url, api_key=api_key, timeout=DEFAULT_TIMEOUT, ca_bundle=ca_bundle)
    temp_client = OpenAIClient(
        temp_config.base_url,
        temp_config.api_key,
        temp_config.timeout,
        temp_config.ca_bundle,
        debug_http=False,
        debug_sink=console.print,
    )
    model_id = _prompt_for_model(temp_client, temp_config)

    persistent = PersistentConfig(
        base_url=base_url,
        api_key=api_key,
        ca_bundle=ca_bundle,
        model=model_id,
    )
    save_persistent_config(persistent)
    return persistent


def _prompt_for_model(client: OpenAIClient, config: Config) -> str:
    try:
        with console.status("Discovering models...", spinner="dots"):
            results = discover_models(client, prefer_endpoint="auto", candidates=config.candidates)
    except (ApiError, NetworkError) as exc:
        print_error_panel(str(exc))
        return "gpt-oss-120b"

    if not results:
        console.print("No models discovered; falling back to gpt-oss-120b.")
        return "gpt-oss-120b"

    print_models_table(results)
    available = [result.model_id for result in results if result.status == "available"]
    choices = available or [result.model_id for result in results]
    default_choice = choices[0] if choices else "gpt-oss-120b"
    return Prompt.ask("Select model", choices=choices, default=default_choice)


def _build_command_handler(
    client: OpenAIClient,
    config: Config,
    persistent: PersistentConfig,
    model: str,
    max_tokens: int,
) -> Callable[[str], CommandResult]:
    active_client = client
    active_model = model
    active_max_tokens = max_tokens
    active_config = config
    active_persistent = persistent

    def _save_persistent() -> None:
        save_persistent_config(active_persistent)

    def _update_client() -> None:
        nonlocal active_client
        active_client = OpenAIClient(
            active_config.base_url,
            active_config.api_key,
            active_config.timeout,
            active_config.ca_bundle,
            debug_http=False,
            debug_sink=console.print,
        )

    def _refresh_config() -> None:
        nonlocal active_config
        active_config = load_config(
            base_url=active_persistent.base_url,
            api_key=active_persistent.api_key,
            timeout=active_config.timeout,
            ca_bundle=active_persistent.ca_bundle,
        )

    def _handle_server(parts: list[str]) -> CommandResult:
        nonlocal active_persistent, active_model, active_max_tokens
        if len(parts) >= 2:
            base_url = parts[1]
            ca_bundle = parts[2] if len(parts) > 2 else active_persistent.ca_bundle
        else:
            base_url = Prompt.ask("Server URL", default=active_persistent.base_url or active_config.base_url)
            ca_bundle_input = Prompt.ask(
                "PEM bundle file location (optional)", default=active_persistent.ca_bundle or ""
            )
            ca_bundle = ca_bundle_input.strip() or None
        active_persistent = PersistentConfig(
            base_url=base_url,
            api_key=active_persistent.api_key,
            ca_bundle=ca_bundle,
            model=active_persistent.model,
        )
        _save_persistent()
        _refresh_config()
        _update_client()
        console.print("Updated server configuration.")
        return CommandResult(handled=True, client=active_client, model=active_model, max_tokens=active_max_tokens)

    def _handle_token(parts: list[str]) -> CommandResult:
        nonlocal active_persistent
        token = parts[1] if len(parts) > 1 else ""
        while not token:
            token = Prompt.ask("Access token", password=True).strip()
        active_persistent = PersistentConfig(
            base_url=active_persistent.base_url,
            api_key=token,
            ca_bundle=active_persistent.ca_bundle,
            model=active_persistent.model,
        )
        _save_persistent()
        _refresh_config()
        _update_client()
        console.print("Updated access token.")
        return CommandResult(handled=True, client=active_client, model=active_model, max_tokens=active_max_tokens)

    def _handle_model(parts: list[str]) -> CommandResult:
        nonlocal active_persistent, active_model, active_max_tokens
        if len(parts) > 1:
            model_id = parts[1]
        else:
            model_id = _prompt_for_model(active_client, active_config)
        active_persistent = PersistentConfig(
            base_url=active_persistent.base_url,
            api_key=active_persistent.api_key,
            ca_bundle=active_persistent.ca_bundle,
            model=model_id,
        )
        _save_persistent()
        active_model = model_id
        active_max_tokens = _resolve_max_tokens(active_client, model_id)
        console.print(f"Using model {model_id}.")
        return CommandResult(handled=True, client=active_client, model=active_model, max_tokens=active_max_tokens)

    def handler(command: str) -> CommandResult:
        parts = command.split()
        if parts[0] == "/server":
            return _handle_server(parts)
        if parts[0] == "/token":
            return _handle_token(parts)
        if parts[0] == "/model":
            return _handle_model(parts)
        return CommandResult(handled=False)

    return handler
