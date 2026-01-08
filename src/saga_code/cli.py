from __future__ import annotations

import os
from typing import Iterable

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from saga_code import __version__
from saga_code.chat import run_chat_loop
from saga_code.config import Config, load_config
from saga_code.explain_cpp import (
    CPP_EXTENSIONS,
    SkipInfo,
    build_system_prompt,
    build_user_prompt,
    collect_source_files,
    discover_source_files,
    parse_json_response,
    parse_sections,
    read_files_with_budget,
    render_explanation,
    render_warning,
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
        f"CA bundle: {config.ca_bundle or 'default'}",
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


@main.command("explain-cpp")
@click.argument("paths", nargs=-1)
@click.option("--base-url", envvar="OPENAI_BASE_URL", help="Override the API base URL.")
@click.option("--api-key", envvar="OPENAI_API_KEY", help="Override the API key.")
@click.option("--timeout", envvar="SAGA_CODE_TIMEOUT", type=int, help="Request timeout in seconds.")
@click.option(
    "--ca-bundle",
    envvar="SAGA_CODE_CA_BUNDLE",
    help="Path to a PEM-encoded CA bundle for TLS verification.",
)
@click.option("--model", "model_override", help="Model ID to use for analysis.")
@click.option("--system", "system_prompt", default="", help="Optional system prompt override.")
@click.option("--max-files", type=int, default=20, show_default=True)
@click.option("--max-bytes", type=int, default=200000, show_default=True)
@click.option("--max-tokens", type=int, default=1500, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON only.")
@click.option("--show-reasoning/--no-show-reasoning", default=False, show_default=True)
def explain_cpp(
    paths: tuple[str, ...],
    base_url: str | None,
    api_key: str | None,
    timeout: int | None,
    ca_bundle: str | None,
    model_override: str | None,
    system_prompt: str,
    max_files: int,
    max_bytes: int,
    max_tokens: int,
    as_json: bool,
    show_reasoning: bool,
) -> None:
    """Explain C++ source code files or directories."""
    if not paths:
        print_error_panel("No input paths provided.")
        raise SystemExit(1)

    config = load_config(base_url=base_url, api_key=api_key, timeout=timeout, ca_bundle=ca_bundle)
    client = OpenAIClient(config.base_url, config.api_key, config.timeout, config.ca_bundle)
    model_id = model_override or _resolve_default_model(client, config)

    all_files = discover_source_files(list(paths), CPP_EXTENSIONS)
    if not all_files:
        print_error_panel("No C/C++ source files found.")
        raise SystemExit(1)

    limited_files = collect_source_files(list(paths), CPP_EXTENSIONS, max_files)
    skipped: list[SkipInfo] = []
    if len(all_files) > max_files:
        for path in all_files[max_files:]:
            skipped.append(SkipInfo(path=os.path.relpath(path, os.getcwd()), reason="max-files"))

    file_blobs, budget_skipped = read_files_with_budget(limited_files, max_bytes)
    skipped.extend(budget_skipped)

    if not file_blobs:
        print_error_panel("No files remaining after applying limits.")
        raise SystemExit(1)

    system_message = build_system_prompt(system_prompt)
    user_message = build_user_prompt(file_blobs, as_json)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    warning_console = Console(stderr=True) if as_json else console
    render_warning(warning_console, skipped)

    try:
        with console.status("Waiting for response...", spinner="dots"):
            response, final_text = _call_explain_model(
                client=client,
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                json_mode=as_json,
            )
    except (ApiError, NetworkError) as exc:
        print_error_panel(str(exc))
        raise SystemExit(1) from exc

    if not final_text:
        content_present = bool(response.get("content"))
        reasoning_present = bool(response.get("reasoning_content"))
        print_error_panel(
            "Model returned no usable final content after retry. "
            f"content present: {content_present}, reasoning_content present: {reasoning_present}."
        )
        raise SystemExit(1)

    reasoning_text = response.get("reasoning_content", "")
    if show_reasoning and not as_json and isinstance(reasoning_text, str) and reasoning_text:
        console.print(Panel(Markdown(reasoning_text), title="Assistant Reasoning (debug)", style="dim"))

    if as_json:
        payload = parse_json_response(final_text)
        if payload is None:
            print_error_panel("Failed to parse JSON response after retry.")
            raise SystemExit(1)
        print_json(payload)
        return

    sections = parse_sections(final_text)
    render_explanation(console, sections)


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


def _call_explain_model(
    client: OpenAIClient,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    json_mode: bool,
) -> tuple[dict[str, object], str]:
    last_response: dict[str, object] = {}
    for attempt in range(2):
        try:
            response = client.chat_completion(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                top_p=1.0,
            )
        except ApiError as exc:
            if exc.message == "Model returned no usable output" and attempt == 0:
                messages.append(
                    {
                        "role": "user",
                        "content": "You did not provide a final answer. "
                        "Reply again with FINAL: followed by the requested output only.",
                    }
                )
                continue
            raise

        last_response = response
        response_text = _select_final_text(response)
        if not response_text:
            if attempt == 0:
                messages.append(
                    {
                        "role": "user",
                        "content": "You did not provide a final answer. "
                        "Reply again with FINAL: followed by the requested output only.",
                    }
                )
                continue
            return response, ""

        if json_mode:
            if not response_text.strip().startswith("FINAL:"):
                if attempt == 0:
                    messages.append(
                        {
                            "role": "user",
                            "content": "Your response did not include the required FINAL: prefix. "
                            "Reply again with FINAL: followed immediately by the JSON object only.",
                        }
                    )
                    continue
                return response, response_text
            if parse_json_response(response_text) is None:
                if attempt == 0:
                    messages.append(
                        {
                            "role": "user",
                            "content": "Your response was not valid JSON. "
                            "Reply again with FINAL: followed immediately by the JSON object only.",
                        }
                    )
                    continue
                return response, response_text

        return response, response_text

    return last_response, ""


def _select_final_text(response: dict[str, object]) -> str:
    content = response.get("content")
    if isinstance(content, str) and content:
        return content
    reasoning = response.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        if "FINAL:" in reasoning:
            return "FINAL:" + reasoning.split("FINAL:")[-1].lstrip()
        return reasoning
    return ""


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
