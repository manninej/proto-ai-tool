from __future__ import annotations


import json
from typing import Callable, Iterable
import os
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from jinja2 import Environment

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
from saga_code.explain_cpp import (
    CPP_EXTENSIONS,
    SkipInfo,
    build_files_block,
    collect_source_files,
    discover_source_files,
    parse_json_response,
    parse_sections,
    read_files_with_budget,
    render_explanation,
    render_warning,
    strip_final_prefix,

)
from saga_code.model_discovery import ModelResult, discover_models
from saga_code.openai_client import ApiError, NetworkError, OpenAIClient
from saga_code.prompting import PromptError, PromptManager
from saga_code.render import print_error_panel, print_json, print_models_table

console = Console()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _prompt_manager() -> PromptManager:
    return PromptManager(_repo_root())


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


@main.command("prompts")
@click.argument("stack", required=False)
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON.")
@click.option("--show-resolved", "show_resolved", help="Show composed template for BUNDLE/ROLE.")
@click.option("--render", "render_bundle", help="Render BUNDLE/ROLE with merged variables.")
@click.option("--validate", "validate_stack", is_flag=True, help="Validate the active prompt stack.")
def prompts(
    stack: str | None,
    as_json: bool,
    show_resolved: str | None,
    render_bundle: str | None,
    validate_stack: bool,
) -> None:
    """List or configure prompt layers."""
    prompt_manager = _prompt_manager()
    if stack and (show_resolved or render_bundle or validate_stack):
        print_error_panel("Cannot combine stack updates with --show-resolved, --render, or --validate.")
        raise SystemExit(1)

    if stack:
        stack_list = _parse_stack_arg(stack)
        try:
            prompt_manager.write_active_stack(stack_list)
        except PromptError as exc:
            print_error_panel(str(exc))
            raise SystemExit(1) from exc
        console.print(f"Active prompt stack set to: {','.join(stack_list)}")
        return

    if validate_stack:
        try:
            _validate_prompt_stack(prompt_manager)
        except PromptError as exc:
            print_error_panel(str(exc))
            raise SystemExit(1) from exc
        console.print("Prompt stack validation passed.")
        return

    if show_resolved:
        bundle, role = _parse_bundle_role(show_resolved)
        try:
            stack_list = prompt_manager.read_active_stack()
            template_text, sources = prompt_manager.compose_text(stack_list, bundle, role)
        except PromptError as exc:
            print_error_panel(str(exc))
            raise SystemExit(1) from exc
        console.print("Resolved prompt stack:")
        console.print(f"  Stack: {', '.join(sources.stack)}")
        console.print(f"  Body: {sources.body_path}")
        if sources.prepend_paths:
            console.print("  Prepends:")
            for path in sources.prepend_paths:
                console.print(f"    - {path}")
        if sources.append_paths:
            console.print("  Appends:")
            for path in sources.append_paths:
                console.print(f"    - {path}")
        console.print("\nComposed template:\n")
        console.print(template_text)
        return

    if render_bundle:
        bundle, role = _parse_bundle_role(render_bundle)
        try:
            stack_list = prompt_manager.read_active_stack()
            rendered = prompt_manager.render(stack_list, bundle, role)
        except PromptError as exc:
            print_error_panel(str(exc))
            raise SystemExit(1) from exc
        console.print(rendered)
        return

    try:
        layers = prompt_manager.list_layers()
        active_path = _repo_root() / "prompts" / "active_stack.txt"
        if active_path.exists():
            active_stack = prompt_manager.read_active_stack()
            active_label = "Active stack"
        else:
            active_stack = prompt_manager.read_active_stack()
            active_label = "Recommended stack"
        bundles = prompt_manager.list_bundles(layers)
    except PromptError as exc:
        print_error_panel(str(exc))
        raise SystemExit(1) from exc

    if as_json:
        payload = {
            "layers": layers,
            "stack": active_stack,
            "stack_label": active_label,
            "bundles": bundles,
        }
        print_json(payload)
        return

    console.print("Available prompt layers:")
    if layers:
        for layer in layers:
            console.print(f"  - {layer}")
    else:
        console.print("  (none)")
    console.print(f"{active_label}: {', '.join(active_stack)}")
    if bundles:
        console.print("Discovered prompt bundles:")
        for bundle in bundles:
            console.print(f"  - {bundle}")
    else:
        console.print("No prompt bundles found.")


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
    prompt_manager = _prompt_manager()
    try:
        stack = prompt_manager.read_active_stack()
        runtime_prepend = f"{system_prompt.strip()}\n" if system_prompt.strip() else None
        system_message = prompt_manager.render_with_prepend(
            stack,
            bundle="chat",
            role="system",
            runtime_prepend=runtime_prepend,
        )
    except PromptError as exc:
        print_error_panel(str(exc))
        raise SystemExit(1) from exc
    try:
        run_chat_loop(
            client=client,
            console=console,
            model=model_id,
            system_prompt=system_message,
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

    prompt_manager = _prompt_manager()
    try:
        stack = prompt_manager.read_active_stack()
        runtime_prepend = f"\n\n{system_prompt.strip()}" if system_prompt.strip() else None
        system_message = prompt_manager.render_with_prepend(
            stack,
            bundle="explain_cpp",
            role="system",
            runtime_prepend=runtime_prepend,
        )
        files_block = build_files_block(file_blobs)
        user_message = prompt_manager.render(
            stack,
            bundle="explain_cpp",
            role="user",
            extra_vars={"files_block": files_block, "json_mode": as_json},
        )
    except PromptError as exc:
        print_error_panel(str(exc))
        raise SystemExit(1) from exc
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

    markdown_text = strip_final_prefix(final_text).strip()
    if markdown_text:
        panel = Panel(Markdown(markdown_text), title="Explanation")
        console.print(panel)
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
    max_attempts: int = 3,
) -> tuple[dict[str, object], str]:
    last_response: dict[str, object] = {}
    for attempt in range(max_attempts):
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
                        "content": _retry_prompt(json_mode),
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
                        "content": _retry_prompt(json_mode),
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
        return ""
    return ""


def _retry_prompt(json_mode: bool) -> str:
    if json_mode:
        return "You did not provide a final answer. Reply again with FINAL: followed by the requested output only."
    return "You did not provide a final answer. Reply again with the requested Markdown output only."


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


def _parse_stack_arg(raw_stack: str) -> list[str]:
    parts = [part.strip() for part in raw_stack.split(",")]
    if not parts or any(not part for part in parts):
        raise PromptError("Prompt stack must be a comma-separated list of layer names.")
    for part in parts:
        if any(ch.isspace() for ch in part):
            raise PromptError("Prompt layer names must not contain whitespace.")
    return parts


def _parse_bundle_role(raw: str) -> tuple[str, str]:
    if "/" not in raw:
        raise PromptError("Expected bundle/role format.")
    bundle, role = raw.split("/", 1)
    if not bundle or not role:
        raise PromptError("Expected bundle/role format.")
    if role not in {"system", "user"}:
        raise PromptError("Role must be 'system' or 'user'.")
    return bundle, role


def _validate_prompt_stack(prompt_manager: PromptManager) -> None:
    stack = prompt_manager.read_active_stack()
    bundles = prompt_manager.list_bundles(stack)
    if not bundles:
        raise PromptError("No prompt bundles found in the active stack.")
    for bundle in bundles:
        for role in ("system", "user"):
            prompt_manager.resolve_sources(stack, bundle, role)
    prompt_manager.load_variables(stack)
    env = Environment(autoescape=False)
    prompts_root = _repo_root() / "prompts"
    for layer in stack:
        layer_root = prompts_root / layer
        for path in sorted(layer_root.rglob("*.j2")):
            source = path.read_text(encoding="utf-8")
            env.parse(source)


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

    available = [result.model_id for result in results if result.status == "available"]
    choices = available or [result.model_id for result in results]
    if not choices:
        return "gpt-oss-120b"
    console.print("Available models:")
    for index, model_id in enumerate(choices, start=1):
        console.print(f"  {index}. {model_id}")
    default_index = 1
    max_index = len(choices)
    selection = ""
    valid_choices = [str(i) for i in range(1, max_index + 1)]
    while selection not in valid_choices:
        selection = Prompt.ask(
            f"Select model [1-{max_index}]",
            default=str(default_index),
        ).strip()
    return choices[int(selection) - 1]


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
