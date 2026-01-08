from __future__ import annotations

from collections.abc import Callable
import json

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from proto_codegen.openai_client import OpenAIClient


def run_chat_loop(
    client: OpenAIClient,
    console: Console,
    model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    no_history: bool,
    json_output: bool,
    show_reasoning: bool,
    raw_response: bool,
    input_provider: Callable[[], str] | None = None,
) -> None:
    history: list[dict[str, str]] = []
    system_message = _build_system_prompt(system_prompt)
    history.append({"role": "system", "content": system_message})

    def _prompt() -> str:
        if input_provider is not None:
            return input_provider()
        return Prompt.ask("You> ")

    while True:
        user_input = _prompt()
        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped in {"/quit", "/exit"}:
            break

        if no_history:
            messages = history[:1]
        else:
            messages = history[:]
        messages.append({"role": "user", "content": user_input})

        with console.status("Waiting for response...", spinner="dots"):
            response = client.chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        if raw_response or json_output:
            console.print_json(json.dumps(response["raw"], sort_keys=True))
        else:
            assistant_text = response.get("content", "")
            model_id = response.get("model", model)
            reasoning_text = response.get("reasoning_content", "")
            if not assistant_text:
                console.print(
                    Panel(
                        "Model did not produce a final answer; showing reasoning only.",
                        title="Warning",
                        style="yellow",
                    )
                )
            if show_reasoning and isinstance(reasoning_text, str) and reasoning_text:
                console.print(
                    Panel(
                        Markdown(reasoning_text),
                        title="Assistant Reasoning (debug)",
                        style="dim",
                    )
                )
            if assistant_text:
                panel = Panel(
                    Markdown(f"**Model:** {model_id}\n\n{assistant_text}"),
                    title="Assistant",
                )
                console.print(panel)

        if not no_history:
            history.append({"role": "user", "content": user_input})
            assistant_value = response.get("content") or response.get("reasoning_content", "")
            history.append({"role": "assistant", "content": assistant_value})


def _build_system_prompt(system_prompt: str) -> str:
    suffix = "Always produce a final answer in addition to any internal reasoning."
    if system_prompt.strip():
        return f"{system_prompt.strip()}\n{suffix}"
    return suffix
