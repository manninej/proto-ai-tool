from __future__ import annotations

from collections.abc import Callable
import json

from rich.console import Console
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
    input_provider: Callable[[], str] | None = None,
) -> None:
    history: list[dict[str, str]] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

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
            messages = history[:] if system_prompt else []
        else:
            messages = history[:]
        messages.append({"role": "user", "content": user_input})

        response = _request_with_retry(
            client=client,
            console=console,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if json_output:
            console.print_json(json.dumps(response["data"], sort_keys=True))
        else:
            assistant_text = response.get("assistant_text", "")
            model_id = response.get("model", model)
            reasoning_text = response.get("reasoning_text", "")
            if show_reasoning and isinstance(reasoning_text, str) and reasoning_text:
                console.print(Panel(reasoning_text, title="Reasoning"))
            panel = Panel(f"Model: {model_id}\n\n{assistant_text}", title="Assistant")
            console.print(panel)

        if not no_history:
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response.get("assistant_text", "")})


def _request_with_retry(
    client: OpenAIClient,
    console: Console,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(2):
        with console.status("Waiting for response...", spinner="dots"):
            try:
                response = client.chat_completion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - surface API errors to user after retry
                last_error = exc
                if _should_retry(exc) and attempt == 0:
                    messages = messages + [
                        {
                            "role": "user",
                            "content": "You did not provide a FINAL answer. Reply again with FINAL: followed by the answer only.",
                        }
                    ]
                    continue
                raise

        assistant_text = response.get("assistant_text")
        if isinstance(assistant_text, str) and assistant_text.startswith("FINAL:"):
            return response
        if attempt == 0:
            messages = messages + [
                {
                    "role": "user",
                    "content": "You did not provide a FINAL answer. Reply again with FINAL: followed by the answer only.",
                }
            ]
            continue
        raise RuntimeError("Assistant response did not include a FINAL answer.") from last_error

    raise RuntimeError("Assistant response did not include a FINAL answer.") from last_error


def _should_retry(exc: Exception) -> bool:
    message = str(exc)
    return message.startswith("No final assistant content produced")
