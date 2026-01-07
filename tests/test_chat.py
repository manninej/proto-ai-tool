from __future__ import annotations

import respx
from click.testing import CliRunner

from proto_codegen.cli import main
from proto_codegen.openai_client import OpenAIClient


def test_chat_quit_exits(monkeypatch: object) -> None:
    def fail_chat_completion(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("chat_completion should not be called")

    monkeypatch.setattr(OpenAIClient, "chat_completion", fail_chat_completion)
    runner = CliRunner()
    result = runner.invoke(main, ["chat", "--model", "test-model"], input="/quit\n", env={"RICH_DISABLE": "1"})

    assert result.exit_code == 0


def test_chat_completion_payload_shape() -> None:
    with respx.mock(base_url="https://api.example.com") as router:
        route = router.post("/v1/chat/completions").respond(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "model": "model-x"},
        )
        client = OpenAIClient("https://api.example.com", api_key="test", timeout=5)
        response = client.chat_completion(
            model="model-x",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.5,
            max_tokens=10,
        )

    assert response["assistant_text"] == "ok"
    assert route.called
    payload = route.calls[0].request.json()
    assert payload == {
        "model": "model-x",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.5,
        "max_tokens": 10,
    }


def test_chat_history_on(monkeypatch: object) -> None:
    calls: list[list[dict[str, str]]] = []

    def fake_chat_completion(
        self: OpenAIClient,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, object]:
        calls.append(messages)
        return {
            "assistant_text": "ok",
            "model": model,
            "data": {"choices": [{"message": {"content": "ok"}}], "model": model},
        }

    monkeypatch.setattr(OpenAIClient, "chat_completion", fake_chat_completion)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["chat", "--model", "test-model"],
        input="hello\nthere\n/quit\n",
        env={"RICH_DISABLE": "1"},
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        [{"role": "user", "content": "hello"}],
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "there"},
        ],
    ]


def test_chat_history_off(monkeypatch: object) -> None:
    calls: list[list[dict[str, str]]] = []

    def fake_chat_completion(
        self: OpenAIClient,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, object]:
        calls.append(messages)
        return {
            "assistant_text": "ok",
            "model": model,
            "data": {"choices": [{"message": {"content": "ok"}}], "model": model},
        }

    monkeypatch.setattr(OpenAIClient, "chat_completion", fake_chat_completion)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["chat", "--model", "test-model", "--system", "You are a helper.", "--no-history"],
        input="hello\nthere\n/quit\n",
        env={"RICH_DISABLE": "1"},
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        [{"role": "system", "content": "You are a helper."}, {"role": "user", "content": "hello"}],
        [{"role": "system", "content": "You are a helper."}, {"role": "user", "content": "there"}],
    ]
