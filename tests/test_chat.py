from __future__ import annotations

import respx
from click.testing import CliRunner

from saga_code.cli import main
from saga_code.openai_client import ApiError, OpenAIClient


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

    assert response["content"] == "ok"
    assert route.called
    payload = route.calls[0].request.json()
    assert payload == {
        "model": "model-x",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.5,
        "max_tokens": 10,
        "top_p": 1.0,
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
            "content": "ok",
            "reasoning_content": "",
            "model": model,
            "raw": {"choices": [{"message": {"content": "ok"}}], "model": model},
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
        [
            {
                "role": "system",
                "content": "Always produce a final answer in addition to any internal reasoning.",
            },
            {"role": "user", "content": "hello"},
        ],
        [
            {
                "role": "system",
                "content": "Always produce a final answer in addition to any internal reasoning.",
            },
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
            "content": "ok",
            "reasoning_content": "",
            "model": model,
            "raw": {"choices": [{"message": {"content": "ok"}}], "model": model},
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
        [
            {
                "role": "system",
                "content": "You are a helper.\nAlways produce a final answer in addition to any internal reasoning.",
            },
            {"role": "user", "content": "hello"},
        ],
        [
            {
                "role": "system",
                "content": "You are a helper.\nAlways produce a final answer in addition to any internal reasoning.",
            },
            {"role": "user", "content": "there"},
        ],
    ]


def test_reasoning_content_with_final_is_used() -> None:
    with respx.mock(base_url="https://api.example.com") as router:
        router.post("/v1/chat/completions").respond(
            200,
            json={
                "choices": [{"message": {"content": "", "reasoning_content": "FINAL: ok"}}],
                "model": "model-x",
            },
        )
        client = OpenAIClient("https://api.example.com", api_key="test", timeout=5)
        response = client.chat_completion(
            model="model-x",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.0,
            max_tokens=10,
        )

    assert response["content"] == ""
    assert response["reasoning_content"] == "FINAL: ok"


def test_reasoning_content_without_final_errors() -> None:
    with respx.mock(base_url="https://api.example.com") as router:
        router.post("/v1/chat/completions").respond(
            200,
            json={
                "choices": [{"message": {"content": "", "reasoning_content": ""}}],
                "model": "model-x",
            },
        )
        client = OpenAIClient("https://api.example.com", api_key="test", timeout=5)
        try:
            client.chat_completion(
                model="model-x",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0.0,
                max_tokens=10,
            )
        except ApiError as exc:
            assert "Model returned no usable output" in exc.message
        else:
            raise AssertionError("Expected ApiError for missing output")


def test_mixed_content_prefers_content() -> None:
    with respx.mock(base_url="https://api.example.com") as router:
        router.post("/v1/chat/completions").respond(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "FINAL: content",
                            "reasoning_content": "FINAL: reasoning",
                        }
                    }
                ],
                "model": "model-x",
            },
        )
        client = OpenAIClient("https://api.example.com", api_key="test", timeout=5)
        response = client.chat_completion(
            model="model-x",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.0,
            max_tokens=10,
        )

    assert response["content"] == "FINAL: content"
    assert response["reasoning_content"] == "FINAL: reasoning"


def test_show_reasoning_panel(monkeypatch: object) -> None:
    def fake_chat_completion(
        self: OpenAIClient,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, object]:
        return {
            "content": "Answer",
            "reasoning_content": "Reasoning",
            "model": model,
            "raw": {"choices": [{"message": {"content": "Answer", "reasoning_content": "Reasoning"}}]},
        }

    monkeypatch.setattr(OpenAIClient, "chat_completion", fake_chat_completion)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["chat", "--model", "test-model", "--show-reasoning"],
        input="hello\n/quit\n",
        env={"RICH_DISABLE": "1"},
    )

    assert result.exit_code == 0, result.output
    assert "Assistant Reasoning (debug)" in result.output
    assert "Assistant" in result.output


def test_default_hides_reasoning(monkeypatch: object) -> None:
    def fake_chat_completion(
        self: OpenAIClient,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, object]:
        return {
            "content": "Answer",
            "reasoning_content": "Reasoning",
            "model": model,
            "raw": {"choices": [{"message": {"content": "Answer", "reasoning_content": "Reasoning"}}]},
        }

    monkeypatch.setattr(OpenAIClient, "chat_completion", fake_chat_completion)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["chat", "--model", "test-model"],
        input="hello\n/quit\n",
        env={"RICH_DISABLE": "1"},
    )

    assert result.exit_code == 0, result.output
    assert "Assistant Reasoning (debug)" not in result.output


def test_reasoning_only_shows_warning(monkeypatch: object) -> None:
    def fake_chat_completion(
        self: OpenAIClient,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, object]:
        return {
            "content": "",
            "reasoning_content": "Reasoning",
            "model": model,
            "raw": {"choices": [{"message": {"content": "", "reasoning_content": "Reasoning"}}]},
        }

    monkeypatch.setattr(OpenAIClient, "chat_completion", fake_chat_completion)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["chat", "--model", "test-model", "--show-reasoning"],
        input="hello\n/quit\n",
        env={"RICH_DISABLE": "1"},
    )

    assert result.exit_code == 0, result.output
    assert "Model did not produce a final answer; showing reasoning only" in result.output
