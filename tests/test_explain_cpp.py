from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from saga_code.cli import main
from saga_code.openai_client import OpenAIClient


def test_explain_cpp_single_file_prompt_and_output(monkeypatch: object) -> None:
    calls: list[list[dict[str, str]]] = []

    def fake_chat_completion(
        self: OpenAIClient,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
    ) -> dict[str, object]:
        calls.append(messages)
        assert temperature == 0.0
        assert max_tokens == 1500
        assert top_p == 1.0
        user_prompt = messages[1]["content"]
        assert "ANALYSIS ONLY" in user_prompt
        assert "<file path=\"src/foo.cpp\">" in user_prompt
        assert "int main()" in user_prompt
        return {
            "content": (
                "Overview: Sample summary\n"
                "Key Components:\n- main\n"
                "Data Flow: none\n"
                "Assumptions: none\n"
                "Risks / Pitfalls: none\n"
                "Open Questions: none"
            ),
            "reasoning_content": "",
            "model": model,
            "raw": {"choices": [{"message": {"content": "ok"}}], "model": model},
        }

    monkeypatch.setattr(OpenAIClient, "chat_completion", fake_chat_completion)
    runner = CliRunner()
    with runner.isolated_filesystem():
        source_path = Path("src/foo.cpp")
        source_path.parent.mkdir(parents=True)
        source_path.write_text("int main() { return 0; }", encoding="utf-8")
        result = runner.invoke(
            main,
            ["explain-cpp", str(source_path), "--model", "test-model"],
            env={"RICH_DISABLE": "1"},
        )

    assert result.exit_code == 0, result.output
    assert "Overview" in result.output
    assert "Key Components" in result.output
    assert calls


def test_explain_cpp_directory_order(monkeypatch: object) -> None:
    captured: list[str] = []

    def fake_chat_completion(
        self: OpenAIClient,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
    ) -> dict[str, object]:
        captured.append(messages[1]["content"])
        return {
            "content": "Overview: ok\nKey Components: ok\nData Flow: ok\nAssumptions: ok\nRisks / Pitfalls: ok\nOpen Questions: ok",
            "reasoning_content": "",
            "model": model,
            "raw": {"choices": [{"message": {"content": "ok"}}], "model": model},
        }

    monkeypatch.setattr(OpenAIClient, "chat_completion", fake_chat_completion)
    runner = CliRunner()
    with runner.isolated_filesystem():
        root = Path("code")
        root.mkdir()
        (root / "a.cpp").write_text("// a", encoding="utf-8")
        (root / "b.hpp").write_text("// b", encoding="utf-8")
        (root / "ignore.txt").write_text("no", encoding="utf-8")
        result = runner.invoke(
            main,
            ["explain-cpp", str(root), "--model", "test-model"],
            env={"RICH_DISABLE": "1"},
        )

    assert result.exit_code == 0, result.output
    assert captured
    prompt = captured[0]
    assert prompt.index("<file path=\"code/a.cpp\">") < prompt.index("<file path=\"code/b.hpp\">")
    assert "ignore.txt" not in prompt


def test_explain_cpp_json_output(monkeypatch: object) -> None:
    payload = {
        "overview": "summary",
        "components": [{"name": "main", "responsibility": "entry"}],
        "data_flow": "none",
        "assumptions": ["none"],
        "risks": ["none"],
        "open_questions": ["none"],
    }

    def fake_chat_completion(
        self: OpenAIClient,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
    ) -> dict[str, object]:
        return {
            "content": f"FINAL: {json.dumps(payload)}",
            "reasoning_content": "",
            "model": model,
            "raw": {"choices": [{"message": {"content": "ok"}}], "model": model},
        }

    monkeypatch.setattr(OpenAIClient, "chat_completion", fake_chat_completion)
    runner = CliRunner(mix_stderr=False)
    with runner.isolated_filesystem():
        source_path = Path("main.cpp")
        source_path.write_text("// main", encoding="utf-8")
        result = runner.invoke(
            main,
            ["explain-cpp", str(source_path), "--model", "test-model", "--json"],
            env={"RICH_DISABLE": "1"},
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == payload


def test_explain_cpp_limits_warn(monkeypatch: object) -> None:

    def fake_chat_completion(
        self: OpenAIClient,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
    ) -> dict[str, object]:
        return {
            "content": "Overview: ok\nKey Components: ok\nData Flow: ok\nAssumptions: ok\nRisks / Pitfalls: ok\nOpen Questions: ok",
            "reasoning_content": "",
            "model": model,
            "raw": {"choices": [{"message": {"content": "ok"}}], "model": model},
        }

    monkeypatch.setattr(OpenAIClient, "chat_completion", fake_chat_completion)
    runner = CliRunner(mix_stderr=False)
    with runner.isolated_filesystem():
        root = Path(".")
        (root / "a.cpp").write_text("a", encoding="utf-8")
        (root / "b.cpp").write_text("b" * 10, encoding="utf-8")
        (root / "c.cpp").write_text("c", encoding="utf-8")
        result = runner.invoke(
            main,
            [
                "explain-cpp",
                str(root),
                "--model",
                "test-model",
                "--max-files",
                "2",
                "--max-bytes",
                "2",
            ],
            env={"RICH_DISABLE": "1"},
        )

        assert result.exit_code == 0, result.output
        assert "max-files" in result.output
        assert "max-bytes" in result.output


def test_explain_cpp_json_retry(monkeypatch: object) -> None:
    payload = {
        "overview": "summary",
        "components": [{"name": "main", "responsibility": "entry"}],
        "data_flow": "none",
        "assumptions": ["none"],
        "risks": ["none"],
        "open_questions": ["none"],
    }
    calls: list[list[dict[str, str]]] = []

    def fake_chat_completion(
        self: OpenAIClient,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
    ) -> dict[str, object]:
        calls.append(messages)
        if len(calls) == 1:
            return {
                "content": "FINAL: not-json",
                "reasoning_content": "",
                "model": model,
                "raw": {"choices": [{"message": {"content": "bad"}}], "model": model},
            }
        return {
            "content": f"FINAL: {json.dumps(payload)}",
            "reasoning_content": "",
            "model": model,
            "raw": {"choices": [{"message": {"content": "ok"}}], "model": model},
        }

    monkeypatch.setattr(OpenAIClient, "chat_completion", fake_chat_completion)
    runner = CliRunner(mix_stderr=False)
    with runner.isolated_filesystem():
        source_path = Path("main.cpp")
        source_path.write_text("// main", encoding="utf-8")
        result = runner.invoke(
            main,
            ["explain-cpp", str(source_path), "--model", "test-model", "--json"],
            env={"RICH_DISABLE": "1"},
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == payload
    assert len(calls) == 2
    assert "valid JSON" in calls[1][-1]["content"]


def test_explain_cpp_reasoning_with_final(monkeypatch: object) -> None:
    def fake_chat_completion(
        self: OpenAIClient,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
    ) -> dict[str, object]:
        return {
            "content": "",
            "reasoning_content": (
                "Some analysis.\n"
                "FINAL: Overview: ok\n"
                "Key Components: ok\n"
                "Data Flow: ok\n"
                "Assumptions: ok\n"
                "Risks / Pitfalls: ok\n"
                "Open Questions: ok"
            ),
            "model": model,
            "raw": {"choices": [{"message": {"content": "", "reasoning_content": "ok"}}], "model": model},
        }

    monkeypatch.setattr(OpenAIClient, "chat_completion", fake_chat_completion)
    runner = CliRunner()
    with runner.isolated_filesystem():
        source_path = Path("main.cpp")
        source_path.write_text("// main", encoding="utf-8")
        result = runner.invoke(
            main,
            ["explain-cpp", str(source_path), "--model", "test-model"],
            env={"RICH_DISABLE": "1"},
        )

    assert result.exit_code == 0, result.output
    assert "Overview" in result.output


def test_explain_cpp_reasoning_without_final_retries(monkeypatch: object) -> None:
    calls: list[list[dict[str, str]]] = []

    def fake_chat_completion(
        self: OpenAIClient,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
    ) -> dict[str, object]:
        calls.append(messages)
        if len(calls) == 1:
            return {
                "content": "",
                "reasoning_content": "Some analysis without final.",
                "model": model,
                "raw": {"choices": [{"message": {"content": "", "reasoning_content": "ok"}}], "model": model},
            }
        return {
            "content": (
                "Overview: ok\n"
                "Key Components: ok\n"
                "Data Flow: ok\n"
                "Assumptions: ok\n"
                "Risks / Pitfalls: ok\n"
                "Open Questions: ok"
            ),
            "reasoning_content": "",
            "model": model,
            "raw": {"choices": [{"message": {"content": "ok"}}], "model": model},
        }

    monkeypatch.setattr(OpenAIClient, "chat_completion", fake_chat_completion)
    runner = CliRunner()
    with runner.isolated_filesystem():
        source_path = Path("main.cpp")
        source_path.write_text("// main", encoding="utf-8")
        result = runner.invoke(
            main,
            ["explain-cpp", str(source_path), "--model", "test-model"],
            env={"RICH_DISABLE": "1"},
        )

    assert result.exit_code == 0, result.output
    assert len(calls) == 2
    assert "final answer" in calls[1][-1]["content"]
