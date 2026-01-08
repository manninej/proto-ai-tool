from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from saga_code.cli import main
from saga_code.prompting import PromptManager


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_layer(root: Path, layer: str) -> Path:
    layer_root = root / "prompts" / layer
    layer_root.mkdir(parents=True, exist_ok=True)
    return layer_root


def test_list_layers(tmp_path: Path) -> None:
    _make_layer(tmp_path, "default")
    _make_layer(tmp_path, "fiat")
    manager = PromptManager(tmp_path)

    assert manager.list_layers() == ["default", "fiat"]


def test_active_stack_fallback(tmp_path: Path) -> None:
    _make_layer(tmp_path, "default")
    manager = PromptManager(tmp_path)

    assert manager.read_active_stack() == ["default"]


def test_stack_prepend_append_and_override(tmp_path: Path) -> None:
    layer_root = _make_layer(tmp_path, "default")
    _write(layer_root / "bundle" / "system.prepend.j2", "pre1-")
    _write(layer_root / "bundle" / "system.j2", "base-")
    _write(layer_root / "bundle" / "system.append.j2", "app1-")

    layer_root = _make_layer(tmp_path, "fiat")
    _write(layer_root / "bundle" / "system.prepend.j2", "pre2-")
    _write(layer_root / "bundle" / "system.j2", "override-")
    _write(layer_root / "bundle" / "system.append.j2", "app2-")

    manager = PromptManager(tmp_path)
    text, sources = manager.compose_text(["default", "fiat"], "bundle", "system")

    assert sources.body_path.name == "system.j2"
    assert text == "pre1-pre2-override-app1-app2-"


def test_variables_override(tmp_path: Path) -> None:
    layer_root = _make_layer(tmp_path, "default")
    _write(
        layer_root / "variables.yaml",
        "abbrev:\n  DUT: Device Under Test\ncoding_style:\n  cpp: C++17\n",
    )
    _write(layer_root / "bundle" / "system.j2", "{{ abbrev.DUT }} {{ coding_style.cpp }}")

    layer_root = _make_layer(tmp_path, "fiat")
    _write(layer_root / "variables.yaml", "coding_style:\n  cpp: C++20\n")
    _write(layer_root / "bundle" / "system.j2", "{{ abbrev.DUT }} {{ coding_style.cpp }}")

    manager = PromptManager(tmp_path)
    rendered = manager.render(["default", "fiat"], "bundle", "system")

    assert rendered == "Device Under Test C++20"


def test_prompts_cli_lists_layers(monkeypatch: object, tmp_path: Path) -> None:
    _make_layer(tmp_path, "default")
    _make_layer(tmp_path, "fiat")
    _write(tmp_path / "prompts" / "active_stack.txt", "default")
    _write(tmp_path / "prompts" / "default" / "bundle" / "system.j2", "")
    _write(tmp_path / "prompts" / "default" / "bundle" / "user.j2", "")

    monkeypatch.setattr("saga_code.cli._repo_root", lambda: tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["prompts"], env={"RICH_DISABLE": "1"})

    assert result.exit_code == 0, result.output
    assert "default" in result.output
    assert "fiat" in result.output
    assert "Active stack" in result.output


def test_prompts_cli_set_stack(monkeypatch: object, tmp_path: Path) -> None:
    _make_layer(tmp_path, "default")
    _make_layer(tmp_path, "fiat")

    monkeypatch.setattr("saga_code.cli._repo_root", lambda: tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["prompts", "default,fiat"], env={"RICH_DISABLE": "1"})

    assert result.exit_code == 0, result.output
    assert (tmp_path / "prompts" / "active_stack.txt").read_text(encoding="utf-8") == "default,fiat"


def test_prompts_cli_invalid_layer(monkeypatch: object, tmp_path: Path) -> None:
    _make_layer(tmp_path, "default")

    monkeypatch.setattr("saga_code.cli._repo_root", lambda: tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["prompts", "default,missing"], env={"RICH_DISABLE": "1"})

    assert result.exit_code == 1
    assert "Prompt layers not found" in result.output
