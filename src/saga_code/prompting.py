from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import ChoiceLoader, Environment, FileSystemLoader
import yaml


class PromptError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedPromptSources:
    body_path: Path
    prepend_paths: list[Path]
    append_paths: list[Path]
    stack: list[str]


class PromptManager:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.prompts_root = repo_root / "prompts"

    def list_layers(self) -> list[str]:
        if not self.prompts_root.exists():
            return []
        layers = [path.name for path in self.prompts_root.iterdir() if path.is_dir()]
        return sorted(layers)

    def read_active_stack(self) -> list[str]:
        active_path = self.prompts_root / "active_stack.txt"
        if active_path.exists():
            content = active_path.read_text(encoding="utf-8").strip()
            if not content:
                raise PromptError("Active prompt stack file is empty.")
            stack = [item for item in content.split(",") if item]
            self._validate_layers(stack)
            return stack
        if (self.prompts_root / "default").is_dir():
            return ["default"]
        raise PromptError("No active stack and no prompts/default present")

    def write_active_stack(self, stack: list[str]) -> None:
        if not stack:
            raise PromptError("Prompt stack must include at least one layer.")
        self._validate_layers(stack)
        content = ",".join(stack)
        active_path = self.prompts_root / "active_stack.txt"
        active_path.write_text(content, encoding="utf-8")

    def list_bundles(self, layers: list[str]) -> list[str]:
        bundles: set[str] = set()
        for layer in layers:
            layer_root = self._layer_root(layer)
            for entry in layer_root.iterdir():
                if entry.is_dir() and entry.name != "shared":
                    bundles.add(entry.name)
        return sorted(bundles)

    def resolve_sources(self, stack: list[str], bundle: str, role: str) -> ResolvedPromptSources:
        body_path: Path | None = None
        prepend_paths: list[Path] = []
        append_paths: list[Path] = []
        for layer in stack:
            layer_root = self._layer_root(layer)
            bundle_root = layer_root / bundle
            if not bundle_root.is_dir():
                continue
            prepend_path = bundle_root / f"{role}.prepend.j2"
            if prepend_path.is_file():
                prepend_paths.append(prepend_path)
            body_candidate = bundle_root / f"{role}.j2"
            if body_candidate.is_file():
                body_path = body_candidate
            append_path = bundle_root / f"{role}.append.j2"
            if append_path.is_file():
                append_paths.append(append_path)
        if body_path is None:
            raise PromptError(f"Missing prompt template for {bundle}/{role}.")
        return ResolvedPromptSources(
            body_path=body_path,
            prepend_paths=prepend_paths,
            append_paths=append_paths,
            stack=list(stack),
        )

    def render(self, stack: list[str], bundle: str, role: str, extra_vars: dict[str, Any] | None = None) -> str:
        template_text, _sources = self._compose_text(stack, bundle, role, runtime_prepend=None)
        env = self._build_environment(stack)
        context = self._merge_dicts(self._load_variables(stack), extra_vars or {})
        return env.from_string(template_text).render(**context)

    def render_with_prepend(
        self,
        stack: list[str],
        bundle: str,
        role: str,
        runtime_prepend: str | None,
        extra_vars: dict[str, Any] | None = None,
    ) -> str:
        template_text, _sources = self._compose_text(stack, bundle, role, runtime_prepend=runtime_prepend)
        env = self._build_environment(stack)
        context = self._merge_dicts(self._load_variables(stack), extra_vars or {})
        return env.from_string(template_text).render(**context)

    def compose_text(
        self,
        stack: list[str],
        bundle: str,
        role: str,
        runtime_prepend: str | None = None,
    ) -> tuple[str, ResolvedPromptSources]:
        template_text, sources = self._compose_text(stack, bundle, role, runtime_prepend=runtime_prepend)
        return template_text, sources

    def parse_template(self, stack: list[str], relative_path: str) -> None:
        env = self._build_environment(stack)
        source = env.loader.get_source(env, relative_path)[0]
        env.parse(source)

    def load_variables(self, stack: list[str]) -> dict[str, Any]:
        return self._load_variables(stack)

    def _compose_text(
        self,
        stack: list[str],
        bundle: str,
        role: str,
        runtime_prepend: str | None,
    ) -> tuple[str, ResolvedPromptSources]:
        sources = self.resolve_sources(stack, bundle, role)
        parts: list[str] = []
        for path in sources.prepend_paths:
            parts.append(path.read_text(encoding="utf-8"))
        if runtime_prepend:
            parts.append(runtime_prepend)
        parts.append(sources.body_path.read_text(encoding="utf-8"))
        for path in sources.append_paths:
            parts.append(path.read_text(encoding="utf-8"))
        return "".join(parts), sources

    def _build_environment(self, stack: list[str]) -> Environment:
        loaders = [
            FileSystemLoader(str(self._layer_root(layer)))
            for layer in reversed(stack)
        ]
        return Environment(loader=ChoiceLoader(loaders), autoescape=False)

    def _load_variables(self, stack: list[str]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for layer in stack:
            variables_path = self._layer_root(layer) / "variables.yaml"
            if not variables_path.exists():
                continue
            raw = yaml.safe_load(variables_path.read_text(encoding="utf-8"))
            if raw is None:
                continue
            if not isinstance(raw, dict):
                raise PromptError(f"variables.yaml must contain a mapping in {variables_path}.")
            merged = self._merge_dicts(merged, raw)
        return merged

    def _merge_dicts(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _layer_root(self, layer: str) -> Path:
        layer_root = self.prompts_root / layer
        if not layer_root.is_dir():
            raise PromptError(f"Prompt layer not found: {layer}")
        return layer_root

    def _validate_layers(self, stack: list[str]) -> None:
        missing = [layer for layer in stack if not (self.prompts_root / layer).is_dir()]
        if missing:
            raise PromptError(f"Prompt layers not found: {', '.join(missing)}")
