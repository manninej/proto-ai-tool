from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


CPP_EXTENSIONS = {".cpp", ".cc", ".cxx", ".c", ".hpp", ".hh", ".hxx", ".h"}

SYSTEM_PROMPT = """You are a senior C++ engineer performing static analysis.
Do not generate code.
Do not suggest changes unless explicitly asked.
Base conclusions strictly on the provided files only.
If something is unclear, say so explicitly."""

SECTION_TITLES = {
    "overview": "Overview",
    "components": "Key Components",
    "data_flow": "Data Flow",
    "assumptions": "Assumptions",
    "risks": "Risks / Pitfalls",
    "open_questions": "Open Questions",
}


@dataclass(frozen=True)
class FileBlob:
    path: str
    content: str
    byte_size: int


@dataclass(frozen=True)
class SkipInfo:
    path: str
    reason: str


def collect_source_files(paths: list[str], exts: set[str], max_files: int) -> list[Path]:
    all_files = discover_source_files(paths, exts)
    return all_files[:max_files]


def discover_source_files(paths: Iterable[str], exts: set[str]) -> list[Path]:
    return _discover_source_files(paths, exts)


def _discover_source_files(paths: Iterable[str], exts: set[str]) -> list[Path]:
    cwd = Path.cwd()
    collected: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        if path.is_file():
            if path.suffix.lower() in exts:
                collected.append(path)
            continue
        if path.is_dir():
            for candidate in path.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() in exts:
                    collected.append(candidate)
    return sorted(collected, key=lambda item: os.path.relpath(item, cwd))


def read_files_with_budget(files: list[Path], max_bytes: int) -> tuple[list[FileBlob], list[SkipInfo]]:
    cwd = Path.cwd()
    total_bytes = 0
    blobs: list[FileBlob] = []
    skipped: list[SkipInfo] = []
    for index, path in enumerate(files):
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            content = ""
        content_bytes = len(content.encode("utf-8"))
        if total_bytes + content_bytes > max_bytes:
            skipped.append(SkipInfo(path=_relpath(path, cwd), reason="max-bytes"))
            for remaining in files[index + 1 :]:
                skipped.append(SkipInfo(path=_relpath(remaining, cwd), reason="max-bytes"))
            break
        blobs.append(FileBlob(path=_relpath(path, cwd), content=content, byte_size=content_bytes))
        total_bytes += content_bytes
    return blobs, skipped


def build_system_prompt(system_override: str) -> str:
    if system_override.strip():
        return f"{SYSTEM_PROMPT}\n\n{system_override.strip()}"
    return SYSTEM_PROMPT


def build_user_prompt(files: list[FileBlob], json_mode: bool) -> str:
    files_block = "\n\n".join(_format_file_block(blob) for blob in files)
    if json_mode:
        return (
            "ANALYSIS ONLY. Do not generate code or propose changes.\n"
            "Provide a structured explanation of the provided C++ files.\n"
            "Reply with FINAL: followed immediately by a JSON object matching this exact shape:\n"
            "{\n"
            "  \"overview\": string,\n"
            "  \"components\": [\n"
            "    { \"name\": string, \"responsibility\": string }\n"
            "  ],\n"
            "  \"data_flow\": string,\n"
            "  \"assumptions\": [string],\n"
            "  \"risks\": [string],\n"
            "  \"open_questions\": [string]\n"
            "}\n\n"
            "Files:\n"
            f"{files_block}"
        )
    return (
        "ANALYSIS ONLY. Do not generate code or propose changes.\n"
        "Provide a structured explanation of the provided C++ files with these sections:\n"
        "Overview, Key Components (bulleted list), Data Flow, Assumptions, Risks / Pitfalls, Open Questions.\n"
        "Use clear section headings exactly as listed.\n\n"
        "Files:\n"
        f"{files_block}"
    )


def parse_sections(text: str) -> dict[str, str]:
    normalized = _strip_final_prefix(text).strip()
    sections = {key: "" for key in SECTION_TITLES}
    current_key: str | None = None
    for line in normalized.splitlines():
        stripped = line.strip()
        matched_key = _match_section_heading(stripped)
        if matched_key:
            current_key = matched_key
            remainder = stripped[len(SECTION_TITLES[matched_key]) :].lstrip(":").strip()
            if remainder:
                sections[current_key] += remainder + "\n"
            continue
        if current_key:
            sections[current_key] += line + "\n"
    if not any(value.strip() for value in sections.values()):
        sections["overview"] = normalized
    for key, value in sections.items():
        sections[key] = value.strip() or "Not provided."
    return sections


def parse_json_response(text: str) -> dict[str, object] | None:
    normalized = _strip_final_prefix(text).strip()
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        return None
    if not _validate_json_shape(payload):
        return None
    return payload


def render_explanation(console: Console, sections: dict[str, str]) -> None:
    for key, title in SECTION_TITLES.items():
        panel = Panel(Markdown(sections.get(key, "")), title=title)
        console.print(panel)


def render_json_explanation(console: Console, payload: dict[str, object]) -> None:
    overview = payload.get("overview", "")
    data_flow = payload.get("data_flow", "")
    components = payload.get("components", [])
    assumptions = payload.get("assumptions", [])
    risks = payload.get("risks", [])
    open_questions = payload.get("open_questions", [])

    panels = {
        "overview": overview,
        "components": _format_components(components),
        "data_flow": data_flow,
        "assumptions": _format_list(assumptions),
        "risks": _format_list(risks),
        "open_questions": _format_list(open_questions),
    }

    for key, title in SECTION_TITLES.items():
        panel = Panel(Markdown(panels.get(key, "Not provided.")), title=title)
        console.print(panel)


def render_warning(console: Console, skipped: list[SkipInfo]) -> None:
    if not skipped:
        return
    lines = [f"- {item.path} ({item.reason})" for item in skipped]
    panel = Panel("\n".join(lines), title="Warning", style="yellow")
    console.print(panel)


def _format_file_block(blob: FileBlob) -> str:
    return f"<file path=\"{blob.path}\">\n{blob.content}\n</file>"


def _match_section_heading(line: str) -> str | None:
    lower = line.lower()
    for key, title in SECTION_TITLES.items():
        if lower.startswith(title.lower()):
            return key
    return None


def _strip_final_prefix(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("FINAL:"):
        return stripped[len("FINAL:") :].lstrip()
    return text


def _relpath(path: Path, cwd: Path) -> str:
    return os.path.relpath(path, cwd)


def _format_components(components: object) -> str:
    if not isinstance(components, list) or not components:
        return "Not provided."
    lines: list[str] = []
    for item in components:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "")
        responsibility = item.get("responsibility", "")
        if name and responsibility:
            lines.append(f"- **{name}**: {responsibility}")
        elif name:
            lines.append(f"- **{name}**")
        elif responsibility:
            lines.append(f"- {responsibility}")
    return "\n".join(lines) if lines else "Not provided."


def _format_list(items: object) -> str:
    if not isinstance(items, list) or not items:
        return "Not provided."
    lines = [f"- {item}" for item in items if isinstance(item, str) and item.strip()]
    return "\n".join(lines) if lines else "Not provided."


def _validate_json_shape(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    required_keys = {
        "overview",
        "components",
        "data_flow",
        "assumptions",
        "risks",
        "open_questions",
    }
    if set(payload.keys()) != required_keys:
        return False
    if not isinstance(payload["overview"], str):
        return False
    if not isinstance(payload["data_flow"], str):
        return False
    if not isinstance(payload["components"], list):
        return False
    for item in payload["components"]:
        if not isinstance(item, dict):
            return False
        if set(item.keys()) != {"name", "responsibility"}:
            return False
        if not isinstance(item["name"], str) or not isinstance(item["responsibility"], str):
            return False
    for key in ("assumptions", "risks", "open_questions"):
        if not isinstance(payload[key], list):
            return False
        if not all(isinstance(entry, str) for entry in payload[key]):
            return False
    return True
