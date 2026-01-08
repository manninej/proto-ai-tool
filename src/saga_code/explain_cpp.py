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


def build_files_block(files: list[FileBlob]) -> str:
    return "\n\n".join(_format_file_block(blob) for blob in files)


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


def parse_sections(text: str) -> dict[str, str]:
    normalized = strip_final_prefix(text).strip()
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
    normalized = strip_final_prefix(text).strip()
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


def strip_final_prefix(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("FINAL:"):
        return stripped[len("FINAL:") :].lstrip()
    return text


def _relpath(path: Path, cwd: Path) -> str:
    return os.path.relpath(path, cwd)


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
