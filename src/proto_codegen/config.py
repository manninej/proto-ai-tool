from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable, Sequence

DEFAULT_BASE_URL = "https://api.openai.com"
DEFAULT_TIMEOUT = 30
DEFAULT_CANDIDATES = ["gpt-oss-120b"]
CANDIDATE_ENV_VAR = "PROTO_CODEGEN_CANDIDATE_MODELS"


@dataclass(frozen=True)
class Config:
    base_url: str
    api_key: str | None
    timeout: int
    candidates: tuple[str, ...]

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)


def _split_candidates(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_config(
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: int | None = None,
    candidates: Sequence[str] | None = None,
) -> Config:
    env_base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    env_api_key = os.getenv("OPENAI_API_KEY")
    env_candidates = _split_candidates(os.getenv(CANDIDATE_ENV_VAR))

    resolved_candidates: list[str] = []
    resolved_candidates.extend(DEFAULT_CANDIDATES)
    resolved_candidates.extend(env_candidates)
    if candidates:
        resolved_candidates.extend(candidates)

    return Config(
        base_url=base_url or env_base_url,
        api_key=api_key or env_api_key,
        timeout=timeout if timeout is not None else DEFAULT_TIMEOUT,
        candidates=tuple(_dedupe_preserve_order(resolved_candidates)),
    )


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
