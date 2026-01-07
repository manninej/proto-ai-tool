from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import httpx

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
INITIAL_BACKOFF = 0.5
BACKOFF_FACTOR = 2.0


@dataclass(frozen=True)
class ApiError(Exception):
    status_code: int
    message: str
    endpoint: str


@dataclass(frozen=True)
class NetworkError(Exception):
    message: str


class OpenAIClient:
    def __init__(self, base_url: str, api_key: str | None, timeout: int, ca_bundle: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.ca_bundle = ca_bundle

    def list_models(self) -> list[str] | None:
        response = self._request("GET", "/v1/models")
        data = self._parse_json(response, "/v1/models")
        items = data.get("data", [])
        if isinstance(items, list):
            return [item["id"] for item in items if isinstance(item, dict) and "id" in item]
        return None

    def probe_chat_completion(self, model: str) -> bool:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
        try:
            response = self._request("POST", "/v1/chat/completions", json=payload)
        except ApiError as exc:
            if exc.status_code in {400, 403, 404}:
                return False
            raise
        return response.status_code // 100 == 2

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = self._request("POST", "/v1/chat/completions", json=payload)
        data = self._parse_json(response, "/v1/chat/completions")
        assistant_text = ""
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    assistant_text = content
        return {
            "assistant_text": assistant_text,
            "model": data.get("model") if isinstance(data.get("model"), str) else model,
            "data": data,
        }

    def _request(self, method: str, endpoint: str, json: dict[str, Any] | None = None) -> httpx.Response:
        url = f"{self.base_url}{endpoint}"
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                verify = self.ca_bundle if self.ca_bundle else True
                with httpx.Client(timeout=self.timeout, verify=verify) as client:
                    response = client.request(method, url, headers=headers, json=json)
                if response.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                    self._sleep_backoff(attempt)
                    continue
                if response.status_code // 100 != 2:
                    raise ApiError(
                        status_code=response.status_code,
                        message=response.text,
                        endpoint=endpoint,
                    )
                return response
            except httpx.RequestError as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    self._sleep_backoff(attempt)
                    continue
                raise NetworkError(str(exc)) from exc

        raise NetworkError(str(last_error) if last_error else "Unknown network error")

    def _parse_json(self, response: httpx.Response, endpoint: str) -> dict[str, Any]:
        try:
            return response.json()
        except ValueError as exc:
            raise ApiError(status_code=response.status_code, message="Invalid JSON", endpoint=endpoint) from exc

    def _sleep_backoff(self, attempt: int) -> None:
        delay = INITIAL_BACKOFF * (BACKOFF_FACTOR ** (attempt - 1))
        time.sleep(delay)
