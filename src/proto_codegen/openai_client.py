from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

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
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        timeout: int,
        ca_bundle: str | None = None,
        debug_http: bool = False,
        debug_sink: Callable[[str], None] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.ca_bundle = ca_bundle
        self.debug_http = debug_http
        self.debug_sink = debug_sink

    def list_models(self) -> list[str] | None:
        response = self._request("GET", "/v1/models")
        data = self._parse_json(response, "/v1/models")
        items = data.get("data", None)
        if isinstance(items, list):
            return [item["id"] for item in items if isinstance(item, dict) and "id" in item]
        models = data.get("models")
        if isinstance(models, dict):
            results = []
            for key, value in models.items():
                if isinstance(value, dict) and "model_name" in value:
                    results.append(value["model_name"])
                else:
                    results.append(key)
            return results
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
        assistant_text = self._extract_assistant_text(data, response, "/v1/chat/completions")
        reasoning_text = self._extract_reasoning_text(data)
        return {
            "assistant_text": assistant_text,
            "model": data.get("model") if isinstance(data.get("model"), str) else model,
            "reasoning_text": reasoning_text,
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
            data = response.json()
        except ValueError as exc:
            raise ApiError(status_code=response.status_code, message="Invalid JSON", endpoint=endpoint) from exc
        if self.debug_http:
            self._emit_debug(response, data)
        return data

    def _sleep_backoff(self, attempt: int) -> None:
        delay = INITIAL_BACKOFF * (BACKOFF_FACTOR ** (attempt - 1))
        time.sleep(delay)

    def _extract_assistant_text(self, data: dict[str, Any], response: httpx.Response, endpoint: str) -> str:
        choices = data.get("choices")
        message = None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            candidate = choices[0].get("message")
            if isinstance(candidate, dict):
                message = candidate

        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and content.strip():
            return content

        reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
        if isinstance(reasoning, str) and reasoning.strip() and "FINAL:" in reasoning:
            return reasoning

        top_level_keys = list(data.keys()) if isinstance(data, dict) else []
        message_keys = list(message.keys()) if isinstance(message, dict) else []
        raise ApiError(
            status_code=response.status_code,
            message=(
                "No final assistant content produced. "
                f"keys={top_level_keys} message_keys={message_keys}"
            ),
            endpoint=endpoint,
        )

    def _extract_reasoning_text(self, data: dict[str, Any]) -> str | None:
        choices = data.get("choices")
        if not (isinstance(choices, list) and choices and isinstance(choices[0], dict)):
            return None
        message = choices[0].get("message")
        if not isinstance(message, dict):
            return None
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning
        return None

    def _emit_debug(self, response: httpx.Response, data: Any) -> None:
        sink = self.debug_sink or print
        status = response.status_code
        top_level_keys = list(data.keys()) if isinstance(data, dict) else []
        message_keys: list[str] = []
        if isinstance(data, dict):
            choices = data.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                message = choices[0].get("message")
                if isinstance(message, dict):
                    message_keys = list(message.keys())
        sink(f"HTTP {status}")
        sink(f"Response keys: {top_level_keys}")
        sink(f"choices[0].message keys: {message_keys}")
