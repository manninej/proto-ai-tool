from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from proto_codegen.openai_client import ApiError, NetworkError, OpenAIClient

PreferEndpoint = Literal["models", "probe", "auto"]


@dataclass(frozen=True)
class ModelResult:
    model_id: str
    discovery_method: str
    status: str
    details: str


def discover_models(
    client: OpenAIClient,
    prefer_endpoint: PreferEndpoint,
    candidates: Iterable[str],
) -> list[ModelResult]:
    if prefer_endpoint != "probe":
        try:
            models = client.list_models()
            if models is not None:
                return [
                    ModelResult(
                        model_id=model_id,
                        discovery_method="models_endpoint",
                        status="available",
                        details="listed",
                    )
                    for model_id in models
                ]
            if prefer_endpoint == "models":
                return []
        except ApiError as exc:
            if prefer_endpoint == "models" or exc.status_code not in {403, 404}:
                raise
        except NetworkError:
            if prefer_endpoint == "models":
                raise

    return _probe_models(client, candidates)


def _probe_models(client: OpenAIClient, candidates: Iterable[str]) -> list[ModelResult]:
    results: list[ModelResult] = []
    for model_id in candidates:
        try:
            available = client.probe_chat_completion(model_id)
            if available:
                results.append(
                    ModelResult(
                        model_id=model_id,
                        discovery_method="probe",
                        status="available",
                        details="probe succeeded",
                    )
                )
            else:
                results.append(
                    ModelResult(
                        model_id=model_id,
                        discovery_method="probe",
                        status="unavailable",
                        details="probe rejected",
                    )
                )
        except ApiError as exc:
            results.append(
                ModelResult(
                    model_id=model_id,
                    discovery_method="probe",
                    status="error",
                    details=f"HTTP {exc.status_code}",
                )
            )
        except NetworkError as exc:
            results.append(
                ModelResult(
                    model_id=model_id,
                    discovery_method="probe",
                    status="error",
                    details=str(exc),
                )
            )
    return results
