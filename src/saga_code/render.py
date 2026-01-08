from __future__ import annotations

import json
from typing import Iterable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from saga_code.model_discovery import ModelResult

console = Console()


def print_models_table(results: Iterable[ModelResult]) -> None:
    table = Table(title="Discovered Models")
    table.add_column("model_id", style="cyan")
    table.add_column("discovery_method", style="magenta")
    table.add_column("status", style="green")
    table.add_column("details", style="white")

    for result in results:
        table.add_row(
            result.model_id,
            result.discovery_method,
            result.status,
            result.details,
        )

    console.print(table)


def print_error_panel(err: str) -> None:
    console.print(Panel(err, title="Error", style="red"))


def print_json(obj: object) -> None:
    console.print_json(json.dumps(obj, sort_keys=True))
