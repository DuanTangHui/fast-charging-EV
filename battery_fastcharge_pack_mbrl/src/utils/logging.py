"""Logging utilities for metrics and figures under runs/."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists and return its Path."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_metrics(path: str | Path, metrics: Dict[str, Any]) -> None:
    """Append a metrics dictionary as JSON line."""

    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics) + "\n")


def summarize_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Compute mean for numeric metric keys."""

    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for record in records:
        for key, value in record.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in totals}
