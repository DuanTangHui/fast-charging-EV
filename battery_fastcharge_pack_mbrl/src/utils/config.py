"""Configuration loading helpers for YAML-based experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    """Simple wrapper for configuration dictionaries."""

    data: Dict[str, Any]

    def get(self, key: str, default: Any | None = None) -> Any:
        """Return a top-level config field with default."""

        return self.data.get(key, default)


def load_config(path: str | Path) -> Config:
    """Load a YAML configuration file.

    Args:
        path: Path to YAML file.

    Returns:
        Config object with parsed data.
    """

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return Config(data)
