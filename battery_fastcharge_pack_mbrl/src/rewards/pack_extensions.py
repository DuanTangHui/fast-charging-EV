"""Optional pack-level reward extensions (disabled by default)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class PackExtensionConfig:
    """Penalty weights for cell spread extensions."""

    w_dv: float = 0.0
    w_dt: float = 0.0


def compute_pack_extensions(info: Dict, config: PackExtensionConfig) -> float:
    """Compute extra penalty terms for voltage/temperature spread."""

    return -config.w_dv * float(info.get("dV", 0.0)) - config.w_dt * float(info.get("dT", 0.0))
