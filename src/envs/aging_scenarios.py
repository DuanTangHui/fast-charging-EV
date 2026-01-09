"""Aging scenario utilities for pack simulations."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgingParams:
    """Simple aging parameters for toy environment."""

    capacity_scale: float
    resistance_scale: float
    thermal_scale: float


def compute_aging_params(cycle_index: int, theta_hat: list[float] | None = None) -> AgingParams:
    """Return aging parameters for a cycle.

    Args:
        cycle_index: Index of adaptive cycle.
        theta_hat: Optional calibrated parameters.

    Returns:
        AgingParams with scaling factors.
    """

    base_capacity = 1.0 - 0.01 * cycle_index
    base_resistance = 1.0 + 0.02 * cycle_index
    base_thermal = 1.0 + 0.01 * cycle_index
    if theta_hat:
        capacity_scale = base_capacity * float(theta_hat[0])
        resistance_scale = base_resistance * float(theta_hat[1])
        thermal_scale = base_thermal * float(theta_hat[2])
    else:
        capacity_scale = base_capacity
        resistance_scale = base_resistance
        thermal_scale = base_thermal
    return AgingParams(capacity_scale, resistance_scale, thermal_scale)
