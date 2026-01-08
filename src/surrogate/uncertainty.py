"""Uncertainty utilities for surrogate outputs."""
from __future__ import annotations

import numpy as np


def aggregate_uncertainty(stds: np.ndarray) -> float:
    """Aggregate per-dimension std into a scalar measure."""

    return float(np.mean(stds))
