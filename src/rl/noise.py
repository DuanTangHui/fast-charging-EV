"""Exploration noise processes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianNoise:
    """Gaussian noise for exploration."""

    sigma: float

    def sample(self) -> float:
        return float(np.random.normal(0.0, self.sigma))
