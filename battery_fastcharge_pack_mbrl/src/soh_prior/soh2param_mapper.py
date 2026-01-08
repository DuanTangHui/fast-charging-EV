"""Mapping from SOH to parameter priors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SOHToParamMapper:
    """Linear mapping from SOH to parameter vector."""

    theta_dim: int

    def map(self, soh: float) -> np.ndarray:
        """Map SOH to a parameter prior vector."""

        base = np.ones(self.theta_dim)
        return base * float(soh)
