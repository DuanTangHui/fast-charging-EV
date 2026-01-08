"""STAPINN predictor interface and dummy implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class SOHPredictor(Protocol):
    """Protocol for SOH predictors."""

    def predict(self, features: np.ndarray) -> float:
        """Predict SOH from features."""


@dataclass
class DummyPredictor:
    """Dummy predictor returning a fixed SOH value."""

    fixed_soh: float = 0.95

    def predict(self, features: np.ndarray) -> float:
        return float(self.fixed_soh)
