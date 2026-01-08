"""Feature extraction for SOH predictor inputs."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def extract_features(segments: List[Dict[str, np.ndarray]]) -> np.ndarray:
    """Extract simple features from charge segments.

    Args:
        segments: List of dicts containing time-series arrays.

    Returns:
        Feature vector for SOH prediction.
    """

    if not segments:
        return np.zeros(4, dtype=np.float32)
    soc_gains = [seg["soc"][-1] - seg["soc"][0] for seg in segments]
    dv_max = [np.max(seg["voltage"]) - np.min(seg["voltage"]) for seg in segments]
    dt_max = [np.max(seg["temperature"]) - np.min(seg["temperature"]) for seg in segments]
    duration = [seg["time"][-1] - seg["time"][0] for seg in segments]
    feats = np.array([
        float(np.mean(soc_gains)),
        float(np.mean(dv_max)),
        float(np.mean(dt_max)),
        float(np.mean(duration)),
    ])
    return feats
