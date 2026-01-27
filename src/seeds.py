"""Seed utilities for reproducibility."""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: Optional[int]) -> None:
    """Set global random seeds for numpy, torch, and random."""

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
