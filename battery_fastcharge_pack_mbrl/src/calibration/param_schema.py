"""Parameter schema definitions for calibration."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParamSchema:
    """Schema describing calibratable parameters."""

    names: list[str]

    @property
    def dim(self) -> int:
        """Return number of parameters."""

        return len(self.names)
