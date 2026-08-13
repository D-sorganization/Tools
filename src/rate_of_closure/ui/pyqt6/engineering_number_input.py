"""Shared immutable specification for PyQt6 engineering-number inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PyQt6.QtWidgets import QDoubleSpinBox


@dataclass(frozen=True)
class NumberInputSpec:
    """Immutable display and range constraints for one numeric control."""

    suffix: str = ""
    step: float = 0.1
    minimum: float = -1e9
    maximum: float = 1e9

    def __post_init__(self) -> None:
        """Reject unusable Qt ranges before a widget is constructed."""
        if not math.isfinite(self.step) or self.step <= 0.0:
            raise ValueError("step must be finite and positive")
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("minimum and maximum must be finite")
        if self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")


def engineering_number_input(
    name: str,
    value: float,
    spec: NumberInputSpec,
) -> QDoubleSpinBox:
    """Create one consistently configured accessible SI number input."""
    field = QDoubleSpinBox()
    field.setAccessibleName(name)
    field.setDecimals(6)
    field.setRange(spec.minimum, spec.maximum)
    field.setSingleStep(spec.step)
    field.setSuffix(spec.suffix)
    field.setValue(value)
    field.setToolTip(
        f"{name}. Edit this SI draft value, then validate the surface plan."
    )
    return field


__all__ = ["NumberInputSpec", "engineering_number_input"]
