"""Immutable metadata contract shared by plotting catalog consumers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from rate_of_closure._contracts import require
from rate_of_closure.simulation.session import SimulationRun

__all__ = [
    "CATEGORIES",
    "DISTANCE_KEYS",
    "Extractor",
    "SCALE_HINTS",
    "SERIES_CATEGORIES",
    "VariableSpec",
]

#: Catalog categories in display order. Array-valued (per-sample)
#: categories first, scalar (per-run) categories after.
CATEGORIES: tuple[str, ...] = (
    "Input",
    "Swing Sample",
    "Kinetics",
    "Impact",
    "Launch",
    "Flight",
    "Metric",
)

#: Categories whose extractors return per-sample arrays.
SERIES_CATEGORIES: frozenset[str] = frozenset({"Swing Sample", "Kinetics", "Flight"})

#: Axis-scale hints accepted by :class:`VariableSpec`.
SCALE_HINTS: tuple[str, ...] = ("linear", "log")

Extractor = Callable[[SimulationRun], "np.ndarray | float"]


@dataclass(frozen=True)
class VariableSpec:
    """Describe one plottable simulation variable.

    Args:
        key: Stable namespaced identifier, ``category.name``.
        label: Title Case display label.
        unit: Display unit string, or an empty string when dimensionless.
        category: One of :data:`CATEGORIES`.
        extractor: Callable returning a one-dimensional array or scalar.
        scale: Axis-scale hint, ``"linear"`` or ``"log"``.
    """

    key: str
    label: str
    unit: str
    category: str
    extractor: Extractor
    scale: str = "linear"

    def __post_init__(self) -> None:
        require(
            bool(self.key) and "." in self.key,
            "key must be namespaced as 'category.name'",
            self.key,
        )
        require(bool(self.label), "label must be non-empty", self.label)
        require(self.category in CATEGORIES, "unknown category", self.category)
        require(self.scale in SCALE_HINTS, "unknown scale hint", self.scale)
        require(callable(self.extractor), "extractor must be callable")

    @property
    def is_series(self) -> bool:
        """Return whether the extractor yields a per-sample array."""
        return self.category in SERIES_CATEGORIES

    @property
    def axis_label(self) -> str:
        """Return ``Label [unit]``, omitting the unit when empty."""
        return f"{self.label} [{self.unit}]" if self.unit else self.label


#: Ball-flight distance variables that follow the user's Distance
#: display unit. Heights and swing-scale positions stay in metres.
DISTANCE_KEYS: frozenset[str] = frozenset(
    {
        "flight.x_m",
        "flight.z_m",
        "metric.carry_m",
        "putting.path_x",
        "putting.path_y",
        "putting.rollout",
        "putting.skid_distance",
        "putting.break",
    }
)
