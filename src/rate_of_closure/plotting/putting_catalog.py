"""Putting output catalog — additive registry for the Putting tab (#4125 H3).

A parallel, additive registry in the style of
:mod:`rate_of_closure.plotting.catalog`, scoped to
:class:`~shared.python.swing_sim.putting.PuttResult` instead of
``SimulationRun``. Kept separate so the pinned SimulationRun catalog
(and its web parity fixture) is untouched — putting outputs register
additively alongside it, and the Putting tab's plot list and exports
read from here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from rate_of_closure._contracts import ensure, require
from shared.python.swing_sim.putting import PuttResult

__all__ = [
    "PUTTING_CATALOG",
    "PuttingVariableSpec",
    "extract_putting",
    "putting_catalog_keys",
]

PuttingExtractor = Callable[[PuttResult], "np.ndarray | float"]


@dataclass(frozen=True)
class PuttingVariableSpec:
    """One plottable putting variable.

    Args:
        key: Stable namespaced identifier, ``putting.<name>``.
        label: Title Case display label.
        unit: Display unit string ("" for dimensionless).
        extractor: ``result -> np.ndarray | float`` in SI units.
        is_series: True when the extractor yields a per-sample array.
    """

    key: str
    label: str
    unit: str
    extractor: PuttingExtractor
    is_series: bool = False

    def __post_init__(self) -> None:
        require(
            self.key.startswith("putting."),
            "putting keys are namespaced 'putting.<name>'",
            self.key,
        )
        require(bool(self.label), "label must be non-empty", self.label)
        require(callable(self.extractor), "extractor must be callable")

    @property
    def axis_label(self) -> str:
        """Axis label: ``Label [unit]`` (unit omitted when empty)."""
        return f"{self.label} [{self.unit}]" if self.unit else self.label


def _series(values: tuple[float, ...]) -> np.ndarray:
    catalog: np.ndarray = np.asarray(values, dtype=float)
    return catalog


def _entries() -> list[PuttingVariableSpec]:
    """Build the putting registry (one literal list, easy to review)."""
    series: list[tuple[str, str, str, PuttingExtractor]] = [
        ("path_x", "Path Along Putt Line", "m", lambda r: _series(r.path_x_m)),
        ("path_y", "Path Lateral (Left +)", "m", lambda r: _series(r.path_y_m)),
        ("speed", "Ball Speed", "m/s", lambda r: _series(r.speeds_mps)),
        ("time", "Time", "s", lambda r: _series(r.times_s)),
    ]
    scalars: list[tuple[str, str, str, PuttingExtractor]] = [
        ("rollout", "Roll-Out Distance", "m", lambda r: r.total_distance_m),
        ("skid_distance", "Skid Distance", "m", lambda r: r.skid_distance_m),
        ("skid_fraction", "Skid Fraction", "", lambda r: r.skid_fraction),
        ("time_total", "Time To Rest", "s", lambda r: r.time_s),
        ("break", "Break (Left +)", "m", lambda r: r.break_m),
        ("holed", "Holed (1 = Yes)", "", lambda r: 1.0 if r.holed else 0.0),
    ]
    return [
        PuttingVariableSpec(
            key=f"putting.{name}",
            label=label,
            unit=unit,
            extractor=extractor,
            is_series=True,
        )
        for name, label, unit, extractor in series
    ] + [
        PuttingVariableSpec(
            key=f"putting.{name}", label=label, unit=unit, extractor=extractor
        )
        for name, label, unit, extractor in scalars
    ]


#: The additive putting registry, keyed ``putting.<name>``.
PUTTING_CATALOG: dict[str, PuttingVariableSpec] = {
    spec.key: spec for spec in _entries()
}
ensure(len(PUTTING_CATALOG) == len(_entries()), "putting keys must be unique")


def putting_catalog_keys() -> tuple[str, ...]:
    """Stable key list, catalog order."""
    return tuple(PUTTING_CATALOG)


def extract_putting(result: PuttResult, key: str) -> np.ndarray | float:
    """Extract one catalogued variable from a putt result.

    Args:
        result: An integrated putt.
        key: A :data:`PUTTING_CATALOG` key.

    Returns:
        The variable value (array for series, float for scalars).

    Raises:
        ValueError: If the key is unknown.
    """
    require(key in PUTTING_CATALOG, f"unknown putting key {key!r}", key)
    return PUTTING_CATALOG[key].extractor(result)
