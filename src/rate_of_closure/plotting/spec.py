"""Frozen plot definitions with JSON round-trip.

A :class:`PlotSpec` fully describes one investigative plot: X variable,
one or more Y variables, an optional color/series variable, the plot
kind, title, log-axis flags, and (for sweep plots) the swept input
range. Definitions export to and import from a small versioned JSON
schema shared verbatim with the web clone, so investigations are
reproducible across sessions and UIs.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from rate_of_closure._contracts import ensure, require
from rate_of_closure.plotting.catalog import CATALOG

__all__ = [
    "PLOT_KINDS",
    "SPEC_FORMAT",
    "PlotSpec",
    "spec_from_json",
    "spec_to_json",
]

#: Supported plot kinds. ``sweep`` re-runs the simulation across the
#: X input's range; the others read series straight off one run.
PLOT_KINDS: tuple[str, ...] = ("line", "scatter", "sweep", "histogram")

#: JSON schema identifier (bump on breaking changes; web pins it too).
SPEC_FORMAT = "rate_of_closure.plot_spec/1"

#: Scalar categories a sweep plot may use as Y variables.
_SWEEP_Y_CATEGORIES = frozenset({"Impact", "Launch", "Metric"})


@dataclass(frozen=True)
class PlotSpec:
    """One plot definition.

    Args:
        kind: One of :data:`PLOT_KINDS`.
        x_key: Catalog key of the X variable (for ``histogram`` the
            variable whose distribution is binned).
        y_keys: Catalog keys of the Y variables (empty for histogram).
        series_key: Optional catalog key coloring scatter points.
        title: Plot title (Title Case by convention).
        x_log: Log-scale the X axis.
        y_log: Log-scale the Y axis.
        x_start: Sweep start value (sweep kind only).
        x_stop: Sweep stop value (sweep kind only).
        x_count: Number of sweep points (sweep kind only).
    """

    kind: str
    x_key: str
    y_keys: tuple[str, ...] = ()
    series_key: str | None = None
    title: str = ""
    x_log: bool = False
    y_log: bool = False
    x_start: float | None = None
    x_stop: float | None = None
    x_count: int = 25

    def __post_init__(self) -> None:
        require(self.kind in PLOT_KINDS, f"unknown plot kind {self.kind!r}")
        require(self.x_key in CATALOG, f"unknown x_key {self.x_key!r}")
        object.__setattr__(self, "y_keys", tuple(self.y_keys))
        for key in self.y_keys:
            require(key in CATALOG, f"unknown y_key {key!r}")
        if self.series_key is not None:
            require(
                self.series_key in CATALOG,
                f"unknown series_key {self.series_key!r}",
            )
        if self.kind == "histogram":
            require(
                CATALOG[self.x_key].is_series,
                "histogram needs a per-sample x variable",
                self.x_key,
            )
        elif self.kind == "sweep":
            require(
                CATALOG[self.x_key].category == "Input",
                "sweep x_key must be an Input variable",
                self.x_key,
            )
            require(bool(self.y_keys), "sweep needs at least one y_key")
            for key in self.y_keys:
                require(
                    CATALOG[key].category in _SWEEP_Y_CATEGORIES,
                    "sweep y_keys must be scalar outputs (Impact / Launch / Metric)",
                    key,
                )
            require(
                self.x_start is not None and self.x_stop is not None,
                "sweep needs x_start and x_stop",
            )
            assert self.x_start is not None and self.x_stop is not None
            require(
                math.isfinite(self.x_start) and math.isfinite(self.x_stop),
                "sweep bounds must be finite",
            )
            require(self.x_start < self.x_stop, "x_start must be < x_stop")
            require(2 <= self.x_count <= 501, "x_count must be in [2, 501]")
        else:
            require(bool(self.y_keys), f"{self.kind} needs at least one y_key")
            require(
                CATALOG[self.x_key].is_series,
                f"{self.kind} x_key must be a per-sample variable",
                self.x_key,
            )
            for key in self.y_keys:
                require(
                    CATALOG[key].is_series,
                    f"{self.kind} y_keys must be per-sample variables",
                    key,
                )

    def to_json_dict(self) -> dict[str, Any]:
        """The definition as a JSON-serialisable dictionary."""
        payload = asdict(self)
        payload["y_keys"] = list(self.y_keys)
        return {"format": SPEC_FORMAT, **payload}

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> PlotSpec:
        """Rebuild a definition from :meth:`to_json_dict` output.

        Args:
            data: The parsed JSON dictionary.

        Returns:
            The validated plot definition.
        """
        require(isinstance(data, dict), "plot definition must be an object", data)
        require(
            data.get("format") == SPEC_FORMAT,
            f"unsupported plot definition format {data.get('format')!r}",
        )
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        require(
            "kind" in kwargs and "x_key" in kwargs,
            "plot definition needs kind and x_key",
        )
        if "y_keys" in kwargs:
            kwargs["y_keys"] = tuple(kwargs["y_keys"])
        spec = cls(**kwargs)
        ensure(spec.to_json_dict()["format"] == SPEC_FORMAT, "round-trip failed")
        return spec


def spec_to_json(spec: PlotSpec, path: str | Path) -> None:
    """Write a plot definition to a ``.json`` file.

    Args:
        spec: The definition to save.
        path: Destination file path.
    """
    require(isinstance(spec, PlotSpec), "spec must be a PlotSpec", spec)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(spec.to_json_dict(), handle, indent=2)


def spec_from_json(path: str | Path) -> PlotSpec:
    """Load a plot definition from a ``.json`` file.

    Args:
        path: Source file path.

    Returns:
        The validated plot definition.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        return PlotSpec.from_json_dict(json.load(handle))
