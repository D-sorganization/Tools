"""Immutable exact-point and derived-bin plans for managed plot inspection."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sized
from dataclasses import dataclass
from numbers import Real
from typing import Literal, cast

MAX_PLOT_SAMPLES = 8_192
MAX_PLOT_SERIES = 8
MAX_PLOT_VERTICES = 8_192
MAX_ABS_PLOT_VALUE = 1.0e12
DEFAULT_PLOT_HIT_RADIUS_PX = 12.0

PlotNavigation = Literal["previous", "next", "up", "down", "home", "end", "clear"]


def _finite(value: object, field: str, *, pixel: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite bounded real number")
    number = float(value)
    limit = 1.0e9 if pixel else MAX_ABS_PLOT_VALUE
    if not math.isfinite(number) or abs(number) > limit:
        raise ValueError(f"{field} must be a finite bounded real number")
    return number


def _snapshot_values(value: object, field: str) -> tuple[float, ...]:
    if (
        isinstance(value, (str, bytes, bytearray))
        or not isinstance(value, Sized)
        or not isinstance(value, Iterable)
    ):
        raise ValueError(f"{field} must be a sized numeric sequence")
    count = len(value)
    if not 1 <= count <= MAX_PLOT_SAMPLES:
        raise ValueError(f"plot evidence must contain 1..{MAX_PLOT_SAMPLES} samples")
    return tuple(_finite(item, field) for item in cast(Iterable[object], value))


@dataclass(frozen=True)
class PlotSeries:
    label: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class HistogramBin:
    index: int
    lower: float
    upper: float
    count: int


@dataclass(frozen=True)
class PlotInspectionPlan:
    kind: Literal["series", "histogram"]
    x: tuple[float, ...]
    series: tuple[PlotSeries, ...]
    bins: tuple[HistogramBin, ...]

    @property
    def raw_count(self) -> int:
        return len(self.x)


@dataclass(frozen=True)
class SeriesSelection:
    kind: Literal["series"]
    series_index: int
    raw_index: int

    def __init__(self, series_index: int, raw_index: int) -> None:
        object.__setattr__(self, "kind", "series")
        object.__setattr__(self, "series_index", series_index)
        object.__setattr__(self, "raw_index", raw_index)
        if any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in (series_index, raw_index)
        ):
            raise ValueError("series and raw indices must be integers")
        if series_index < 0 or raw_index < 0:
            raise ValueError("series and raw indices must be nonnegative")


@dataclass(frozen=True)
class HistogramSelection:
    kind: Literal["histogram"]
    bin_index: int

    def __init__(self, bin_index: int) -> None:
        object.__setattr__(self, "kind", "histogram")
        object.__setattr__(self, "bin_index", bin_index)
        if (
            isinstance(bin_index, bool)
            or not isinstance(bin_index, int)
            or bin_index < 0
        ):
            raise ValueError("histogram bin index must be a nonnegative integer")


PlotSelection = SeriesSelection | HistogramSelection


def _histogram_bins(x: tuple[float, ...]) -> tuple[HistogramBin, ...]:
    count = min(40, max(10, len(x) // 10))
    low, high = min(x), max(x)
    if low == high:
        low -= 0.5
        high += 0.5
    width = (high - low) / count
    counts = [0] * count
    for value in x:
        index = count - 1 if value == high else int((value - low) / width)
        counts[min(max(index, 0), count - 1)] += 1
    return tuple(
        HistogramBin(index, low + index * width, low + (index + 1) * width, value)
        for index, value in enumerate(counts)
    )


def plan_plot_inspection(
    kind: object, x_input: object, series_input: object
) -> PlotInspectionPlan:
    """Validate and deep-snapshot one exact plot evidence cohort."""
    if kind not in {"line", "scatter", "sweep", "histogram"}:
        raise ValueError("plot kind is not inspectable")
    x = _snapshot_values(x_input, "x")
    if not isinstance(series_input, (list, tuple)):
        raise ValueError("plot series must be a sized sequence")
    if kind == "histogram":
        if series_input:
            raise ValueError("histogram evidence must not contain y series")
        return PlotInspectionPlan("histogram", x, (), _histogram_bins(x))
    if not 1 <= len(series_input) <= MAX_PLOT_SERIES:
        raise ValueError(f"plot must contain 1..{MAX_PLOT_SERIES} series")
    if len(x) * len(series_input) > MAX_PLOT_VERTICES:
        raise ValueError(f"plot exceeds {MAX_PLOT_VERTICES} inspectable vertices")
    series: list[PlotSeries] = []
    for item in series_input:
        if not isinstance(item, Mapping):
            raise ValueError("plot series must contain label and values")
        label, values_input = item.get("label"), item.get("values")
        if not isinstance(label, str) or not label or len(label) > 512:
            raise ValueError("plot series label must contain 1..512 characters")
        values = _snapshot_values(values_input, "series values")
        if len(values) != len(x):
            raise ValueError("plot series values must align with x")
        series.append(PlotSeries(label, values))
    return PlotInspectionPlan("series", x, tuple(series), ())


def _validate_selection(
    plan: PlotInspectionPlan, selection: PlotSelection | None
) -> PlotSelection | None:
    if selection is None:
        return None
    if plan.kind == "series" and isinstance(selection, SeriesSelection):
        if selection.series_index < len(plan.series) and selection.raw_index < len(
            plan.x
        ):
            return selection
    if plan.kind == "histogram" and isinstance(selection, HistogramSelection):
        if selection.bin_index < len(plan.bins):
            return selection
    raise ValueError("selection is outside the inspection plan")


def navigate_plot_selection(
    plan: PlotInspectionPlan,
    current: PlotSelection | None,
    command: PlotNavigation,
) -> PlotSelection | None:
    """Navigate exact samples or exact derived bins without interpolation."""
    if command not in {"previous", "next", "up", "down", "home", "end", "clear"}:
        raise ValueError("unknown plot navigation command")
    current = _validate_selection(plan, current)
    if command == "clear":
        return None
    if plan.kind == "histogram":
        index = current.bin_index if isinstance(current, HistogramSelection) else None
        if command == "home":
            return HistogramSelection(0)
        if command == "end":
            return HistogramSelection(len(plan.bins) - 1)
        if index is None:
            return HistogramSelection(
                0 if command in {"next", "down"} else len(plan.bins) - 1
            )
        delta = 1 if command in {"next", "down"} else -1
        return HistogramSelection(min(max(index + delta, 0), len(plan.bins) - 1))
    item = current if isinstance(current, SeriesSelection) else None
    if item is None:
        if command in {"previous", "up", "end"}:
            return SeriesSelection(len(plan.series) - 1, len(plan.x) - 1)
        return SeriesSelection(0, 0)
    if command == "home":
        return SeriesSelection(item.series_index, 0)
    if command == "end":
        return SeriesSelection(item.series_index, len(plan.x) - 1)
    if command in {"up", "down"}:
        delta = -1 if command == "up" else 1
        return SeriesSelection(
            min(max(item.series_index + delta, 0), len(plan.series) - 1), item.raw_index
        )
    delta = -1 if command == "previous" else 1
    return SeriesSelection(
        item.series_index, min(max(item.raw_index + delta, 0), len(plan.x) - 1)
    )


def nearest_series_point(
    plan: PlotInspectionPlan,
    projected: object,
    pointer_px: object,
    hit_radius_px: object = DEFAULT_PLOT_HIT_RADIUS_PX,
) -> SeriesSelection | None:
    """Pick the closest exact series point in rendered pixel space."""
    if plan.kind != "series":
        raise ValueError("series picking requires a series plan")
    if not isinstance(projected, (list, tuple)) or len(projected) != len(plan.series):
        raise ValueError("projected series must match the complete inspection plan")
    if not isinstance(pointer_px, (list, tuple)) or len(pointer_px) != 2:
        raise ValueError("pointer must contain two pixel coordinates")
    pointer_x, pointer_y = (
        _finite(value, "pointer", pixel=True) for value in pointer_px
    )
    radius = _finite(hit_radius_px, "hit radius", pixel=True)
    if not 0 < radius <= 100:
        raise ValueError("hit radius must be a positive pixel distance")
    nearest: tuple[float, int, int] | None = None
    for series_index, points in enumerate(projected):
        if not isinstance(points, (list, tuple)) or len(points) != len(plan.x):
            raise ValueError("projected series must match the complete inspection plan")
        for raw_index, point in enumerate(points):
            if (
                isinstance(point, (str, bytes, bytearray))
                or not isinstance(point, Sized)
                or not isinstance(point, Iterable)
                or len(point) != 2
            ):
                raise ValueError("projected point must contain two pixel coordinates")
            x, y = (
                _finite(value, "projected point", pixel=True)
                for value in cast(Iterable[object], point)
            )
            candidate = (
                math.hypot(x - pointer_x, y - pointer_y),
                series_index,
                raw_index,
            )
            if nearest is None or candidate < nearest:
                nearest = candidate
    assert nearest is not None
    return SeriesSelection(nearest[1], nearest[2]) if nearest[0] <= radius else None


def histogram_bin_at_data(
    plan: PlotInspectionPlan, x_value: object, y_value: object
) -> HistogramSelection | None:
    """Select a derived bin only when a data-space point is inside its bar."""
    if plan.kind != "histogram":
        raise ValueError("histogram picking requires a histogram plan")
    x = _finite(x_value, "histogram pointer")
    y = _finite(y_value, "histogram pointer")
    if y < 0:
        return None
    for item in plan.bins:
        in_x = item.lower <= x < item.upper or (
            item.index == len(plan.bins) - 1 and x == item.upper
        )
        if in_x:
            return HistogramSelection(item.index) if y <= item.count else None
    return None


__all__ = [
    "DEFAULT_PLOT_HIT_RADIUS_PX",
    "HistogramBin",
    "HistogramSelection",
    "MAX_PLOT_SAMPLES",
    "MAX_PLOT_SERIES",
    "MAX_PLOT_VERTICES",
    "PlotInspectionPlan",
    "PlotNavigation",
    "PlotSelection",
    "PlotSeries",
    "SeriesSelection",
    "histogram_bin_at_data",
    "navigate_plot_selection",
    "nearest_series_point",
    "plan_plot_inspection",
]
