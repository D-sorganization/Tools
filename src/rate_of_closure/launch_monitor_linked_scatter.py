"""Deterministic, bounded launch-monitor linked-scatter planning."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

from rate_of_closure.launch_monitor_numeric import finite_launch_monitor_scalar

MAX_DISPLAY_POINTS = 2_000
MAX_RETAINED_ROWS = 300_000
_IDENTITY_FIELDS = ("shot_id", "session_id", "monitor_vendor")


@dataclass(frozen=True)
class LinkedScatterPoint:
    """One finite displayed point bound to its retained raw row."""

    raw_index: int
    x: float
    y: float
    shot_id: str | None
    session_id: str | None
    monitor_vendor: str | None


@dataclass(frozen=True)
class LinkedScatterPlan:
    """Immutable bounded display plan; retained records remain untouched."""

    x_field: str
    y_field: str
    raw_count: int
    finite_count: int
    displayed_count: int
    selected_raw_index: int | None
    points: tuple[LinkedScatterPoint, ...]


@dataclass(frozen=True)
class PlotAxisProjection:
    """Overflow-safe normalized positions plus their declared raw scale."""

    coordinates: tuple[float, ...]
    scale: float


def project_plot_axis(values: Sequence[object]) -> PlotAxisProjection:
    """Map finite values to [-1, 1] without subtracting extreme raw values."""
    if not values:
        raise ValueError("plot axis values must be a nonempty float sequence")
    numeric_values: list[float] = []
    try:
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("plot axis values must be finite numbers")
            numeric_values.append(float(value))
    except OverflowError as error:
        raise ValueError("plot axis values must be finite") from error
    numeric = tuple(numeric_values)
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("plot axis values must be finite")
    scale = max(abs(value) for value in numeric)
    if scale == 0:
        return PlotAxisProjection(tuple(0.0 for _ in numeric), 1.0)
    low, high = min(numeric), max(numeric)
    if low == high:
        return PlotAxisProjection(tuple(0.0 for _ in values), scale)
    if low < 0 < high:
        basis = tuple(value / scale for value in numeric)
    else:
        basis = numeric
    basis_low, basis_high = min(basis), max(basis)
    span = basis_high - basis_low
    return PlotAxisProjection(
        tuple(2 * ((value - basis_low) / span) - 1 for value in basis), scale
    )


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def plan_linked_scatter(
    rows: Sequence[Mapping[str, Any]],
    x_field: str,
    y_field: str,
    *,
    selected_raw_index: int | None = None,
    cap: int = MAX_DISPLAY_POINTS,
) -> LinkedScatterPlan:
    """Plan finite points in O(raw) time and O(cap) output memory."""
    if (
        not isinstance(x_field, str)
        or not isinstance(y_field, str)
        or not x_field
        or not y_field
        or x_field == y_field
    ):
        raise ValueError("linked scatter requires two distinct field names")
    if not isinstance(rows, Sequence):
        raise ValueError("linked scatter rows must be a retained record sequence")
    if isinstance(cap, bool) or not isinstance(cap, int) or not 2 <= cap <= 2_000:
        raise ValueError("linked scatter cap must be an integer from 2 through 2000")
    row_count = len(rows)
    if row_count > MAX_RETAINED_ROWS:
        raise ValueError(f"linked scatter retains at most {MAX_RETAINED_ROWS} rows")
    if selected_raw_index is not None and (
        isinstance(selected_raw_index, bool)
        or not isinstance(selected_raw_index, int)
        or not 0 <= selected_raw_index < row_count
    ):
        raise ValueError("selected raw row index is outside the retained records")

    finite_count = 0
    selected_is_finite = False
    candidates: dict[int, LinkedScatterPoint] = {}
    selected_bucket: int | None = None
    for raw_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("each linked scatter row must be a record")
        x_value = finite_launch_monitor_scalar(row.get(x_field))
        y_value = finite_launch_monitor_scalar(row.get(y_field))
        if x_value is None or y_value is None:
            continue
        finite_count += 1
        bucket = raw_index if row_count <= cap else raw_index * cap // row_count
        if raw_index == selected_raw_index:
            candidates[bucket] = LinkedScatterPoint(
                raw_index=raw_index,
                x=x_value,
                y=y_value,
                **{field: _text(row.get(field)) for field in _IDENTITY_FIELDS},
            )
            selected_bucket = bucket
            selected_is_finite = True
        elif bucket not in candidates and bucket != selected_bucket:
            candidates[bucket] = LinkedScatterPoint(
                raw_index=raw_index,
                x=x_value,
                y=y_value,
                **{field: _text(row.get(field)) for field in _IDENTITY_FIELDS},
            )
    points = tuple(candidates[index] for index in sorted(candidates))
    return LinkedScatterPlan(
        x_field=x_field,
        y_field=y_field,
        raw_count=row_count,
        finite_count=finite_count,
        displayed_count=len(points),
        selected_raw_index=(selected_raw_index if selected_is_finite else None),
        points=points,
    )


NavigationCommand = Literal["previous", "next", "home", "end", "clear"]


def navigate_linked_scatter(
    plan: LinkedScatterPlan,
    current_raw_index: int | None,
    command: NavigationCommand,
) -> int | None:
    """Navigate only displayed raw rows; Escape maps to ``clear``."""
    if command not in {"previous", "next", "home", "end", "clear"}:
        raise ValueError("unknown linked scatter navigation command")
    indices = tuple(point.raw_index for point in plan.points)
    if command == "clear" or not indices:
        return None
    if command == "home":
        return indices[0]
    if command == "end":
        return indices[-1]
    if current_raw_index not in indices:
        return indices[0] if command == "next" else indices[-1]
    position = indices.index(current_raw_index)
    offset = 1 if command == "next" else -1
    return indices[(position + offset) % len(indices)]


__all__ = [
    "LinkedScatterPlan",
    "LinkedScatterPoint",
    "PlotAxisProjection",
    "MAX_DISPLAY_POINTS",
    "MAX_RETAINED_ROWS",
    "NavigationCommand",
    "navigate_linked_scatter",
    "plan_linked_scatter",
    "project_plot_axis",
]
