"""Immutable exact-sample plans for synchronized flight inspection."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal, cast

import numpy as np

from rate_of_closure.simulation.flight_explorer import FlightExploration

MAX_FLIGHT_SAMPLES = 1_002
MAX_FLIGHT_TIME_S = 10.001
MAX_FLIGHT_POSITION_M = 10_000.0
DEFAULT_FLIGHT_HIT_RADIUS_PX = 12.0

FlightPhase = Literal["launch", "ascent", "apex", "descent", "landing"]
FlightNavigation = Literal["previous", "next", "home", "end", "clear"]
FlightCohort = Literal["current"]


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite real number")
    return number


@dataclass(frozen=True)
class FlightSampleSeries:
    """Deep immutable snapshot of one runtime-local flight trajectory."""

    times_s: Sequence[float]
    positions_m: Sequence[Sequence[float]]

    def __post_init__(self) -> None:
        times_value: object = self.times_s
        positions_value: object = self.positions_m
        if not isinstance(times_value, Sequence) or isinstance(
            times_value, (str, bytes, bytearray)
        ):
            raise ValueError("times_s must be a sized numeric sequence")
        if not isinstance(positions_value, Sequence) or isinstance(
            positions_value, (str, bytes, bytearray)
        ):
            raise ValueError("positions_m must be a sized row sequence")
        times_input = cast(Sequence[object], times_value)
        positions_input = cast(Sequence[object], positions_value)
        count = len(times_input)
        if not 2 <= count <= MAX_FLIGHT_SAMPLES:
            raise ValueError(
                f"flight evidence must contain 2..{MAX_FLIGHT_SAMPLES} samples"
            )
        if len(positions_input) != count:
            raise ValueError("flight positions must have shape (N, 3)")
        checked_rows: list[Sequence[object]] = []
        for row in positions_input:
            if not isinstance(row, Sequence) or isinstance(
                row, (str, bytes, bytearray)
            ):
                raise ValueError("flight positions must have shape (N, 3)")
            if len(row) != 3:
                raise ValueError("flight positions must have shape (N, 3)")
            checked_rows.append(cast(Sequence[object], row))
        times = tuple(_finite(value, "times_s") for value in times_input)
        positions = tuple(
            tuple(_finite(value, "positions_m") for value in row)
            for row in checked_rows
        )
        if times[0] < 0 or any(
            right <= left for left, right in zip(times, times[1:], strict=False)
        ):
            raise ValueError(
                "flight sample times must be nonnegative and strictly increasing"
            )
        if times[-1] > MAX_FLIGHT_TIME_S:
            raise ValueError("flight sample time exceeds the explorer contract")
        if any(
            abs(value) > MAX_FLIGHT_POSITION_M for row in positions for value in row
        ):
            raise ValueError("flight position exceeds the explorer contract")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "positions_m", positions)

    @classmethod
    def from_exploration(cls, exploration: FlightExploration) -> FlightSampleSeries:
        """Copy a flight result before presentation publication."""
        if not isinstance(exploration, FlightExploration):
            raise ValueError("flight sample source must be a FlightExploration")
        if not isinstance(exploration.times, np.ndarray) or not isinstance(
            exploration.positions, np.ndarray
        ):
            raise ValueError("flight sample arrays must be NumPy arrays")
        times = exploration.times
        positions = exploration.positions
        if (
            times.ndim != 1
            or positions.ndim != 2
            or positions.shape[1:] != (3,)
            or not 2 <= times.shape[0] <= MAX_FLIGHT_SAMPLES
            or positions.shape[0] != times.shape[0]
        ):
            raise ValueError("flight sample arrays have invalid dimensions")
        return cls(times.tolist(), positions.tolist())


@dataclass(frozen=True)
class FlightDisplaySample:
    """One exact raw solver sample used by both profile markers."""

    raw_index: int
    time_s: float
    downrange_m: float
    height_m: float
    right_m: float
    phase: FlightPhase


@dataclass(frozen=True)
class FlightSampleSelection:
    """Exact cohort-local identity; calm comparison is deliberately not selectable."""

    cohort: FlightCohort
    raw_index: int

    def __post_init__(self) -> None:
        if self.cohort != "current":
            raise ValueError("only the current primary flight is selectable")
        if isinstance(self.raw_index, bool) or not isinstance(self.raw_index, int):
            raise ValueError("raw sample index must be an integer")
        if self.raw_index < 0:
            raise ValueError("raw sample index must be nonnegative")


@dataclass(frozen=True)
class FlightSamplePlan:
    """Fixed exact geometry and phase authority for one accepted flight."""

    series: FlightSampleSeries
    samples: tuple[FlightDisplaySample, ...]
    apex_raw_index: int

    @property
    def raw_count(self) -> int:
        return len(self.samples)

    @property
    def raw_indices(self) -> tuple[int, ...]:
        return tuple(range(self.raw_count))

    def raw_sample(self, raw_index: int) -> FlightDisplaySample:
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError("raw sample index must be an integer")
        if not 0 <= raw_index < self.raw_count:
            raise ValueError("raw sample index is outside the accepted flight")
        return self.samples[raw_index]


def plan_flight_samples(series: FlightSampleSeries) -> FlightSamplePlan:
    """Build one immutable O(raw) plan without interpolation or decimation."""
    if not isinstance(series, FlightSampleSeries):
        raise ValueError("flight planner requires validated sample evidence")
    heights = tuple(row[1] for row in series.positions_m)
    apex = max(range(len(heights)), key=lambda index: (heights[index], -index))
    last = len(heights) - 1
    samples: list[FlightDisplaySample] = []
    for index, (time_s, position) in enumerate(
        zip(series.times_s, series.positions_m, strict=True)
    ):
        # Coincident events use launch > landing > apex precedence. Normal
        # flights still expose an interior apex; a two-sample descending or
        # monotone-ascent edge honestly cannot display every event label.
        if index == 0:
            phase: FlightPhase = "launch"
        elif index == last:
            phase = "landing"
        elif index == apex:
            phase = "apex"
        elif index < apex:
            phase = "ascent"
        else:
            phase = "descent"
        samples.append(
            FlightDisplaySample(
                index,
                time_s,
                position[0],
                position[1],
                position[2],
                phase,
            )
        )
    return FlightSamplePlan(series, tuple(samples), apex)


def navigate_flight_samples(
    plan: FlightSamplePlan, current_raw_index: int | None, command: FlightNavigation
) -> int | None:
    """Navigate exact runtime-local raw samples only."""
    if command not in {"previous", "next", "home", "end", "clear"}:
        raise ValueError("unknown flight sample navigation command")
    if command == "clear":
        return None
    if command == "home":
        return 0
    if command == "end":
        return plan.raw_count - 1
    if not isinstance(current_raw_index, int) or isinstance(current_raw_index, bool):
        return 0 if command == "next" else plan.raw_count - 1
    if not 0 <= current_raw_index < plan.raw_count:
        return 0 if command == "next" else plan.raw_count - 1
    delta = 1 if command == "next" else -1
    return min(max(current_raw_index + delta, 0), plan.raw_count - 1)


def nearest_flight_sample(
    plan: FlightSamplePlan,
    projected: Sequence[tuple[object, object, object, object]],
    pointer_px: tuple[object, object],
    *,
    hit_radius_px: object = DEFAULT_FLIGHT_HIT_RADIUS_PX,
) -> FlightSampleSelection | None:
    """Pick the closest exact raw sample in rendered pixel space."""
    if len(pointer_px) != 2:
        raise ValueError("pointer must contain two finite pixel coordinates")
    pointer_x, pointer_y = (_finite(value, "pointer") for value in pointer_px)
    radius = _finite(hit_radius_px, "hit radius")
    if not 0 < radius <= 100:
        raise ValueError("hit radius must be a positive pixel distance")
    candidates: list[tuple[float, int]] = []
    seen: set[int] = set()
    for item in projected:
        if len(item) != 4:
            raise ValueError("projected samples require cohort, index, and two pixels")
        cohort, raw_index, raw_x, raw_y = item
        if cohort != "current":
            raise ValueError("calm comparison samples are not selectable")
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, int)
            or raw_index < 0
        ):
            raise ValueError("projected raw index must be a nonnegative integer")
        if raw_index >= plan.raw_count or raw_index in seen:
            raise ValueError("projected raw indices must be unique and in range")
        seen.add(raw_index)
        x, y = _finite(raw_x, "projected x"), _finite(raw_y, "projected y")
        candidates.append((math.hypot(x - pointer_x, y - pointer_y), raw_index))
    if seen != set(plan.raw_indices):
        raise ValueError("projected samples must cover the complete primary plan")
    distance, raw_index = min(candidates, key=lambda item: (item[0], item[1]))
    return FlightSampleSelection("current", raw_index) if distance <= radius else None
