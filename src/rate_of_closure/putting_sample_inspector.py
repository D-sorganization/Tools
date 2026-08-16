"""Immutable, bounded presentation plans for putting sample inspection."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal

from shared.python.swing_sim.putting import PuttResult

MAX_PUTTING_RAW_SAMPLES = 30_001
MAX_PUTTING_DISPLAY_SAMPLES = 1_024
DEFAULT_HIT_RADIUS_PX = 12.0

PuttingPhase = Literal["skid", "pure-roll"]
PuttingNavigation = Literal["previous", "next", "home", "end", "clear"]


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be finite")
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def _finite_tuple(values: object, field: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a numeric sequence")
    result: list[float] = []
    for value in values:
        result.append(_finite_number(value, f"{field} values"))
    return tuple(result)


@dataclass(frozen=True)
class PuttingSampleSeries:
    """Validated aligned raw evidence from one accepted putt result."""

    path_x_m: Sequence[float]
    path_y_m: Sequence[float]
    speeds_mps: Sequence[float]
    times_s: Sequence[float]
    skid_end_index: int

    def __post_init__(self) -> None:
        arrays = {
            "path_x_m": _finite_tuple(self.path_x_m, "path_x_m"),
            "path_y_m": _finite_tuple(self.path_y_m, "path_y_m"),
            "speeds_mps": _finite_tuple(self.speeds_mps, "speeds_mps"),
            "times_s": _finite_tuple(self.times_s, "times_s"),
        }
        count = len(arrays["times_s"])
        if not 1 <= count <= MAX_PUTTING_RAW_SAMPLES:
            raise ValueError(
                f"putting evidence must contain 1..{MAX_PUTTING_RAW_SAMPLES} samples"
            )
        if any(len(values) != count for values in arrays.values()):
            raise ValueError("putting sample arrays must have equal lengths")
        if any(speed < 0 for speed in arrays["speeds_mps"]):
            raise ValueError("putting sample speeds must be nonnegative")
        times = arrays["times_s"]
        if times[0] < 0 or any(
            right <= left for left, right in zip(times, times[1:], strict=False)
        ):
            raise ValueError(
                "putting sample times must be nonnegative and strictly increasing"
            )
        split = self.skid_end_index
        if (
            isinstance(split, bool)
            or not isinstance(split, int)
            or not 0 <= split < count
        ):
            raise ValueError("skid_end_index must identify the first pure-roll sample")
        for field, values in arrays.items():
            object.__setattr__(self, field, values)

    @classmethod
    def from_result(cls, result: PuttResult) -> PuttingSampleSeries:
        """Copy one immutable physics result into the presentation boundary."""
        if not isinstance(result, PuttResult):
            raise ValueError("putting sample source must be a PuttResult")
        return cls(
            path_x_m=result.path_x_m,
            path_y_m=result.path_y_m,
            speeds_mps=result.speeds_mps,
            times_s=result.times_s,
            skid_end_index=result.skid_end_index,
        )


@dataclass(frozen=True)
class PuttingDisplaySample:
    """One exact raw solver sample retained in the bounded display plan."""

    raw_index: int
    time_s: float
    cumulative_distance_m: float
    x_m: float
    y_m: float
    speed_mps: float
    phase: PuttingPhase


@dataclass(frozen=True)
class PuttingSamplePlan:
    """Fixed immutable geometry and exact raw evidence for one result."""

    series: PuttingSampleSeries
    cumulative_distance_m: tuple[float, ...]
    samples: tuple[PuttingDisplaySample, ...]

    @property
    def raw_count(self) -> int:
        """Number of retained solver samples."""
        return len(self.series.times_s)

    @property
    def displayed_count(self) -> int:
        """Number of exact samples in the bounded display plan."""
        return len(self.samples)

    @property
    def skid_end_index(self) -> int:
        """First raw sample in pure roll."""
        return self.series.skid_end_index

    @property
    def displayed_raw_indices(self) -> tuple[int, ...]:
        """Stable raw indices rendered by both synchronized plots."""
        return tuple(sample.raw_index for sample in self.samples)

    @property
    def skid_polyline_indices(self) -> tuple[int, ...]:
        """Displayed skid indices plus the split for line continuity."""
        if self.skid_end_index == 0:
            return ()
        return tuple(
            index
            for index in self.displayed_raw_indices
            if index <= self.skid_end_index
        )

    @property
    def pure_roll_polyline_indices(self) -> tuple[int, ...]:
        """Displayed pure-roll indices, beginning at the authoritative split."""
        return tuple(
            index
            for index in self.displayed_raw_indices
            if index >= self.skid_end_index
        )

    def raw_sample(self, raw_index: int) -> PuttingDisplaySample:
        """Return exact evidence for a validated raw index."""
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError("raw sample index must be an integer")
        if not 0 <= raw_index < self.raw_count:
            raise ValueError("raw sample index is outside the accepted result")
        series = self.series
        return PuttingDisplaySample(
            raw_index=raw_index,
            time_s=series.times_s[raw_index],
            cumulative_distance_m=self.cumulative_distance_m[raw_index],
            x_m=series.path_x_m[raw_index],
            y_m=series.path_y_m[raw_index],
            speed_mps=series.speeds_mps[raw_index],
            phase="skid" if raw_index < series.skid_end_index else "pure-roll",
        )


def _stable_extrema(values: Sequence[float]) -> tuple[int, int]:
    indices = range(len(values))
    return min(indices, key=lambda index: (values[index], index)), max(
        indices, key=lambda index: (values[index], -index)
    )


def _display_indices(
    series: PuttingSampleSeries, cap: int, mandatory: set[int] | None = None
) -> tuple[int, ...]:
    count = len(series.times_s)
    if mandatory is None:
        mandatory = {0, count - 1, series.skid_end_index}
        for values in (series.path_x_m, series.path_y_m, series.speeds_mps):
            mandatory.update(_stable_extrema(values))
    if len(mandatory) > cap:
        raise ValueError("putting display cap cannot retain all scientific landmarks")
    if count <= cap:
        return tuple(range(count))
    available = [index for index in range(count) if index not in mandatory]
    needed = cap - len(mandatory)
    if needed == 1:
        selected = [available[len(available) // 2]]
    elif needed > 1:
        selected = [
            available[
                (2 * position * (len(available) - 1) + (needed - 1))
                // (2 * (needed - 1))
            ]
            for position in range(needed)
        ]
    else:
        selected = []
    return tuple(sorted(mandatory.union(selected)))


def plan_putting_samples(
    series: PuttingSampleSeries, *, cap: int = MAX_PUTTING_DISPLAY_SAMPLES
) -> PuttingSamplePlan:
    """Build one O(raw) immutable plan; selection never rebuilds it."""
    if not isinstance(series, PuttingSampleSeries):
        raise ValueError("putting planner requires validated sample evidence")
    if isinstance(cap, bool) or not isinstance(cap, int) or not 3 <= cap <= 1_024:
        raise ValueError("putting display cap must be an integer from 3 through 1024")
    cumulative = [0.0]
    min_x = max_x = series.path_x_m[0]
    max_abs_y = abs(series.path_y_m[0])
    max_speed = series.speeds_mps[0]
    extrema = [[0, 0], [0, 0], [0, 0]]
    for index in range(1, len(series.times_s)):
        step = math.hypot(
            series.path_x_m[index] - series.path_x_m[index - 1],
            series.path_y_m[index] - series.path_y_m[index - 1],
        )
        next_distance = cumulative[-1] + step
        if not math.isfinite(step) or not math.isfinite(next_distance):
            raise ValueError("putting cumulative distance must remain finite")
        cumulative.append(next_distance)
        min_x = min(min_x, series.path_x_m[index])
        max_x = max(max_x, series.path_x_m[index])
        max_abs_y = max(max_abs_y, abs(series.path_y_m[index]))
        max_speed = max(max_speed, series.speeds_mps[index])
        x_value, y_value, speed_value = (
            series.path_x_m[index],
            series.path_y_m[index],
            series.speeds_mps[index],
        )
        if x_value < series.path_x_m[extrema[0][0]]:
            extrema[0][0] = index
        if x_value > series.path_x_m[extrema[0][1]]:
            extrema[0][1] = index
        if y_value < series.path_y_m[extrema[1][0]]:
            extrema[1][0] = index
        if y_value > series.path_y_m[extrema[1][1]]:
            extrema[1][1] = index
        if speed_value < series.speeds_mps[extrema[2][0]]:
            extrema[2][0] = index
        if speed_value > series.speeds_mps[extrema[2][1]]:
            extrema[2][1] = index
    cumulative_tuple = tuple(cumulative)
    display_envelope = (
        max_x + 0.3 - (min_x - 0.3),
        2.0 * max(0.3, max_abs_y),
        max_speed * 1.08,
    )
    if any(not math.isfinite(value) or value <= 0 for value in display_envelope):
        raise ValueError("putting display envelope must remain finite and positive")
    mandatory = {0, len(series.times_s) - 1, series.skid_end_index}
    mandatory.update(index for pair in extrema for index in pair)
    indices = _display_indices(series, cap, mandatory)
    placeholder = PuttingSamplePlan(series, cumulative_tuple, ())
    samples = tuple(placeholder.raw_sample(index) for index in indices)
    return PuttingSamplePlan(series, cumulative_tuple, samples)


def navigate_putting_samples(
    plan: PuttingSamplePlan,
    current_raw_index: int | None,
    command: PuttingNavigation,
) -> int | None:
    """Move only among exact samples in the fixed display plan."""
    if command not in {"previous", "next", "home", "end", "clear"}:
        raise ValueError("unknown putting sample navigation command")
    indices = plan.displayed_raw_indices
    if command == "clear" or not indices:
        return None
    if command == "home":
        return indices[0]
    if command == "end":
        return indices[-1]
    if current_raw_index not in indices:
        return indices[0] if command == "next" else indices[-1]
    position = indices.index(current_raw_index)
    if command == "next":
        return indices[min(position + 1, len(indices) - 1)]
    return indices[max(position - 1, 0)]


def nearest_putting_sample(
    projected: Sequence[tuple[object, object, object]],
    pointer_px: tuple[object, object],
    *,
    hit_radius_px: object = DEFAULT_HIT_RADIUS_PX,
) -> int | None:
    """Return the nearest displayed raw index in rendered pixel space."""
    if len(pointer_px) != 2:
        raise ValueError("pointer must contain two finite pixel coordinates")
    pointer_x = _finite_number(pointer_px[0], "pointer x")
    pointer_y = _finite_number(pointer_px[1], "pointer y")
    radius = _finite_number(hit_radius_px, "hit radius")
    if not 0 < radius <= 100:
        raise ValueError("hit radius must be a finite positive pixel distance")
    candidates: list[tuple[float, int]] = []
    for item in projected:
        if len(item) != 3:
            raise ValueError("projected sample must contain raw index and two pixels")
        raw_index, x_value, y_value = item
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError("projected samples must contain finite pixel coordinates")
        x_pixel = _finite_number(x_value, "projected x")
        y_pixel = _finite_number(y_value, "projected y")
        distance_squared = (x_pixel - pointer_x) ** 2 + (y_pixel - pointer_y) ** 2
        candidates.append((distance_squared, raw_index))
    if not candidates:
        return None
    distance_squared, raw_index = min(candidates, key=lambda item: (item[0], item[1]))
    return raw_index if distance_squared <= radius**2 else None


__all__ = [
    "DEFAULT_HIT_RADIUS_PX",
    "MAX_PUTTING_DISPLAY_SAMPLES",
    "MAX_PUTTING_RAW_SAMPLES",
    "PuttingDisplaySample",
    "PuttingNavigation",
    "PuttingSamplePlan",
    "PuttingSampleSeries",
    "navigate_putting_samples",
    "nearest_putting_sample",
    "plan_putting_samples",
]
