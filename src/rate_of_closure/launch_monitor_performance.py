"""Descriptive performance metrics behind a swappable analytics adapter.

These calculations are deliberately limited to unit conversion, aggregation,
and score bookkeeping. Inferential statistics remain an UpstreamDrift concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, sqrt
from urllib.parse import urlparse

import numpy as np
import pandas as pd

YARDS_PER_METRE = 1.0936132983377078
DistanceUnit = str


def _to_yards(values: pd.Series, unit: DistanceUnit) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if unit == "yd":
        return numeric
    if unit == "m":
        return numeric * YARDS_PER_METRE
    raise ValueError("distance unit must be 'yd' or 'm'")


def _columns(frame: pd.DataFrame, *names: str) -> None:
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"columns are unavailable: {sorted(missing)}")


@dataclass(frozen=True)
class DispersionRequest:
    lateral_column: str
    carry_column: str
    lateral_unit: DistanceUnit
    carry_unit: DistanceUnit


@dataclass(frozen=True)
class DispersionPoint:
    source_index: int
    lateral_yards: float
    carry_yards: float


@dataclass(frozen=True)
class DispersionResult:
    unit: str
    points: tuple[DispersionPoint, ...]
    mean_lateral_yards: float
    standard_deviation_yards: float
    rms_yards: float
    left_count: int
    center_count: int
    right_count: int
    formula: str


def analyze_dispersion(
    frame: pd.DataFrame, request: DispersionRequest
) -> DispersionResult:
    """Summarize carry/lateral points with negative=left, positive=right."""

    _columns(frame, request.lateral_column, request.carry_column)
    lateral = _to_yards(frame[request.lateral_column], request.lateral_unit)
    carry = _to_yards(frame[request.carry_column], request.carry_unit)
    complete = pd.DataFrame({"lateral": lateral, "carry": carry}).dropna()
    if complete.empty:
        raise ValueError("dispersion requires finite lateral and carry values")
    values = complete["lateral"].to_numpy(float)
    points = tuple(
        DispersionPoint(index, float(lateral_value), float(carry_value))
        for index, (lateral_value, carry_value) in enumerate(
            zip(
                complete["lateral"].to_numpy(float),
                complete["carry"].to_numpy(float),
                strict=True,
            )
        )
    )
    standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return DispersionResult(
        unit="yd",
        points=points,
        mean_lateral_yards=float(np.mean(values)),
        standard_deviation_yards=standard_deviation,
        rms_yards=float(sqrt(float(np.mean(np.square(values))))),
        left_count=int(np.sum(values < 0)),
        center_count=int(np.sum(values == 0)),
        right_count=int(np.sum(values > 0)),
        formula=(
            "Lateral sign: negative = yards left, positive = yards right. "
            "RMS = sqrt(mean(lateral_yards^2))."
        ),
    )


@dataclass(frozen=True)
class StrokesGainedRequest:
    expected_before_column: str
    expected_after_column: str
    baseline_source_url: str


@dataclass(frozen=True)
class ScoreResult:
    metric_name: str
    unit: str
    values: tuple[float, ...]
    mean: float
    formula: str
    source_url: str | None


def calculate_strokes_gained(
    frame: pd.DataFrame, request: StrokesGainedRequest
) -> ScoreResult:
    """Compute user-supplied expected-strokes SG without validating a baseline."""

    parsed = urlparse(request.baseline_source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("strokes gained requires an HTTP(S) baseline source URL")
    _columns(frame, request.expected_before_column, request.expected_after_column)
    before = pd.to_numeric(frame[request.expected_before_column], errors="coerce")
    after = pd.to_numeric(frame[request.expected_after_column], errors="coerce")
    values = (before - 1.0 - after).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        raise ValueError("strokes gained requires finite expected-stroke state")
    result = tuple(float(value) for value in values)
    return ScoreResult(
        "user_supplied_expected_strokes_sg",
        "strokes",
        result,
        float(np.mean(result)),
        "User-supplied expected-strokes SG = E(before) - 1 - E(after); "
        "the app did not reproduce or validate the cited baseline table.",
        request.baseline_source_url,
    )


@dataclass(frozen=True)
class TargetErrorRequest:
    carry_column: str
    lateral_column: str
    carry_unit: DistanceUnit
    lateral_unit: DistanceUnit
    target_distance_yards: float


def calculate_target_error(
    frame: pd.DataFrame, request: TargetErrorRequest
) -> ScoreResult:
    """Compute radial target error; this is intentionally not strokes gained."""

    if request.target_distance_yards <= 0:
        raise ValueError("target distance must be positive")
    _columns(frame, request.carry_column, request.lateral_column)
    carry = _to_yards(frame[request.carry_column], request.carry_unit)
    lateral = _to_yards(frame[request.lateral_column], request.lateral_unit)
    complete = pd.DataFrame({"carry": carry, "lateral": lateral}).dropna()
    if complete.empty:
        raise ValueError("target error requires finite carry and lateral values")
    values = tuple(
        hypot(request.target_distance_yards - carry_value, lateral_value)
        for carry_value, lateral_value in zip(
            complete["carry"].to_numpy(float),
            complete["lateral"].to_numpy(float),
            strict=True,
        )
    )
    return ScoreResult(
        "radial_target_error",
        "yd",
        values,
        float(np.mean(values)),
        "radial_target_error = hypot(target_yards - carry_yards, lateral_yards)",
        None,
    )


@dataclass(frozen=True)
class TrendRequest:
    metric_column: str
    session_column: str
    session_order_column: str
    player_column: str
    player_identity_attested: bool
    session_identity_attested: bool


@dataclass(frozen=True)
class TrendPoint:
    player_id: str
    session_id: str
    session_order: float
    sample_count: int
    mean: float
    cumulative_mean: float


@dataclass(frozen=True)
class TrendResult:
    metric: str
    points: tuple[TrendPoint, ...]
    formula: str


def analyze_session_trend(frame: pd.DataFrame, request: TrendRequest) -> TrendResult:
    """Return equal-session-weighted player trends with explicit identity/order."""

    if not request.player_identity_attested or not request.session_identity_attested:
        raise ValueError("player and session identity must both be explicitly attested")
    _columns(
        frame,
        request.metric_column,
        request.session_column,
        request.session_order_column,
        request.player_column,
    )
    selected = frame[
        [
            request.player_column,
            request.session_column,
            request.session_order_column,
            request.metric_column,
        ]
    ].copy()
    selected = selected.set_axis(
        ["player", "session", "order", "metric"], axis="columns"
    )
    selected["order"] = pd.to_numeric(selected["order"], errors="coerce")
    selected["metric"] = pd.to_numeric(selected["metric"], errors="coerce")
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna()
    selected["player"] = selected["player"].astype(str).str.strip()
    selected["session"] = selected["session"].astype(str).str.strip()
    selected = selected[(selected["player"] != "") & (selected["session"] != "")]
    if selected.empty:
        raise ValueError(
            "trend requires finite rows with trusted player/session identity"
        )
    order_counts = selected.groupby(["player", "session"])["order"].nunique()
    if bool((order_counts != 1).any()):
        raise ValueError("each player session must map to exactly one order value")
    summaries = (
        selected.groupby(["player", "session"], as_index=False)
        .agg(
            order=("order", "first"), mean=("metric", "mean"), count=("metric", "size")
        )
        .sort_values(["player", "order", "session"], kind="stable")
    )
    summaries["cumulative"] = (
        summaries.groupby("player")["mean"].expanding().mean().droplevel(0)
    )
    points = tuple(
        TrendPoint(player, session, order, count, mean, cumulative)
        for player, session, order, count, mean, cumulative in zip(
            summaries["player"].astype(str).tolist(),
            summaries["session"].astype(str).tolist(),
            summaries["order"].to_numpy(float).tolist(),
            summaries["count"].to_numpy(int).tolist(),
            summaries["mean"].to_numpy(float).tolist(),
            summaries["cumulative"].to_numpy(float).tolist(),
            strict=True,
        )
    )
    return TrendResult(
        request.metric_column,
        points,
        "Session mean = mean(metric rows); cumulative mean = equal-weight "
        "mean of session means.",
    )


__all__ = [
    "DispersionRequest",
    "DispersionResult",
    "ScoreResult",
    "StrokesGainedRequest",
    "TargetErrorRequest",
    "TrendRequest",
    "TrendResult",
    "analyze_dispersion",
    "analyze_session_trend",
    "calculate_strokes_gained",
    "calculate_target_error",
]
