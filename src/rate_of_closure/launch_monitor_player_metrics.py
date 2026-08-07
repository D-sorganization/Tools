"""Transparent dispersion, strokes-gained, and longitudinal player metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rate_of_closure.launch_monitor_data import infer_unit

METERS_TO_YARDS = 1.0936133
BROADIE_SOURCE_URL = (
    "https://www.columbia.edu/~mnb2/broadie/Assets/"
    "strokes_gained_pga_broadie_20110408.pdf"
)

# Table 9, estimated from more than eight million PGA TOUR shots, 2003-2010.
_BROADIE_ROWS = (
    (10, None, 2.18, 2.34, 2.43, 3.45),
    (20, None, 2.40, 2.59, 2.53, 3.51),
    (30, None, 2.52, 2.70, 2.66, 3.57),
    (40, None, 2.60, 2.78, 2.82, 3.71),
    (50, None, 2.66, 2.87, 2.92, 3.79),
    (60, None, 2.70, 2.91, 3.15, 3.83),
    (70, None, 2.72, 2.93, 3.21, 3.84),
    (80, None, 2.75, 2.96, 3.24, 3.84),
    (90, None, 2.77, 2.99, 3.24, 3.82),
    (100, 2.92, 2.80, 3.02, 3.23, 3.80),
    (120, 2.99, 2.85, 3.08, 3.21, 3.78),
    (140, 2.97, 2.91, 3.15, 3.22, 3.80),
    (160, 2.99, 2.98, 3.23, 3.28, 3.81),
    (180, 3.05, 3.08, 3.31, 3.40, 3.82),
    (200, 3.12, 3.19, 3.42, 3.55, 3.87),
    (220, 3.17, 3.32, 3.53, 3.70, 3.92),
    (240, 3.25, 3.45, 3.64, 3.84, 3.97),
    (260, 3.45, 3.58, 3.74, 3.93, 4.03),
    (280, 3.65, 3.69, 3.83, 4.00, 4.10),
    (300, 3.71, 3.78, 3.90, 4.04, 4.20),
    (320, 3.79, 3.84, 3.95, 4.12, 4.31),
    (340, 3.86, 3.88, 4.02, 4.26, 4.44),
    (360, 3.92, 3.95, 4.11, 4.41, 4.56),
    (380, 3.96, 4.03, 4.21, 4.55, 4.66),
    (400, 3.99, 4.11, 4.30, 4.69, 4.75),
    (420, 4.02, 4.19, 4.40, 4.83, 4.84),
    (440, 4.08, 4.27, 4.49, 4.97, 4.94),
    (460, 4.17, 4.34, 4.58, 5.11, 5.03),
    (480, 4.28, 4.42, 4.68, 5.25, 5.13),
    (500, 4.41, 4.50, 4.77, 5.40, 5.22),
    (520, 4.54, 4.58, 4.87, 5.54, 5.32),
    (540, 4.65, 4.66, 4.96, 5.68, 5.41),
    (560, 4.74, 4.74, 5.06, 5.82, 5.51),
    (580, 4.79, 4.82, 5.15, 5.96, 5.60),
    (600, 4.82, 4.89, 5.25, 6.10, 5.70),
)


@dataclass(frozen=True)
class DispersionAnalysis:
    sample_count: int
    left_count: int
    center_count: int
    right_count: int
    mean_lateral_yd: float
    lateral_std_yd: float
    absolute_p50_yd: float
    absolute_p80_yd: float
    ellipse_major_radius_yd: float | None
    ellipse_minor_radius_yd: float | None
    ellipse_angle_deg: float | None
    backing_data: pd.DataFrame
    method_description: str


@dataclass(frozen=True)
class StrokesGainedAnalysis:
    sample_count: int
    mean_strokes_gained_proxy: float
    median_strokes_gained_proxy: float
    clamped_fraction: float
    backing_data: pd.DataFrame
    reference_table: pd.DataFrame
    method_description: str


@dataclass(frozen=True)
class SessionAnalysis:
    summary: pd.DataFrame
    trend_slope_per_session: float
    trend_slope_per_day: float | None
    metric_unit: str
    method_description: str


def _distance_in_yards(values: pd.Series, column: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    unit = infer_unit(column)
    if unit == "m":
        return numeric * METERS_TO_YARDS
    if unit in {"yd", "unitless"}:
        return numeric
    raise ValueError(f"{column} must be a distance in metres or yards")


def analyze_dispersion(
    frame: pd.DataFrame,
    lateral_column: str,
    downrange_column: str | None = None,
    center_tolerance_yd: float = 0.0,
) -> DispersionAnalysis:
    """Summarize signed left/right outcomes and an optional 80% covariance ellipse."""

    if center_tolerance_yd < 0:
        raise ValueError("center tolerance cannot be negative")
    lateral = _distance_in_yards(frame[lateral_column], lateral_column)
    backing = pd.DataFrame({"lateral_yd": lateral}).dropna()
    if backing.empty:
        raise ValueError("dispersion requires at least one finite lateral value")
    downrange: pd.Series | None = None
    if downrange_column:
        downrange = _distance_in_yards(frame[downrange_column], downrange_column)
        backing["downrange_yd"] = downrange.loc[backing.index]
        backing = backing.dropna()
    signed = backing["lateral_yd"].to_numpy(dtype=float)
    left = int(np.count_nonzero(signed < -center_tolerance_yd))
    right = int(np.count_nonzero(signed > center_tolerance_yd))
    center = len(signed) - left - right
    ellipse: tuple[float | None, float | None, float | None] = (None, None, None)
    if downrange is not None and len(backing) >= 3:
        coordinates = backing[["lateral_yd", "downrange_yd"]].to_numpy()
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(coordinates, rowvar=False))
        order = np.argsort(eigenvalues)[::-1]
        scale_80 = float(np.sqrt(-2.0 * np.log(1.0 - 0.80)))
        radii = scale_80 * np.sqrt(np.maximum(eigenvalues[order], 0.0))
        vector = eigenvectors[:, order[0]]
        ellipse = (
            float(radii[0]),
            float(radii[1]),
            float(np.degrees(np.arctan2(vector[1], vector[0]))),
        )
    return DispersionAnalysis(
        sample_count=len(backing),
        left_count=left,
        center_count=center,
        right_count=right,
        mean_lateral_yd=float(np.mean(signed)),
        lateral_std_yd=float(np.std(signed, ddof=1)) if len(signed) > 1 else 0.0,
        absolute_p50_yd=float(np.quantile(np.abs(signed), 0.50)),
        absolute_p80_yd=float(np.quantile(np.abs(signed), 0.80)),
        ellipse_major_radius_yd=ellipse[0],
        ellipse_minor_radius_yd=ellipse[1],
        ellipse_angle_deg=ellipse[2],
        backing_data=backing.reset_index(names="source_index"),
        method_description=(
            "Negative lateral yards are left of target; positive values are right. "
            "The optional ellipse is the bivariate-normal 80% covariance contour."
        ),
    )


def strokes_gained_reference_table() -> pd.DataFrame:
    """Return Broadie Table 9 benchmarks with full source traceability."""

    return pd.DataFrame(
        _BROADIE_ROWS,
        columns=["distance_yd", "tee", "fairway", "rough", "sand", "recovery"],
    ).assign(source_url=BROADIE_SOURCE_URL)


def _interpolate_expected(
    distance: np.ndarray, lie: str, reference: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    if lie not in {"tee", "fairway", "rough", "sand", "recovery"}:
        raise ValueError(f"unsupported lie: {lie}")
    valid = reference[["distance_yd", lie]].dropna()
    lower = float(valid["distance_yd"].min())
    upper = float(valid["distance_yd"].max())
    clamped = (distance < lower) | (distance > upper)
    expected = np.interp(
        np.clip(distance, lower, upper),
        valid["distance_yd"].to_numpy(dtype=float),
        valid[lie].to_numpy(dtype=float),
    )
    return expected, clamped


def calculate_strokes_gained_proxy(
    frame: pd.DataFrame,
    *,
    carry_column: str,
    lateral_column: str,
    target_distance_yd: float,
    start_lie: str,
    end_lie: str,
) -> StrokesGainedAnalysis:
    """Estimate a range-shot SG proxy from carry/lateral endpoint geometry."""

    if target_distance_yd <= 0:
        raise ValueError("target distance must be positive")
    carry = _distance_in_yards(frame[carry_column], carry_column)
    lateral = _distance_in_yards(frame[lateral_column], lateral_column)
    backing = pd.DataFrame({"carry_yd": carry, "lateral_yd": lateral}).dropna()
    if backing.empty:
        raise ValueError("strokes gained requires finite carry and lateral values")
    remaining = np.hypot(
        target_distance_yd - backing["carry_yd"].to_numpy(dtype=float),
        backing["lateral_yd"].to_numpy(dtype=float),
    )
    reference = strokes_gained_reference_table()
    before, before_clamped = _interpolate_expected(
        np.array([target_distance_yd]), start_lie, reference
    )
    after, after_clamped = _interpolate_expected(remaining, end_lie, reference)
    gained = before[0] - 1.0 - after
    backing["target_distance_yd"] = target_distance_yd
    backing["remaining_distance_yd"] = remaining
    backing["expected_strokes_before"] = before[0]
    backing["expected_strokes_after"] = after
    backing["strokes_gained_proxy"] = gained
    backing["benchmark_clamped"] = after_clamped | before_clamped[0]
    return StrokesGainedAnalysis(
        sample_count=len(backing),
        mean_strokes_gained_proxy=float(np.mean(gained)),
        median_strokes_gained_proxy=float(np.median(gained)),
        clamped_fraction=float(backing["benchmark_clamped"].mean()),
        backing_data=backing.reset_index(names="source_index"),
        reference_table=reference,
        method_description=(
            "Broadie-style proxy: expected strokes before - 1 - expected strokes "
            "after. Remaining distance is planar carry/side geometry. Ending lie "
            "and target are user assumptions, so this is not official ShotLink SG."
        ),
    )


def analyze_sessions(
    frame: pd.DataFrame,
    *,
    metric_column: str,
    session_column: str,
    player_column: str | None = None,
    time_column: str | None = None,
) -> SessionAnalysis:
    """Aggregate a metric by player/session and estimate a simple improvement slope."""

    if session_column not in frame or metric_column not in frame:
        raise ValueError("session and metric columns must exist")
    working = frame.copy()
    working["_metric"] = pd.to_numeric(working[metric_column], errors="coerce")
    working = working.dropna(subset=["_metric", session_column])
    if working.empty:
        raise ValueError("session analysis requires finite metric observations")
    group_columns = [session_column]
    if player_column and player_column in working:
        group_columns.insert(0, player_column)
    grouped = working.groupby(group_columns, sort=False, dropna=False)["_metric"]
    summary = grouped.agg(shot_count="size", mean="mean", std="std").reset_index()
    summary["std"] = summary["std"].fillna(0.0)
    has_time = bool(time_column and time_column in working)
    if has_time and time_column is not None:
        working["_session_time"] = pd.to_datetime(
            working[time_column], errors="coerce", utc=True
        )
        times = (
            working.groupby(group_columns, sort=False, dropna=False)["_session_time"]
            .min()
            .reset_index(name="session_start")
        )
        summary = summary.merge(times, on=group_columns, how="left", validate="1:1")
        sort_columns = ([player_column] if player_column else []) + ["session_start"]
        summary = summary.sort_values(sort_columns, kind="stable", na_position="last")
    if player_column and player_column in working:
        summary["session_sequence"] = (
            summary.groupby(player_column, sort=False).cumcount() + 1
        )
    else:
        summary["session_sequence"] = np.arange(1, len(summary) + 1)
    summary["metric_unit"] = infer_unit(metric_column)
    slope_by_player: dict[object, float] = {}
    daily_slope_by_player: dict[object, float] = {}
    trend_groups = (
        summary.groupby(player_column, sort=False, dropna=False)
        if player_column and player_column in summary
        else [("all", summary)]
    )
    for player, player_sessions in trend_groups:
        player_slope = 0.0
        if len(player_sessions) > 1:
            player_slope = float(
                np.polyfit(
                    player_sessions["session_sequence"],
                    player_sessions["mean"],
                    deg=1,
                )[0]
            )
        slope_by_player[player] = player_slope
        if has_time and player_sessions["session_start"].notna().sum() > 1:
            dated = player_sessions.dropna(subset=["session_start"])
            elapsed = (
                dated["session_start"] - dated["session_start"].iloc[0]
            ).dt.total_seconds() / 86_400.0
            if elapsed.nunique() > 1:
                daily_slope_by_player[player] = float(
                    np.polyfit(elapsed, dated["mean"], deg=1)[0]
                )
    if player_column and player_column in summary:
        summary["trend_slope_per_session"] = summary[player_column].map(slope_by_player)
    else:
        summary["trend_slope_per_session"] = slope_by_player["all"]
    slope = float(np.mean(list(slope_by_player.values())))
    daily_slope = (
        float(np.mean(list(daily_slope_by_player.values())))
        if daily_slope_by_player
        else None
    )
    if has_time:
        if player_column and player_column in summary:
            summary["trend_slope_per_day"] = summary[player_column].map(
                daily_slope_by_player
            )
        else:
            summary["trend_slope_per_day"] = daily_slope_by_player.get("all")
    return SessionAnalysis(
        summary=summary,
        trend_slope_per_session=slope,
        trend_slope_per_day=daily_slope,
        metric_unit=infer_unit(metric_column),
        method_description=(
            "Session means and sample standard deviations use every finite shot. "
            "The improvement slope is OLS change in the selected metric per "
            "displayed session, fitted separately for each player. When a time "
            "column is selected, an additional OLS slope per elapsed day is "
            "reported. The scalar "
            "summary is the unweighted mean of player slopes; it is descriptive "
            "and does not adjust for context."
        ),
    )


__all__ = [
    "BROADIE_SOURCE_URL",
    "DispersionAnalysis",
    "SessionAnalysis",
    "StrokesGainedAnalysis",
    "analyze_dispersion",
    "analyze_sessions",
    "calculate_strokes_gained_proxy",
    "strokes_gained_reference_table",
]
