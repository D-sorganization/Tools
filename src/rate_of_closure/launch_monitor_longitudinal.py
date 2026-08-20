"""Identity-safe player/session longitudinal performance analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class LongitudinalRequest:
    """Select trusted identity, order, metric direction, and inference rules."""

    metric_column: str
    session_column: str
    session_order_column: str
    player_column: str
    player_identity_attested: bool
    session_identity_attested: bool
    higher_is_better: bool
    confidence_level: float = 0.95
    min_sessions: int = 3


@dataclass(frozen=True)
class LongitudinalSessionPoint:
    """Equal-session analysis point with within-session uncertainty."""

    player_id: str
    session_id: str
    session_order: float
    sample_count: int
    mean: float
    standard_deviation: float | None
    standard_error: float | None
    cumulative_mean: float


@dataclass(frozen=True)
class PlayerLongitudinalEstimate:
    """One player's OLS session-mean trend."""

    player_id: str
    session_count: int
    slope_per_session: float | None
    standard_error: float | None
    ci_lower: float | None
    ci_upper: float | None
    p_value: float | None
    r_squared: float | None
    first_to_last_change: float | None
    status: str


@dataclass(frozen=True)
class PopulationLongitudinalSummary:
    """Inverse-variance fixed/random synthesis of eligible player slopes."""

    contributor_count: int
    fixed_effect_slope: float | None
    fixed_ci_lower: float | None
    fixed_ci_upper: float | None
    random_effect_slope: float | None
    random_ci_lower: float | None
    random_ci_upper: float | None
    tau_squared: float | None
    q_statistic: float | None
    i_squared_pct: float | None
    improvement_probability: float | None


@dataclass(frozen=True)
class LongitudinalResult:
    """Export-ready sessions, player trends, population effect, and caveats."""

    request: LongitudinalRequest
    session_points: tuple[LongitudinalSessionPoint, ...]
    players: tuple[PlayerLongitudinalEstimate, ...]
    population: PopulationLongitudinalSummary
    formula: str
    warnings: tuple[str, ...]


def _validate(frame: pd.DataFrame, request: LongitudinalRequest) -> None:
    if not request.player_identity_attested or not request.session_identity_attested:
        raise ValueError("player and session identity must both be explicitly attested")
    if request.min_sessions < 3:
        raise ValueError("min_sessions must be at least three")
    if not 0.5 < request.confidence_level < 1:
        raise ValueError("confidence_level must be between 0.5 and 1")
    required = {
        request.metric_column,
        request.session_column,
        request.session_order_column,
        request.player_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"columns are unavailable: {missing}")


def _session_frame(frame: pd.DataFrame, request: LongitudinalRequest) -> pd.DataFrame:
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
    selected["player"] = selected["player"].astype(str).str.strip()
    selected["session"] = selected["session"].astype(str).str.strip()
    selected["order"] = pd.to_numeric(selected["order"], errors="coerce")
    selected["metric"] = pd.to_numeric(selected["metric"], errors="coerce")
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna()
    selected = selected[(selected["player"] != "") & (selected["session"] != "")]
    if selected.empty:
        raise ValueError("longitudinal analysis requires complete trusted session rows")
    order_counts = selected.groupby(["player", "session"])["order"].nunique()
    if bool((order_counts != 1).any()):
        raise ValueError("each player session must map to exactly one order")
    summaries = selected.groupby(["player", "session"], as_index=False).agg(
        order=("order", "first"),
        mean=("metric", "mean"),
        count=("metric", "size"),
        standard_deviation=("metric", "std"),
    )
    duplicates = summaries.groupby(["player", "order"])["session"].nunique()
    if bool((duplicates > 1).any()):
        raise ValueError("each player session requires a unique order value")
    summaries = summaries.sort_values(["player", "order", "session"], kind="stable")
    summaries["standard_error"] = summaries["standard_deviation"] / np.sqrt(
        summaries["count"]
    )
    summaries["cumulative"] = (
        summaries.groupby("player")["mean"].expanding().mean().droplevel(0)
    )
    return summaries


def _session_points(frame: pd.DataFrame) -> tuple[LongitudinalSessionPoint, ...]:
    return tuple(
        LongitudinalSessionPoint(
            str(row["player"]),
            str(row["session"]),
            float(row["order"]),
            int(row["count"]),
            float(row["mean"]),
            None
            if pd.isna(row["standard_deviation"])
            else float(row["standard_deviation"]),
            None if pd.isna(row["standard_error"]) else float(row["standard_error"]),
            float(row["cumulative"]),
        )
        for _, row in frame.iterrows()
    )


def _player_estimate(
    player: str, frame: pd.DataFrame, request: LongitudinalRequest
) -> PlayerLongitudinalEstimate:
    count = len(frame)
    if count < request.min_sessions:
        return PlayerLongitudinalEstimate(
            player,
            count,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "insufficient_sessions",
        )
    x = frame["order"].to_numpy(float)
    y = frame["mean"].to_numpy(float)
    if np.ptp(x) <= np.finfo(float).eps or np.ptp(y) <= np.finfo(float).eps:
        status = (
            "constant_order" if np.ptp(x) <= np.finfo(float).eps else "constant_metric"
        )
        return PlayerLongitudinalEstimate(
            player,
            count,
            None,
            None,
            None,
            None,
            None,
            None,
            float(y[-1] - y[0]),
            status,
        )
    regression = stats.linregress(x, y)
    critical = float(stats.t.ppf(0.5 + request.confidence_level / 2, count - 2))
    margin = critical * regression.stderr
    return PlayerLongitudinalEstimate(
        player,
        count,
        float(regression.slope),
        float(regression.stderr),
        float(regression.slope - margin),
        float(regression.slope + margin),
        float(regression.pvalue),
        float(regression.rvalue**2),
        float(y[-1] - y[0]),
        "ok",
    )


def _population(
    players: tuple[PlayerLongitudinalEstimate, ...], request: LongitudinalRequest
) -> PopulationLongitudinalSummary:
    slopes_list: list[float] = []
    error_list: list[float] = []
    for player in players:
        slope = player.slope_per_session
        error = player.standard_error
        if (
            player.status == "ok"
            and slope is not None
            and error is not None
            and error > 0
        ):
            slopes_list.append(slope)
            error_list.append(error)
    if len(slopes_list) < 2:
        return _empty_population(len(slopes_list))
    slopes = np.asarray(slopes_list, dtype=np.float64)
    variances = np.square(np.asarray(error_list, dtype=np.float64))
    weights = 1.0 / variances
    fixed = float(np.average(slopes, weights=weights))
    q_statistic = float(np.sum(weights * np.square(slopes - fixed)))
    degrees = len(slopes) - 1
    c_value = float(weights.sum() - np.square(weights).sum() / weights.sum())
    tau_squared = max(0.0, (q_statistic - degrees) / c_value) if c_value else 0.0
    random_weights = 1.0 / (variances + tau_squared)
    random = float(np.average(slopes, weights=random_weights))
    critical = float(stats.norm.ppf(0.5 + request.confidence_level / 2))
    fixed_margin = critical / np.sqrt(weights.sum())
    random_se = float(1 / np.sqrt(random_weights.sum()))
    direction = 1.0 if request.higher_is_better else -1.0
    probability = float(stats.norm.cdf(direction * random / random_se))
    i_squared = (
        max(0.0, (q_statistic - degrees) / q_statistic) * 100 if q_statistic else 0.0
    )
    return PopulationLongitudinalSummary(
        len(slopes_list),
        fixed,
        fixed - fixed_margin,
        fixed + fixed_margin,
        random,
        random - critical * random_se,
        random + critical * random_se,
        tau_squared,
        q_statistic,
        i_squared,
        probability,
    )


def _empty_population(contributor_count: int) -> PopulationLongitudinalSummary:
    return PopulationLongitudinalSummary(
        contributor_count, None, None, None, None, None, None, None, None, None, None
    )


def analyze_longitudinal_performance(
    frame: pd.DataFrame, request: LongitudinalRequest
) -> LongitudinalResult:
    """Estimate session summaries, player slopes, and population trend."""

    _validate(frame, request)
    sessions = _session_frame(frame, request)
    players = tuple(
        _player_estimate(str(player), group, request)
        for player, group in sessions.groupby("player", sort=True)
    )
    return LongitudinalResult(
        request,
        _session_points(sessions),
        players,
        _population(players, request),
        "Each session contributes its metric mean. Player OLS slopes use explicit "
        "session order; eligible slopes use inverse-variance fixed and "
        "DerSimonian-Laird random-effects synthesis.",
        (
            "Observed longitudinal association does not establish causality or "
            "isolate practice effects.",
            "Changes can reflect equipment, intent, monitor, environment, "
            "selection, fatigue, or regression to the mean.",
            "Improvement probability is a normal-approximation summary "
            "conditional on the declared higher/lower-is-better direction.",
        ),
    )


__all__ = [
    "LongitudinalRequest",
    "LongitudinalResult",
    "analyze_longitudinal_performance",
]
