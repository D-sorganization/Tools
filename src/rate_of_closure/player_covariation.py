"""Within-player association and cross-player meta-analysis.

The calculations are descriptive. They separate variation among a player's
shots from differences between player averages, which avoids interpreting a
pooled correlation as though it necessarily described individual players.
"""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

from rate_of_closure._player_covariation_types import (
    MIN_FISHER_SAMPLES as MIN_FISHER_SAMPLES,
)
from rate_of_closure._player_covariation_types import (
    AssociationEstimate,
    CovariationRequest,
    MetaAnalysisSummary,
    PairScanAnalysis,
    PairScanRequest,
    PlayerCovariationAnalysis,
)
from rate_of_closure.launch_monitor_data import infer_unit

EPSILON = np.finfo(float).eps


class _EstimateRules(NamedTuple):
    group_count: int
    minimum: int
    confidence: float


def _validate_request(frame: pd.DataFrame, request: CovariationRequest) -> None:
    if request.x_column == request.y_column:
        raise ValueError("x and y columns must differ")
    required = {request.x_column, request.y_column, request.player_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    if request.min_samples < MIN_FISHER_SAMPLES:
        raise ValueError("min_samples must be at least 4 for Fisher inference")
    if not 0 < request.confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")


def _pairwise_data(frame: pd.DataFrame, request: CovariationRequest) -> pd.DataFrame:
    selected = frame[[request.player_column, request.x_column, request.y_column]].copy()
    selected = selected.set_axis(["player_id", "x", "y"], axis="columns")
    selected["x"] = pd.to_numeric(selected["x"], errors="coerce")
    selected["y"] = pd.to_numeric(selected["y"], errors="coerce")
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna()
    selected["player_id"] = selected["player_id"].astype(str).str.strip()
    selected = selected[selected["player_id"] != ""]
    return selected.sort_values("player_id", kind="stable")


def _status(values_x: np.ndarray, values_y: np.ndarray, minimum: int) -> str:
    if len(values_x) < minimum:
        return "insufficient_samples"
    constant_x = bool(np.ptp(values_x) <= EPSILON)
    constant_y = bool(np.ptp(values_y) <= EPSILON)
    if constant_x and constant_y:
        return "constant_both"
    if constant_x:
        return "constant_x"
    if constant_y:
        return "constant_y"
    return "ok"


def _fisher_interval(
    coefficient: float, count: int, confidence: float
) -> tuple[float, float]:
    transformed = np.arctanh(np.clip(coefficient, -0.999999, 0.999999))
    margin = stats.norm.ppf(0.5 + confidence / 2) / np.sqrt(count - 3)
    return float(np.tanh(transformed - margin)), float(np.tanh(transformed + margin))


def _estimate_arrays(
    values_x: np.ndarray,
    values_y: np.ndarray,
    rules: _EstimateRules,
) -> AssociationEstimate:
    status = _status(values_x, values_y, rules.minimum)
    if status != "ok":
        return AssociationEstimate(
            len(values_x),
            rules.group_count,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            status,
        )
    pearson = float(stats.pearsonr(values_x, values_y).statistic)
    spearman = float(stats.spearmanr(values_x, values_y).statistic)
    slope, intercept = np.polyfit(values_x, values_y, deg=1)
    interval = (
        _fisher_interval(pearson, len(values_x), rules.confidence)
        if len(values_x) >= MIN_FISHER_SAMPLES
        else (None, None)
    )
    return AssociationEstimate(
        len(values_x),
        rules.group_count,
        pearson,
        spearman,
        float(slope),
        float(intercept),
        pearson**2,
        interval[0],
        interval[1],
        status,
    )


def _estimate_frame(
    frame: pd.DataFrame,
    request: CovariationRequest,
    *,
    minimum: int | None = None,
) -> AssociationEstimate:
    return _estimate_arrays(
        frame["x"].to_numpy(float),
        frame["y"].to_numpy(float),
        _EstimateRules(
            int(frame["player_id"].nunique()),
            request.min_samples if minimum is None else minimum,
            request.confidence_level,
        ),
    )


def _player_rows(backing: pd.DataFrame, request: CovariationRequest) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for player, group in backing.groupby("player_id", sort=True):
        estimate = _estimate_frame(group, request)
        rows.append(
            {
                "player_id": player,
                "sample_count": estimate.sample_count,
                "pearson_r": estimate.pearson_r,
                "spearman_r": estimate.spearman_r,
                "slope": estimate.slope,
                "intercept": estimate.intercept,
                "r_squared": estimate.r_squared,
                "ci_lower": estimate.ci_lower,
                "ci_upper": estimate.ci_upper,
                "status": estimate.status,
            }
        )
    return pd.DataFrame(rows)


def _empty_meta(
    contributor_count: int = 0, total_sample_count: int = 0
) -> MetaAnalysisSummary:
    return MetaAnalysisSummary(
        contributor_count,
        total_sample_count,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _weighted_effect(
    fisher_z: np.ndarray, weights: np.ndarray, confidence: float
) -> tuple[float, float, float]:
    mean = float(np.average(fisher_z, weights=weights))
    margin = float(stats.norm.ppf(0.5 + confidence / 2) / np.sqrt(weights.sum()))
    return (
        float(np.tanh(mean)),
        float(np.tanh(mean - margin)),
        float(np.tanh(mean + margin)),
    )


def _meta_analysis(
    per_player: pd.DataFrame, confidence: float
) -> tuple[MetaAnalysisSummary, pd.DataFrame]:
    valid = per_player[per_player["status"] == "ok"].copy()
    if len(valid) < 2:
        total = int(valid["sample_count"].sum())
        return _empty_meta(len(valid), total), per_player.assign(
            fixed_weight=np.nan, random_weight=np.nan
        )
    correlations = valid["pearson_r"].to_numpy(float)
    fisher_z = np.arctanh(np.clip(correlations, -0.999999, 0.999999))
    variances = 1.0 / (valid["sample_count"].to_numpy(float) - 3.0)
    fixed_weights = 1.0 / variances
    fixed_mean = float(np.average(fisher_z, weights=fixed_weights))
    q_statistic = float(np.sum(fixed_weights * (fisher_z - fixed_mean) ** 2))
    degrees = len(valid) - 1
    denominator = (
        fixed_weights.sum() - np.square(fixed_weights).sum() / fixed_weights.sum()
    )
    tau_squared = (
        max(0.0, (q_statistic - degrees) / denominator) if denominator else 0.0
    )
    random_weights = 1.0 / (variances + tau_squared)
    fixed = _weighted_effect(fisher_z, fixed_weights, confidence)
    random = _weighted_effect(fisher_z, random_weights, confidence)
    i_squared = (
        max(0.0, (q_statistic - degrees) / q_statistic) * 100 if q_statistic else 0.0
    )
    weighted = _attach_weights(per_player, valid.index, fixed_weights, random_weights)
    summary = MetaAnalysisSummary(
        len(valid),
        int(valid["sample_count"].sum()),
        *fixed,
        *random,
        float(tau_squared),
        q_statistic,
        float(i_squared),
    )
    return summary, weighted


def _attach_weights(
    table: pd.DataFrame,
    valid_index: pd.Index,
    fixed_weights: np.ndarray,
    random_weights: np.ndarray,
) -> pd.DataFrame:
    output = table.assign(fixed_weight=np.nan, random_weight=np.nan)
    output.loc[valid_index, "fixed_weight"] = fixed_weights / fixed_weights.sum()
    output.loc[valid_index, "random_weight"] = random_weights / random_weights.sum()
    return output


def _centered(backing: pd.DataFrame) -> pd.DataFrame:
    output = backing.copy()
    grouped = output.groupby("player_id", sort=False)
    output["centered_x"] = output["x"] - grouped["x"].transform("mean")
    output["centered_y"] = output["y"] - grouped["y"].transform("mean")
    return output


def _between(backing: pd.DataFrame, request: CovariationRequest) -> AssociationEstimate:
    means = backing.groupby("player_id", sort=True)[["x", "y"]].mean().reset_index()
    return _estimate_frame(means, request, minimum=2)


def _warnings(
    original: pd.DataFrame, backing: pd.DataFrame, per_player: pd.DataFrame
) -> tuple[str, ...]:
    warnings = [
        "Associations are observational and do not establish a causal relationship."
    ]
    removed = len(original) - len(backing)
    if removed:
        warnings.append(
            f"{removed} rows were excluded for missing or non-finite values."
        )
    excluded = int((per_player["status"] != "ok").sum()) if not per_player.empty else 0
    if excluded:
        warnings.append(f"{excluded} players were excluded from Fisher-z synthesis.")
    eligible = int((per_player["status"] == "ok").sum()) if not per_player.empty else 0
    if eligible < 2:
        warnings.append(
            "Meta-analysis requires at least two eligible players; pooled effects "
            "are unavailable."
        )
    return tuple(warnings)


def _reversal_warnings(
    pooled: AssociationEstimate, within: AssociationEstimate
) -> tuple[str, ...]:
    if pooled.pearson_r is None or within.pearson_r is None:
        return ()
    if np.sign(pooled.pearson_r) == np.sign(within.pearson_r):
        return ()
    return (
        "Possible aggregation reversal: pooled and within-player Pearson "
        "correlations have opposite signs; inspect group structure.",
    )


def analyze_player_covariation(
    frame: pd.DataFrame, request: CovariationRequest
) -> PlayerCovariationAnalysis:
    """Estimate pooled, within-player, between-player, and meta associations."""

    _validate_request(frame, request)
    backing = _centered(_pairwise_data(frame, request))
    if backing.empty:
        raise ValueError("analysis requires at least one pairwise-complete player shot")
    per_player = _player_rows(backing, request)
    meta, per_player = _meta_analysis(per_player, request.confidence_level)
    centered = backing.assign(x=backing["centered_x"], y=backing["centered_y"])
    pooled = _estimate_frame(backing, request)
    within = replace(_estimate_frame(centered, request), ci_lower=None, ci_upper=None)
    definitions = {
        "pooled": "Association across all pairwise-complete shots without centering.",
        "within_player": "Association after subtracting each player's x and y means.",
        "between_player": "Unweighted association between player-level x and y means.",
        "meta_analysis": (
            "Fisher-z fixed effect and DerSimonian-Laird random effects of "
            "player Pearson r."
        ),
    }
    return PlayerCovariationAnalysis(
        request,
        per_player,
        pooled,
        within,
        _between(backing, request),
        meta,
        backing.reset_index(names="source_index"),
        {"x": infer_unit(request.x_column), "y": infer_unit(request.y_column)},
        definitions,
        _warnings(frame, backing, per_player) + _reversal_warnings(pooled, within),
        "Pairwise-complete Pearson, Spearman, and OLS estimates; Fisher confidence "
        "intervals and meta-analysis apply only to Pearson r. Spearman is "
        "descriptive. Association does not imply causation.",
    )


def scan_covariation_pairs(
    frame: pd.DataFrame, request: PairScanRequest
) -> PairScanAnalysis:
    """Rank selected numeric pairs using the shared player-level estimator."""

    from rate_of_closure._player_covariation_scan import _scan_covariation_pairs

    return _scan_covariation_pairs(frame, request)


__all__ = [
    "AssociationEstimate",
    "CovariationRequest",
    "MetaAnalysisSummary",
    "PairScanAnalysis",
    "PairScanRequest",
    "PlayerCovariationAnalysis",
    "analyze_player_covariation",
    "scan_covariation_pairs",
]
