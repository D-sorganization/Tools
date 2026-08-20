"""Exploratory all-pairs scan built on player covariation contracts."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from rate_of_closure._player_covariation_types import PairScanAnalysis, PairScanRequest
from rate_of_closure.player_covariation import (
    CovariationRequest,
    analyze_player_covariation,
)


def _scan_columns(frame: pd.DataFrame, request: PairScanRequest) -> tuple[str, ...]:
    columns = request.numeric_columns or tuple(
        column
        for column in frame.columns
        if column != request.player_column and _has_finite_numeric(frame[column])
    )
    missing = sorted(set(columns).difference(frame.columns))
    if missing or request.player_column not in frame:
        raise ValueError(
            f"missing required columns: {', '.join(missing or [request.player_column])}"
        )
    return tuple(sorted(dict.fromkeys(columns)))


def _has_finite_numeric(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    return bool(np.isfinite(values).any())


def _scan_row(
    frame: pd.DataFrame, scan: PairScanRequest, x_column: str, y_column: str
) -> dict[str, object]:
    request = CovariationRequest(
        x_column, y_column, scan.player_column, scan.min_samples, scan.confidence_level
    )
    analysis = analyze_player_covariation(frame, request)
    valid = analysis.per_player.dropna(subset=["pearson_r"])
    effect = analysis.meta_analysis.random_effect_r
    consistency = None
    if effect is not None and not valid.empty:
        consistency = float(np.mean(np.sign(valid["pearson_r"]) == np.sign(effect)))
    return {
        "x_column": x_column,
        "y_column": y_column,
        "x_unit": analysis.units["x"],
        "y_unit": analysis.units["y"],
        "random_effect_r": effect,
        "fixed_effect_r": analysis.meta_analysis.fixed_effect_r,
        "within_player_r": analysis.within_player.pearson_r,
        "between_player_r": analysis.between_player.pearson_r,
        "contributor_count": analysis.meta_analysis.contributor_count,
        "total_sample_count": analysis.meta_analysis.total_sample_count,
        "i_squared_pct": analysis.meta_analysis.i_squared_pct,
        "direction_consistency": consistency,
    }


def _scan_covariation_pairs(
    frame: pd.DataFrame, request: PairScanRequest
) -> PairScanAnalysis:
    """Rank every selected numeric pair by random-effects correlation magnitude."""

    columns = _scan_columns(frame, request)
    if len(columns) < 2:
        raise ValueError("pair scan requires at least two candidate columns")
    pairs = combinations(columns, 2)
    ranking = pd.DataFrame(
        [_scan_row(frame, request, left, right) for left, right in pairs]
    )
    ranking["absolute_random_effect_r"] = ranking["random_effect_r"].abs()
    ranking = ranking.sort_values(
        ["absolute_random_effect_r", "contributor_count", "x_column", "y_column"],
        ascending=[False, False, True, True],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    pair_count = len(ranking)
    warnings = (
        f"Exploratory scan evaluated {pair_count} pairs; rankings are not "
        "confirmatory.",
        "Multiplicity increases false-positive risk; validate selected "
        "relationships on held-out data.",
        "Correlation does not imply causation and may reflect measurement or "
        "context effects.",
    )
    return PairScanAnalysis(
        ranking,
        warnings,
        "Pairs rank by absolute random-effects Pearson correlation, then "
        "contributor count and lexical names.",
    )


__all__ = ["_scan_covariation_pairs"]
