"""Correlation, partial-correlation, and dependency-network analysis.

Ported from UpstreamDrift ``src/shared/python/launch_monitor/relationships.py``
(187 lines) under ADR-0046 Stage 1 — step **P7** of the ADR-0046 G1 port plan
(UpstreamDrift ``docs/adr/0048-launch-monitor-port-plan.md``). The
implementation is UpstreamDrift's, carried over unchanged rather than
reimplemented; its authors retain authorship. No behaviour is added, removed,
or limited by the move.

**Not the same shape as its ``rate_of_closure`` neighbour.** The port plan
records the counterpart as ``_launch_monitor_analysis_statistics.correlations``
— "plain correlation only". That function is a *star*: one outcome against a
list of predictors, one estimate per predictor. This module is a *matrix*:
every pair among the selected metrics, with a Benjamini-Hochberg FDR
correction over the upper triangle, optional partial correlations obtained by
residualising every metric on a shared control set, a per-pair complete-case
count, and a screened dependency network whose edges survive both an absolute
coefficient threshold and the adjusted-p threshold. Neither is a subset of the
other and the two are ungated, so nothing here re-exports or aliases anything
there.

Derived metrics are flagged rather than dropped: an edge that touches a metric
with a ``derived_from`` record in :mod:`shared.python.launch_monitor.schema`
carries ``includes_derived_metric=True``, because a correlation between
``smash_factor`` and ``ball_speed`` is partly an algebraic identity, not
evidence.

Pending owner rulings — deliberately **not** applied here
--------------------------------------------------------
Two rulings, **D15** (FDR excludes under-sampled predictors before correcting)
and **D17** (booleans analysed as 0/1 with explicit projection labelling), were
accepted after this port was scoped and apply to this canonical module. They
are **not** in this port, which carries UpstreamDrift's behaviour verbatim so
the port diff and the behaviour diff can be reviewed separately; a follow-up PR
applies them here. Today's behaviour, pinned by this module's tests so that the
follow-up's diff is visible:

* the only under-sampling floor is the hardcoded three complete pairs (plus a
  two-distinct-value requirement on each side); a pair below it yields ``nan``
  for both coefficient and p-value, and non-finite p-values are already outside
  the Benjamini-Hochberg denominator;
* a boolean column is projected to 0/1 by the ``float`` cast inside
  ``_pair_correlation`` and analysed as numeric, and the result records nothing
  to say a projection happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from shared.python.launch_monitor.schema import METRICS

__all__ = ["CorrelationResult", "DependencyEdge", "compute_correlations"]


@dataclass(frozen=True)
class DependencyEdge:
    """One statistically screened dependency-network edge."""

    source: str
    target: str
    coefficient: float
    p_value: float
    adjusted_p_value: float
    sample_count: int
    includes_derived_metric: bool


@dataclass(frozen=True)
class CorrelationResult:
    """Complete pairwise and optional partial-correlation result."""

    method: str
    coefficients: pd.DataFrame
    p_values: pd.DataFrame
    adjusted_p_values: pd.DataFrame | None
    pair_counts: pd.DataFrame
    partial_coefficients: pd.DataFrame | None
    derived_metrics: tuple[str, ...]
    edges: tuple[DependencyEdge, ...]


def _pair_correlation(
    x: pd.Series, y: pd.Series, method: str
) -> tuple[float, float, int]:
    valid = pd.concat([x, y], axis=1).dropna()
    count = len(valid)
    if count < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return float("nan"), float("nan"), count
    left = valid.iloc[:, 0].to_numpy(float)
    right = valid.iloc[:, 1].to_numpy(float)
    if method == "pearson":
        result = stats.pearsonr(left, right)
    elif method == "spearman":
        result = stats.spearmanr(left, right)
    elif method == "kendall":
        result = stats.kendalltau(left, right)
    else:
        raise ValueError("method must be one of: pearson, spearman, kendall")
    return float(result.statistic), float(result.pvalue), count


def _benjamini_hochberg(matrix: pd.DataFrame) -> pd.DataFrame:
    adjusted = pd.DataFrame(np.nan, index=matrix.index, columns=matrix.columns)
    matrix_values = matrix.to_numpy(dtype=float)
    pairs: list[tuple[int, int, float]] = []
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            value = matrix_values[i, j]
            if np.isfinite(value):
                pairs.append((i, j, float(value)))
    if not pairs:
        return adjusted
    order = np.argsort([item[2] for item in pairs])
    raw = np.asarray([pairs[index][2] for index in order], dtype=float)
    ranks = np.arange(1, len(raw) + 1)
    corrected = np.minimum.accumulate((raw * len(raw) / ranks)[::-1])[::-1]
    corrected = np.clip(corrected, 0.0, 1.0)
    for ordered_index, value in zip(order, corrected, strict=True):
        i, j, _ = pairs[int(ordered_index)]
        adjusted.iat[i, j] = value
        adjusted.iat[j, i] = value
    for index in range(len(adjusted)):
        adjusted.iat[index, index] = 0.0
    return adjusted


def _residualize(values: pd.Series, controls: pd.DataFrame) -> pd.Series:
    combined = pd.concat([values, controls], axis=1).dropna()
    output = pd.Series(np.nan, index=values.index, dtype=float)
    if len(combined) <= controls.shape[1] + 2:
        return output
    y = combined.iloc[:, 0].to_numpy(float)
    x = combined.iloc[:, 1:].to_numpy(float)
    design = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    output.loc[combined.index] = y - design @ coefficients
    return output


def compute_correlations(
    frame: pd.DataFrame,
    *,
    metrics: tuple[str, ...] | list[str],
    method: str = "pearson",
    controls: tuple[str, ...] | list[str] = (),
    edge_threshold: float = 0.3,
    alpha: float = 0.05,
) -> CorrelationResult:
    """Compute pairwise relationships with FDR correction and controls."""
    selected = tuple(metrics)
    if len(selected) < 2:
        raise ValueError("At least two metrics are required")
    missing = (set(selected) | set(controls)) - set(frame.columns)
    if missing:
        raise ValueError(f"Columns not present: {sorted(missing)}")
    numeric = frame[list(selected)].apply(pd.to_numeric, errors="coerce")
    coefficients = pd.DataFrame(np.nan, index=selected, columns=selected)
    p_values = coefficients.copy()
    pair_counts = pd.DataFrame(0, index=selected, columns=selected, dtype=int)
    for i, left in enumerate(selected):
        for j in range(i, len(selected)):
            right = selected[j]
            if i == j:
                count = int(numeric[left].notna().sum())
                coefficient, p_value = 1.0, 0.0
            else:
                coefficient, p_value, count = _pair_correlation(
                    numeric[left], numeric[right], method
                )
            coefficients.iat[i, j] = coefficients.iat[j, i] = coefficient
            p_values.iat[i, j] = p_values.iat[j, i] = p_value
            pair_counts.iat[i, j] = pair_counts.iat[j, i] = count
    adjusted = _benjamini_hochberg(p_values)

    partial: pd.DataFrame | None = None
    if controls:
        control_frame = frame[list(controls)].apply(pd.to_numeric, errors="coerce")
        residuals = {
            metric: _residualize(numeric[metric], control_frame) for metric in selected
        }
        partial = pd.DataFrame(np.nan, index=selected, columns=selected)
        for i, left in enumerate(selected):
            for j in range(i, len(selected)):
                right = selected[j]
                value = (
                    1.0
                    if i == j
                    else _pair_correlation(residuals[left], residuals[right], method)[0]
                )
                partial.iat[i, j] = partial.iat[j, i] = value

    derived = tuple(
        metric
        for metric in selected
        if metric in METRICS and METRICS[metric].derived_from
    )
    edges: list[DependencyEdge] = []
    for i, left in enumerate(selected):
        for j in range(i + 1, len(selected)):
            right = selected[j]
            coefficient = float(coefficients.to_numpy(dtype=float)[i, j])
            adjusted_p = float(adjusted.to_numpy(dtype=float)[i, j])
            if not np.isfinite(coefficient) or not np.isfinite(adjusted_p):
                continue
            if abs(coefficient) < edge_threshold or adjusted_p > alpha:
                continue
            edges.append(
                DependencyEdge(
                    left,
                    right,
                    coefficient,
                    float(p_values.to_numpy(dtype=float)[i, j]),
                    adjusted_p,
                    int(pair_counts.to_numpy(dtype=int)[i, j]),
                    left in derived or right in derived,
                )
            )
    edges.sort(key=lambda edge: (-abs(edge.coefficient), edge.source, edge.target))
    return CorrelationResult(
        method,
        coefficients,
        p_values,
        adjusted,
        pair_counts,
        partial,
        derived,
        tuple(edges),
    )
