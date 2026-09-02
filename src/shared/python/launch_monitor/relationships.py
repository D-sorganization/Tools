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

Owner ruling **D15** does not apply here
-----------------------------------------
ADR-0048's "Owner Rulings (2026-09-02)" section reads as though D15 (FDR
excludes under-sampled predictors before correcting) reaches both this module
and :mod:`~shared.python.launch_monitor.flexible_analysis`. Empirically it
does not reach this one: this module has no separate, configurable
``min_samples`` tier above its own hardcoded three-complete-pairs floor, so
there is no second, later blanking step for the ruling's defect (a predictor
that clears the floor, contributes a finite p to the correction, and is only
*afterwards* blanked) to exist in. The only under-sampling floor here is the
hardcoded three complete pairs (plus a two-distinct-value requirement on each
side); a pair below it yields ``nan`` for both coefficient and p-value
directly out of :func:`_pair_correlation`, and non-finite p-values are already
excluded from the Benjamini-Hochberg denominator by
:func:`_benjamini_hochberg`'s own finite-value filter. This module's behaviour
already matches the ruling by construction; there is no follow-up to land
here. D15's actual fix is in
:func:`~shared.python.launch_monitor.flexible_analysis._correlations`, which
*does* have a separate ``min_samples`` above the floor this module reads
values through.

Owner ruling **D17** applied here
----------------------------------
UD's ``pd.to_numeric``/``float`` cast projects a boolean column to 0/1 and
analyses it as numeric; Tools' ``finite_launch_monitor_scalar`` refused
booleans outright. Per the accepted ruling (UpstreamDrift PR #9392,
``docs/adr/0048-launch-monitor-port-plan.md`` "Owner Rulings (2026-09-02)",
D17), the analysis capability is preserved — a boolean column is still
projected to 0/1 and analysed — but the projection is no longer silent: every
selected metric backed by a boolean column is named in
:attr:`CorrelationResult.boolean_projected`, and every :class:`DependencyEdge`
touching one carries :attr:`DependencyEdge.includes_boolean_projection`. A
boolean-projected column can therefore never be misread as native numeric.
This labels the projection; it does not change it — the correlation
coefficients and p-values for a boolean column are bit-identical to what this
module already computed before this change. Only ``selected`` metrics are
tracked (matrix rows/columns); a boolean ``controls`` column still
participates in partial-correlation residualisation unlabelled, since
``controls`` are inputs to the matrix, not entries in it.
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
    """One statistically screened dependency-network edge.

    ``includes_boolean_projection`` is ``True`` when ``source`` or ``target``
    is one of the result's ``boolean_projected`` metrics (owner ruling D17;
    see :class:`CorrelationResult`).
    """

    source: str
    target: str
    coefficient: float
    p_value: float
    adjusted_p_value: float
    sample_count: int
    includes_derived_metric: bool
    includes_boolean_projection: bool


@dataclass(frozen=True)
class CorrelationResult:
    """Complete pairwise and optional partial-correlation result.

    ``boolean_projected`` names the selected metrics whose column was boolean
    and is analysed as 0/1 via an explicit projection (owner ruling D17). A
    name in this tuple can never be read as native numeric —
    ``coefficients``/``p_values``/etc. still hold the projected result, but
    the projection is labelled rather than silent.
    """

    method: str
    coefficients: pd.DataFrame
    p_values: pd.DataFrame
    adjusted_p_values: pd.DataFrame | None
    pair_counts: pd.DataFrame
    partial_coefficients: pd.DataFrame | None
    derived_metrics: tuple[str, ...]
    boolean_projected: tuple[str, ...]
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
    boolean_projected = tuple(
        metric for metric in selected if pd.api.types.is_bool_dtype(frame[metric])
    )
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
                    left in boolean_projected or right in boolean_projected,
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
        boolean_projected,
        tuple(edges),
    )
