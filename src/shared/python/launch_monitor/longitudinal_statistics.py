"""Session-unit descriptive and player-clustered longitudinal statistics.

Ported from UpstreamDrift
``src/shared/python/launch_monitor/longitudinal_statistics.py`` (147 lines)
under ADR-0046 Stage 1 — step **P15** of the ADR-0046 G1 port plan
(UpstreamDrift ``docs/adr/0048-launch-monitor-port-plan.md``). The
implementation is UpstreamDrift's, carried over rather than reimplemented; its
authors retain authorship.

Decision G1-D1 lands here as a second, named estimator
------------------------------------------------------
:func:`clustered_pooled_association` is UpstreamDrift's, arithmetic unchanged,
now naming itself ``ud-cluster-robust-fe/1``.
:func:`dersimonian_laird_pooled_association` is its named pair,
``dl-random-effects/1``, ported from
``rate_of_closure.launch_monitor_longitudinal._population`` — the estimator G0
pinned as D10/D12. Neither is removed and neither is the other's fallback: the
caller names which one it wants, and the result carries that name.

:func:`player_associations` additionally closes D11. G0 recorded that
``rate_of_closure`` "reports SE/CI/p/R2/first-to-last change per player" while
UpstreamDrift's ``LongitudinalPlayerAssociationV1`` "carries the point estimate
and a direction label only" and "cannot express per-player slope uncertainty at
all". The slope arithmetic is untouched; the uncertainty that was already
sitting unused in the ``linregress`` result is now reported, which is also what
``dl-random-effects/1`` weights by.
"""

from __future__ import annotations

from math import sqrt
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from shared.python.launch_monitor.longitudinal_types import (
    LongitudinalPlayerAssociationV1,
    LongitudinalSessionRequestV1,
    PooledAssociationV1,
)


def _direction(estimate: float) -> Literal["increasing", "decreasing", "flat"]:
    if estimate > 1e-12:
        return "increasing"
    if estimate < -1e-12:
        return "decreasing"
    return "flat"


def _first_to_last_change(group: pd.DataFrame) -> float | None:
    if len(group) < 2:
        return None
    values = group["metric_value"].to_numpy(dtype=float)
    return float(values[-1] - values[0])


def player_associations(
    cells: pd.DataFrame, request: LongitudinalSessionRequestV1
) -> tuple[LongitudinalPlayerAssociationV1, ...]:
    """Fit descriptive slopes after collapsing every player-session equally."""
    session_means = (
        cells.groupby(["player_id", "session_id", "order_value"], as_index=False)[
            "metric_value"
        ]
        .mean()
        .sort_values(["player_id", "order_value", "session_id"])
    )
    results: list[LongitudinalPlayerAssociationV1] = []
    for player_id, group in session_means.groupby("player_id", sort=True):
        session_count = len(group)
        if (
            session_count < request.minimum_sessions_per_player
            or group["order_value"].nunique() < 2
        ):
            results.append(
                LongitudinalPlayerAssociationV1(
                    player_id=str(player_id),
                    session_count=session_count,
                    direction="unavailable",
                    state="unavailable",
                    reason_code="insufficient_ordered_sessions",
                    first_to_last_change=_first_to_last_change(group),
                )
            )
            continue
        regression = stats.linregress(group["order_value"], group["metric_value"])
        estimate = float(regression.slope)
        standard_error = float(regression.stderr)
        critical = float(
            stats.t.ppf(0.5 + request.confidence_level / 2.0, session_count - 2)
        )
        margin = critical * standard_error
        results.append(
            LongitudinalPlayerAssociationV1(
                player_id=str(player_id),
                session_count=session_count,
                estimate_per_order_unit=estimate,
                direction=_direction(estimate),
                state="available",
                standard_error=standard_error,
                ci_lower=estimate - margin,
                ci_upper=estimate + margin,
                p_value=float(regression.pvalue),
                r_squared=float(regression.rvalue**2),
                first_to_last_change=_first_to_last_change(group),
            )
        )
    return tuple(results)


def _design_matrix(
    cells: pd.DataFrame, request: LongitudinalSessionRequestV1
) -> tuple[np.ndarray, tuple[str, ...]]:
    numeric = cells[["order_value", *request.confounders]].astype(float)
    categorical_columns = ["player_id", *request.strata]
    categorical = pd.get_dummies(
        cells[categorical_columns].astype(str),
        columns=categorical_columns,
        drop_first=True,
        dtype=float,
    )
    design = pd.concat(
        [
            pd.Series(1.0, index=cells.index, name="intercept"),
            numeric,
            categorical,
        ],
        axis=1,
    )
    return design.to_numpy(dtype=float), tuple(str(item) for item in design.columns)


def clustered_pooled_association(
    cells: pd.DataFrame, request: LongitudinalSessionRequestV1
) -> tuple[PooledAssociationV1 | None, str | None, tuple[str, ...]]:
    """Fit player-FE OLS with a finite-cluster corrected sandwich covariance."""
    cluster_labels = cells["player_id"].astype(str).to_numpy()
    clusters = tuple(sorted(set(cluster_labels)))
    if len(clusters) < request.minimum_player_clusters:
        return None, "insufficient_player_clusters", ()
    matrix, terms = _design_matrix(cells, request)
    values = cells["metric_value"].to_numpy(dtype=float)
    observations, parameter_count = matrix.shape
    if (
        observations <= parameter_count
        or np.linalg.matrix_rank(matrix) < parameter_count
    ):
        return None, "rank_deficient_session_design", terms
    bread = np.linalg.inv(matrix.T @ matrix)
    coefficients = bread @ matrix.T @ values
    residuals = values - matrix @ coefficients
    meat = np.zeros((parameter_count, parameter_count), dtype=float)
    for cluster in clusters:
        selected = cluster_labels == cluster
        score = matrix[selected].T @ residuals[selected]
        meat += np.outer(score, score)
    correction = (len(clusters) / (len(clusters) - 1)) * (
        (observations - 1) / (observations - parameter_count)
    )
    covariance = correction * bread @ meat @ bread
    variance = float(covariance[1, 1])
    if not np.isfinite(variance) or variance <= 0:
        return None, "degenerate_clustered_variance", terms
    standard_error = sqrt(variance)
    estimate = float(coefficients[1])
    degrees_of_freedom = len(clusters) - 1
    critical = float(
        stats.t.ppf(0.5 + request.confidence_level / 2.0, degrees_of_freedom)
    )
    statistic = estimate / standard_error
    p_value = float(2 * stats.t.sf(abs(statistic), degrees_of_freedom))
    return (
        PooledAssociationV1(
            method="ud-cluster-robust-fe/1",
            estimate_per_order_unit=estimate,
            standard_error=standard_error,
            confidence_interval_low=estimate - critical * standard_error,
            confidence_interval_high=estimate + critical * standard_error,
            p_value=p_value,
            confidence_level=request.confidence_level,
            cluster_count=len(clusters),
            session_cell_count=observations,
        ),
        None,
        terms,
    )


def _improvement_probability(
    estimate: float, standard_error: float, request: LongitudinalSessionRequestV1
) -> float | None:
    """Return P(direction-consistent trend), or ``None`` without a direction.

    ``descriptive_only`` deliberately yields ``None``: a probability of
    improvement is undefined when the request declines to say which way is
    better, and inventing one would be the claim G1-D1 forbids.
    """

    if request.direction == "descriptive_only":
        return None
    sign = 1.0 if request.direction == "higher_is_better" else -1.0
    return float(stats.norm.cdf(sign * estimate / standard_error))


def dersimonian_laird_pooled_association(
    cells: pd.DataFrame,
    associations: tuple[LongitudinalPlayerAssociationV1, ...],
    request: LongitudinalSessionRequestV1,
) -> tuple[PooledAssociationV1 | None, str | None, tuple[str, ...]]:
    """Synthesize per-player slopes by inverse variance with DerSimonian-Laird."""
    clusters = tuple(sorted(set(cells["player_id"].astype(str))))
    if len(clusters) < request.minimum_player_clusters:
        return None, "insufficient_player_clusters", ()
    contributors = tuple(
        item
        for item in associations
        if item.state == "available"
        and item.estimate_per_order_unit is not None
        and item.standard_error is not None
        and item.standard_error > 0
    )
    terms = tuple(item.player_id for item in contributors)
    if len(contributors) < 2:
        return None, "insufficient_estimable_player_slopes", terms
    slopes = np.asarray(
        [float(item.estimate_per_order_unit or 0.0) for item in contributors],
        dtype=np.float64,
    )
    variances = np.square(
        np.asarray(
            [float(item.standard_error or 0.0) for item in contributors],
            dtype=np.float64,
        )
    )
    weights = 1.0 / variances
    fixed = float(np.average(slopes, weights=weights))
    q_statistic = float(np.sum(weights * np.square(slopes - fixed)))
    degrees = len(slopes) - 1
    c_value = float(weights.sum() - np.square(weights).sum() / weights.sum())
    tau_squared = max(0.0, (q_statistic - degrees) / c_value) if c_value else 0.0
    random_weights = 1.0 / (variances + tau_squared)
    estimate = float(np.average(slopes, weights=random_weights))
    standard_error = float(1 / np.sqrt(random_weights.sum()))
    if not np.isfinite(standard_error) or standard_error <= 0:
        return None, "degenerate_random_effects_variance", terms
    critical = float(stats.norm.ppf(0.5 + request.confidence_level / 2.0))
    i_squared = (
        max(0.0, (q_statistic - degrees) / q_statistic) * 100.0 if q_statistic else 0.0
    )
    return (
        PooledAssociationV1(
            method="dl-random-effects/1",
            estimate_per_order_unit=estimate,
            standard_error=standard_error,
            confidence_interval_low=estimate - critical * standard_error,
            confidence_interval_high=estimate + critical * standard_error,
            confidence_level=request.confidence_level,
            cluster_count=len(contributors),
            session_cell_count=len(cells),
            tau_squared=tau_squared,
            q_statistic=q_statistic,
            i_squared_pct=min(100.0, i_squared),
            improvement_probability=_improvement_probability(
                estimate, standard_error, request
            ),
        ),
        None,
        terms,
    )


__all__ = [
    "clustered_pooled_association",
    "dersimonian_laird_pooled_association",
    "player_associations",
]
