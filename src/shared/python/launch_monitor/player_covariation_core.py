"""Pure numerical core for identity-safe player covariation analysis.

Ported from UpstreamDrift
``src/shared/python/launch_monitor/player_covariation_core.py`` (427 lines)
under ADR-0046 Stage 1 — step **P18** of the ADR-0046 G1 port plan
(UpstreamDrift ``docs/adr/0048-launch-monitor-port-plan.md``). The
implementation is UpstreamDrift's, carried over rather than reimplemented; its
authors retain authorship. See
:mod:`shared.python.launch_monitor.player_covariation_types` for why P18 is a
**union port** of UpstreamDrift's trio with this repository's
``rate_of_closure`` trio rather than a plain port.

Union decisions carried by this module
--------------------------------------
:func:`covariation_backing_frame`
    Folded in from ``rate_of_closure.player_covariation``, whose
    ``PlayerCovariationAnalysis.backing_data`` is the second of the two fields
    G0.1's D26 pin records as existing only on the ``rate_of_closure`` side. It
    is a 160x6 frame of ``source_index``, ``player_id``, ``x``, ``y``,
    ``centered_x`` and ``centered_y`` — the per-row values a caller needs to
    plot the within-player centring the analysis performed, which
    UpstreamDrift's result cannot express.

    It is folded in as a **function, not a result field**. UpstreamDrift's
    postcondition is that a result carries content-addressed backing
    *references* and never copies source values into the output document, and
    that is what keeps a covariation result publishable next to this
    repository's private-corpus boundary (ADR-0046's "no private rows in
    project files"). Embedding 160 raw rows in the wire result would delete
    that guarantee to gain values the caller already holds. Exposing the same
    frame as a derivation over the caller's own input keeps both: the
    capability is available to anyone who could have computed it anyway, and
    the result document stays row-free. The column names, ordering, dtypes and
    row order match ``rate_of_closure``'s ``backing_data`` exactly, so a
    consumer of that field can be pointed here without reshaping.

:func:`player_association_frame`
    Folded in from ``rate_of_closure``'s ``PlayerCovariationAnalysis.
    per_player``, which is a ``pandas.DataFrame`` where UpstreamDrift returns a
    tuple of models. The models are strictly richer (typed ``state`` and
    ``reason_code`` where ``rate_of_closure`` has one ``status`` string), so
    they stay the wire representation — but ``rate_of_closure``'s docstring
    names "export-friendly tables" as its purpose, and a DataFrame is what an
    export or a plot wants. This projects the wire models back onto exactly
    ``rate_of_closure``'s eleven columns.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from shared.python.launch_monitor.player_covariation_types import (
    MIN_FISHER_SAMPLES,
    AssociationEstimateV1,
    AssociationUnavailableReason,
    CovariationMissingnessV1,
    MetaAnalysisSummaryV1,
    PlayerAssociationV1,
    PlayerCovariationRequestV1,
)

_EPSILON = np.finfo(float).eps
_REPORT_DECIMAL_PLACES = 12

PLAYER_ASSOCIATION_FRAME_COLUMNS: tuple[str, ...] = (
    "player_id",
    "sample_count",
    "pearson_r",
    "spearman_r",
    "slope",
    "intercept",
    "r_squared",
    "ci_lower",
    "ci_upper",
    "status",
    "fixed_weight",
    "random_weight",
)

BACKING_FRAME_COLUMNS: tuple[str, ...] = (
    "source_index",
    "player_id",
    "x",
    "y",
    "centered_x",
    "centered_y",
)


def _reported_float(value: float) -> float:
    """Normalize public floats across supported BLAS/scientific stacks."""

    rounded = round(float(value), _REPORT_DECIMAL_PLACES)
    return 0.0 if rounded == 0 else rounded


@dataclass(frozen=True)
class PairStatistics:
    """Internal selected-pair estimates before evidence is attached."""

    pooled: AssociationEstimateV1
    within_player: AssociationEstimateV1
    between_player: AssociationEstimateV1
    per_player: tuple[PlayerAssociationV1, ...]
    meta_analysis: MetaAnalysisSummaryV1
    missingness: CovariationMissingnessV1
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _PreparedPair:
    """Numeric pair data retaining all explicitly identified players."""

    identified: pd.DataFrame
    complete: pd.DataFrame
    missing_by_variable: dict[str, int]
    non_numeric_by_variable: dict[str, int]
    non_finite_by_variable: dict[str, int]
    excluded_by_reason: dict[str, int]


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Columns not present in dataset: {', '.join(missing)}")


def _prepare_pair(
    frame: pd.DataFrame, request: PlayerCovariationRequestV1
) -> _PreparedPair:
    selected = (request.player_column, request.x_column, request.y_column)
    _require_columns(frame, selected)
    raw_x = frame[request.x_column]
    raw_y = frame[request.y_column]
    numeric_x = pd.to_numeric(raw_x, errors="coerce")
    numeric_y = pd.to_numeric(raw_y, errors="coerce")
    identity = frame[request.player_column].where(
        frame[request.player_column].notna(), ""
    )
    identity = identity.astype(str).str.strip()
    valid_identity = identity.ne("")
    finite_x = numeric_x.notna() & np.isfinite(numeric_x)
    finite_y = numeric_y.notna() & np.isfinite(numeric_y)
    complete_mask = valid_identity & finite_x & finite_y
    prepared = pd.DataFrame(
        {"player_id": identity, "x": numeric_x, "y": numeric_y},
        index=frame.index,
    )
    identified = prepared.loc[valid_identity].copy()
    complete = prepared.loc[complete_mask].copy()
    missing = {
        request.x_column: int(raw_x.isna().sum()),
        request.y_column: int(raw_y.isna().sum()),
    }
    non_numeric = {
        request.x_column: int((raw_x.notna() & numeric_x.isna()).sum()),
        request.y_column: int((raw_y.notna() & numeric_y.isna()).sum()),
    }
    non_finite = {
        request.x_column: int((numeric_x.notna() & ~np.isfinite(numeric_x)).sum()),
        request.y_column: int((numeric_y.notna() & ~np.isfinite(numeric_y)).sum()),
    }
    exclusions = {
        "blank_player_identity": int((~valid_identity).sum()),
        "pairwise_incomplete": int((valid_identity & (~finite_x | ~finite_y)).sum()),
    }
    return _PreparedPair(
        identified,
        complete,
        missing,
        non_numeric,
        non_finite,
        exclusions,
    )


def _unavailable_estimate(
    count: int,
    groups: int,
    reason: AssociationUnavailableReason,
) -> AssociationEstimateV1:
    return AssociationEstimateV1(
        state="unavailable",
        reason_code=reason,
        sample_count=count,
        group_count=groups,
    )


def _estimate_reason(
    values_x: np.ndarray,
    values_y: np.ndarray,
    minimum: int,
    insufficient_reason: AssociationUnavailableReason,
) -> AssociationUnavailableReason | None:
    if len(values_x) < minimum:
        return insufficient_reason
    constant_x = bool(np.ptp(values_x) <= _EPSILON)
    constant_y = bool(np.ptp(values_y) <= _EPSILON)
    if constant_x and constant_y:
        return "constant_both"
    if constant_x:
        return "constant_x"
    if constant_y:
        return "constant_y"
    return None


def _fisher_interval(
    coefficient: float, count: int, confidence: float
) -> tuple[float, float]:
    transformed = np.arctanh(np.clip(coefficient, -0.999999, 0.999999))
    margin = stats.norm.ppf(0.5 + confidence / 2) / np.sqrt(count - 3)
    return (
        _reported_float(np.tanh(transformed - margin)),
        _reported_float(np.tanh(transformed + margin)),
    )


def _estimate(
    values_x: np.ndarray,
    values_y: np.ndarray,
    *,
    group_count: int,
    minimum: int,
    confidence: float,
    include_interval: bool,
    insufficient_reason: AssociationUnavailableReason = "insufficient_samples",
) -> AssociationEstimateV1:
    reason = _estimate_reason(values_x, values_y, minimum, insufficient_reason)
    if reason is not None:
        return _unavailable_estimate(len(values_x), group_count, reason)
    pearson = _reported_float(
        np.clip(stats.pearsonr(values_x, values_y).statistic, -1, 1)
    )
    spearman = _reported_float(
        np.clip(stats.spearmanr(values_x, values_y).statistic, -1, 1)
    )
    slope, intercept = np.polyfit(values_x, values_y, deg=1)
    interval = (
        _fisher_interval(pearson, len(values_x), confidence)
        if include_interval and len(values_x) >= MIN_FISHER_SAMPLES
        else (None, None)
    )
    return AssociationEstimateV1(
        state="available",
        sample_count=len(values_x),
        group_count=group_count,
        pearson_r=pearson,
        spearman_r=spearman,
        slope=_reported_float(slope),
        intercept=_reported_float(intercept),
        r_squared=_reported_float(min(1.0, pearson**2)),
        ci_lower=interval[0],
        ci_upper=interval[1],
    )


def _frame_estimate(
    frame: pd.DataFrame,
    request: PlayerCovariationRequestV1,
    *,
    minimum: int | None = None,
    include_interval: bool = True,
    insufficient_reason: AssociationUnavailableReason = "insufficient_samples",
) -> AssociationEstimateV1:
    return _estimate(
        frame["x"].to_numpy(float),
        frame["y"].to_numpy(float),
        group_count=int(frame["player_id"].nunique()),
        minimum=request.min_samples if minimum is None else minimum,
        confidence=request.confidence_level,
        include_interval=include_interval,
        insufficient_reason=insufficient_reason,
    )


def _player_estimates(
    prepared: _PreparedPair, request: PlayerCovariationRequestV1
) -> tuple[PlayerAssociationV1, ...]:
    items: list[PlayerAssociationV1] = []
    for player_id in sorted(prepared.identified["player_id"].unique()):
        group = prepared.complete.loc[prepared.complete["player_id"] == player_id]
        items.append(
            PlayerAssociationV1(
                player_id=player_id,
                estimate=_frame_estimate(group, request),
            )
        )
    return tuple(items)


def _weighted_effect(
    effects: np.ndarray, weights: np.ndarray, confidence: float
) -> tuple[float, float, float]:
    mean = float(np.average(effects, weights=weights))
    margin = float(stats.norm.ppf(0.5 + confidence / 2) / np.sqrt(weights.sum()))
    return (
        _reported_float(np.tanh(mean)),
        _reported_float(np.tanh(mean - margin)),
        _reported_float(np.tanh(mean + margin)),
    )


def _unavailable_meta(items: tuple[PlayerAssociationV1, ...]) -> MetaAnalysisSummaryV1:
    eligible = [item for item in items if item.estimate.state == "available"]
    return MetaAnalysisSummaryV1(
        state="unavailable",
        reason_code="insufficient_eligible_players",
        contributor_count=len(eligible),
        total_sample_count=sum(item.estimate.sample_count for item in eligible),
    )


def _meta_analysis(
    items: tuple[PlayerAssociationV1, ...], confidence: float
) -> tuple[MetaAnalysisSummaryV1, tuple[PlayerAssociationV1, ...]]:
    eligible = [item for item in items if item.estimate.state == "available"]
    if len(eligible) < 2:
        return _unavailable_meta(items), items
    correlations = np.array([item.estimate.pearson_r for item in eligible], dtype=float)
    effects = np.arctanh(np.clip(correlations, -0.999999, 0.999999))
    variances = 1 / (
        np.array([item.estimate.sample_count for item in eligible], dtype=float) - 3
    )
    fixed_weights = 1 / variances
    fixed_mean = float(np.average(effects, weights=fixed_weights))
    diff = effects - fixed_mean
    q_statistic = float(np.vdot(fixed_weights, diff * diff))
    degrees = len(eligible) - 1
    denominator = fixed_weights.sum() - (
        np.vdot(fixed_weights, fixed_weights) / fixed_weights.sum()
    )
    tau_squared = max(0.0, (q_statistic - degrees) / denominator)
    random_weights = 1 / (variances + tau_squared)
    i_squared = (
        max(0.0, (q_statistic - degrees) / q_statistic) * 100 if q_statistic else 0.0
    )
    fixed_effect = _weighted_effect(effects, fixed_weights, confidence)
    random_effect = _weighted_effect(effects, random_weights, confidence)
    summary = MetaAnalysisSummaryV1(
        state="available",
        contributor_count=len(eligible),
        total_sample_count=sum(item.estimate.sample_count for item in eligible),
        fixed_effect_r=fixed_effect[0],
        fixed_ci_lower=fixed_effect[1],
        fixed_ci_upper=fixed_effect[2],
        random_effect_r=random_effect[0],
        random_ci_lower=random_effect[1],
        random_ci_upper=random_effect[2],
        tau_squared=_reported_float(tau_squared),
        q_statistic=_reported_float(q_statistic),
        i_squared_pct=_reported_float(i_squared),
    )
    return summary, _attach_weights(items, eligible, fixed_weights, random_weights)


def _attach_weights(
    items: tuple[PlayerAssociationV1, ...],
    eligible: list[PlayerAssociationV1],
    fixed_weights: np.ndarray,
    random_weights: np.ndarray,
) -> tuple[PlayerAssociationV1, ...]:
    fixed = fixed_weights / fixed_weights.sum()
    random = random_weights / random_weights.sum()
    weight_by_player = {
        item.player_id: (
            _reported_float(fixed[index]),
            _reported_float(random[index]),
        )
        for index, item in enumerate(eligible)
    }
    return tuple(
        item.model_copy(
            update={
                "fixed_weight": weight_by_player[item.player_id][0],
                "random_weight": weight_by_player[item.player_id][1],
            }
        )
        if item.player_id in weight_by_player
        else item
        for item in items
    )


def _centered(complete: pd.DataFrame) -> pd.DataFrame:
    centered = complete.copy()
    grouped = centered.groupby("player_id", sort=False)
    centered["x"] = centered["x"] - grouped["x"].transform("mean")
    centered["y"] = centered["y"] - grouped["y"].transform("mean")
    return centered


def _between(
    complete: pd.DataFrame, request: PlayerCovariationRequestV1
) -> AssociationEstimateV1:
    means = complete.groupby("player_id", sort=True)[["x", "y"]].mean()
    means = means.reset_index()
    return _frame_estimate(
        means,
        request,
        minimum=2,
        include_interval=True,
        insufficient_reason="insufficient_groups",
    )


def _missingness(
    frame: pd.DataFrame,
    prepared: _PreparedPair,
    per_player: tuple[PlayerAssociationV1, ...],
) -> CovariationMissingnessV1:
    excluded_players = Counter(
        item.estimate.reason_code
        for item in per_player
        if item.estimate.reason_code is not None
    )
    return CovariationMissingnessV1(
        input_row_count=len(frame),
        pairwise_complete_row_count=len(prepared.complete),
        missing_by_variable=prepared.missing_by_variable,
        non_numeric_by_variable=prepared.non_numeric_by_variable,
        non_finite_by_variable=prepared.non_finite_by_variable,
        excluded_by_reason=prepared.excluded_by_reason,
        eligible_player_count=sum(
            item.estimate.state == "available" for item in per_player
        ),
        excluded_player_count_by_reason={
            str(reason): count for reason, count in sorted(excluded_players.items())
        },
    )


def _warnings(
    frame: pd.DataFrame,
    prepared: _PreparedPair,
    per_player: tuple[PlayerAssociationV1, ...],
    pooled: AssociationEstimateV1,
    within: AssociationEstimateV1,
    meta: MetaAnalysisSummaryV1,
) -> tuple[str, ...]:
    warnings = [
        "Associations are observational and do not establish causality.",
        "Pooled and between-player estimates do not substitute for "
        "within-player effects.",
    ]
    removed = len(frame) - len(prepared.complete)
    if removed:
        warnings.append(f"{removed} rows were excluded from the complete pair.")
    excluded = sum(item.estimate.state == "unavailable" for item in per_player)
    if excluded:
        warnings.append(f"{excluded} players were excluded from Fisher-z synthesis.")
    if meta.state == "unavailable":
        warnings.append("Meta-analysis requires at least two eligible players.")
    if (
        pooled.pearson_r is not None
        and within.pearson_r is not None
        and np.sign(pooled.pearson_r) != np.sign(within.pearson_r)
    ):
        warnings.append(
            "Possible aggregation reversal: pooled and within-player Pearson "
            "correlations have opposite signs; inspect group structure."
        )
    return tuple(warnings)


def compute_pair_statistics(
    frame: pd.DataFrame, request: PlayerCovariationRequestV1
) -> PairStatistics:
    """Return selected-pair statistics; the input frame is not mutated."""

    prepared = _prepare_pair(frame, request)
    per_player = _player_estimates(prepared, request)
    meta, per_player = _meta_analysis(per_player, request.confidence_level)
    pooled = _frame_estimate(prepared.complete, request)
    within = _frame_estimate(
        _centered(prepared.complete), request, include_interval=False
    )
    between = _between(prepared.complete, request)
    return PairStatistics(
        pooled=pooled,
        within_player=within,
        between_player=between,
        per_player=per_player,
        meta_analysis=meta,
        missingness=_missingness(frame, prepared, per_player),
        warnings=_warnings(frame, prepared, per_player, pooled, within, meta),
    )


def covariation_backing_frame(
    frame: pd.DataFrame, request: PlayerCovariationRequestV1
) -> pd.DataFrame:
    """Return the pairwise-complete, group-mean-centred rows for one pair.

    Union port of ``rate_of_closure.player_covariation``'s ``backing_data``
    field; see this module's docstring for why the capability is a function
    rather than a field on :class:`PlayerCovariationResultV1`. Columns, order
    and row order match that field exactly:
    :data:`BACKING_FRAME_COLUMNS`, sorted by ``player_id`` with a stable sort
    so the original row order survives inside each player.

    The input frame is not mutated.
    """

    prepared = _prepare_pair(frame, request)
    ordered = prepared.complete.sort_values("player_id", kind="stable")
    grouped = ordered.groupby("player_id", sort=False)
    centred = ordered.assign(
        centered_x=ordered["x"] - grouped["x"].transform("mean"),
        centered_y=ordered["y"] - grouped["y"].transform("mean"),
    )
    return centred.reset_index(names="source_index")


def player_association_frame(
    per_player: tuple[PlayerAssociationV1, ...],
) -> pd.DataFrame:
    """Project per-player wire models onto an export-friendly table.

    Union port of ``rate_of_closure.player_covariation``'s ``per_player``
    field, which is a ``pandas.DataFrame`` where this layer's wire
    representation is a tuple of models. Columns are
    :data:`PLAYER_ASSOCIATION_FRAME_COLUMNS`, matching that field.
    ``status`` carries ``"ok"`` for an available estimate and the estimate's
    ``reason_code`` otherwise, which is the exact vocabulary
    ``rate_of_closure`` uses.
    """

    rows = [
        {
            "player_id": item.player_id,
            "sample_count": item.estimate.sample_count,
            "pearson_r": item.estimate.pearson_r,
            "spearman_r": item.estimate.spearman_r,
            "slope": item.estimate.slope,
            "intercept": item.estimate.intercept,
            "r_squared": item.estimate.r_squared,
            "ci_lower": item.estimate.ci_lower,
            "ci_upper": item.estimate.ci_upper,
            "status": (
                "ok"
                if item.estimate.state == "available"
                else item.estimate.reason_code
            ),
            "fixed_weight": item.fixed_weight,
            "random_weight": item.random_weight,
        }
        for item in per_player
    ]
    return pd.DataFrame(rows, columns=list(PLAYER_ASSOCIATION_FRAME_COLUMNS))


__all__ = [
    "BACKING_FRAME_COLUMNS",
    "PLAYER_ASSOCIATION_FRAME_COLUMNS",
    "PairStatistics",
    "compute_pair_statistics",
    "covariation_backing_frame",
    "player_association_frame",
]
