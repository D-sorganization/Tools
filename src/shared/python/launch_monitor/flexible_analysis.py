"""Flexible, provenance-preserving correlation and OLS analysis.

Ported from UpstreamDrift
``src/shared/python/launch_monitor/flexible_analysis.py`` (415 lines) under
ADR-0046 Stage 1 — step **P10** of the ADR-0046 G1 port plan (UpstreamDrift
``docs/adr/0048-launch-monitor-port-plan.md``). The implementation is
UpstreamDrift's, carried over unchanged rather than reimplemented; its authors
retain authorship. No behaviour is added, removed, or limited by the move.

The public contract in this module is intentionally UI-neutral so desktop,
web, notebooks, and API adapters can present the same numerical result.

Measured against its ``rate_of_closure`` twin, not assumed
---------------------------------------------------------
ADR-0046 listed flexible analysis as an UpstreamDrift-only capability;
Amendment 1 corrected that. ``rate_of_closure.launch_monitor_analysis`` plus
``_launch_monitor_analysis_statistics`` and ``_launch_monitor_analysis_types``
carry the same correlation + OLS + group analysis behind
``AnalysisRequest``/``AnalysisResult`` and ``analyze_launch_monitor_data``, and
the two stacks define six identically named frozen dataclasses. The port plan
therefore held this row — and everything above it — until the pair had actually
been measured.

It has been. UpstreamDrift#9372 landed the G0.1 gate
``tests/integration/launch_monitor_drift/test_flexible_analysis_drift.py``,
whose thirteen gates run both stacks over the same 160-shot session. Seven of
them are AGREE pins asserted to delta exactly ``0.0``: the three Pearson
correlations (coefficient, raw p, adjusted p, both Fisher-z bounds), every
estimate of the four-parameter OLS, the six shared residual diagnostics, the
four ``group_by`` fits, and — independently arrived at on both sides — the same
``DatasetSummary.fingerprint_sha256`` digest. The six remaining gates pin
divergences D15-D20. That measurement, not a reading, is why P10 and P11 could
proceed; the two stacks agree numerically, so this port carries no arithmetic
across a gap.

**No re-export in either direction.** The names here collide with
``rate_of_closure`` names that D15-D20 prove are not the same thing. The
separate package is the containment; do not add a convenience alias between
them.

Owner rulings applied here (ADR-0048, "Owner Rulings (2026-09-02)")
--------------------------------------------------------------------
The port that landed this module carried UpstreamDrift's behaviour verbatim
and pinned two "before" cases so a follow-up's diff would be visible rather
than silent. This module now *is* that follow-up; both rulings are applied.

* **D15 — FDR multiplicity denominator.** ``_correlations`` used to compute
  the Benjamini-Hochberg adjustment over *every* requested predictor's raw p
  value and only afterwards blank the estimates whose pair count fell below
  ``min_samples``. An under-sampled predictor therefore inflated the
  denominator of the predictors that survived. Per the ruling, the canonical
  layer now excludes under-sampled predictors from the correction pool
  *before* correcting — UD's count-all behaviour was a defect, not a
  preserved method — by feeding ``_adjust_p_values`` ``nan`` in place of an
  under-sampled predictor's raw p; ``_adjust_p_values`` already drops
  non-finite entries from its pool, so no change was needed there. Coefficient
  and p-value arithmetic for adequately sampled predictors is unaffected;
  only the correction denominator moves.
* **D17 — boolean columns.** ``pd.to_numeric`` still projects ``True``/
  ``False`` to 1.0/0.0 and the column is still analysed as though it were
  native numeric — the ruling preserves that capability. What changes is the
  silence: Tools#4901 already applied D17 one layer down, so
  :class:`~shared.python.launch_monitor.relationships.CorrelationResult`
  reports ``boolean_projected``; ``_correlations`` now reads that label off
  the ``compute_correlations`` result it already holds and carries it onto
  each :class:`CorrelationEstimate` as ``is_boolean_projected`` rather than
  computing a new one. No arithmetic changes. Scope: this only covers
  correlation-mode predictors and the outcome-vs-predictor pairs
  ``_correlations`` produces — ``_regression`` performs its own independent
  ``pd.to_numeric`` cast and is unaffected (out of scope for this ruling as
  applied here; a boolean predictor entering ``analysis_mode="regression"``
  is still analysed, still unlabelled).

Both behaviours were pinned as the "before" side of this diff by
``test_flexible_analysis.py``
(``test_undersampled_predictor_still_counts_in_the_fdr_denominator`` and
``test_boolean_predictor_is_silently_projected_to_zero_one``); those two
tests now assert the "after" contract instead.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from math import sqrt
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy import stats

from shared.python.launch_monitor.relationships import compute_correlations
from shared.python.launch_monitor.schema import METRICS

AnalysisMode = Literal["correlation", "regression", "comprehensive"]
MissingPolicy = Literal["pairwise", "listwise", "fail"]
CorrelationMethod = Literal["pearson", "spearman", "kendall"]
CONTRACT_VERSION = "1.0.0"

__all__ = [
    "CONTRACT_VERSION",
    "AnalysisMode",
    "CoefficientEstimate",
    "CorrelationEstimate",
    "CorrelationMethod",
    "DatasetSummary",
    "FlexibleAnalysisRequest",
    "FlexibleAnalysisResult",
    "GroupAnalysis",
    "MissingPolicy",
    "RegressionEstimate",
    "ResidualDiagnostics",
    "analyze_variables",
]


@dataclass(frozen=True)
class FlexibleAnalysisRequest:
    """User-selected analysis settings shared by every presentation layer."""

    outcome: str
    predictors: tuple[str, ...]
    analysis_mode: AnalysisMode = "comprehensive"
    correlation_method: CorrelationMethod = "pearson"
    missing_policy: MissingPolicy = "pairwise"
    group_by: str | None = None
    confidence_level: float = 0.95
    min_samples: int = 10
    allow_aggregate: bool = False

    def __post_init__(self) -> None:
        if not self.outcome.strip():
            raise ValueError("outcome must be non-empty")
        if not self.predictors:
            raise ValueError("At least one predictor is required")
        if len(set(self.predictors)) != len(self.predictors):
            raise ValueError("predictors must be unique")
        if self.analysis_mode not in {"correlation", "regression", "comprehensive"}:
            raise ValueError("Unknown analysis_mode")
        if self.correlation_method not in {"pearson", "spearman", "kendall"}:
            raise ValueError("Unknown correlation_method")
        if self.missing_policy not in {"pairwise", "listwise", "fail"}:
            raise ValueError("Unknown missing_policy")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1")
        if self.min_samples < 3:
            raise ValueError("min_samples must be at least 3")


@dataclass(frozen=True)
class DatasetSummary:
    row_count: int
    complete_row_count: int
    selected_columns: tuple[str, ...]
    monitor_vendors: tuple[str, ...]
    session_ids: tuple[str, ...]
    observation_kinds: tuple[str, ...]
    fingerprint_sha256: str


@dataclass(frozen=True)
class CorrelationEstimate:
    """One predictor's correlation against the request's outcome.

    ``is_boolean_projected`` is ``True`` when ``predictor`` was a boolean
    column analysed via the explicit 0/1 projection (owner ruling D17). The
    label is carried through from
    :attr:`~shared.python.launch_monitor.relationships.CorrelationResult.boolean_projected`
    — computed one layer down, in
    :func:`~shared.python.launch_monitor.relationships.compute_correlations` —
    not recomputed here.
    """

    predictor: str
    coefficient: float
    p_value: float
    adjusted_p_value: float
    ci_lower: float
    ci_upper: float
    sample_count: int
    method: str
    is_boolean_projected: bool


@dataclass(frozen=True)
class CoefficientEstimate:
    estimate: float
    standard_error: float
    t_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True)
class ResidualDiagnostics:
    rmse: float
    mae: float
    residual_mean: float
    residual_std: float
    durbin_watson: float
    jarque_bera_p_value: float
    influential_count: int


@dataclass(frozen=True)
class RegressionEstimate:
    sample_count: int
    r_squared: float
    adjusted_r_squared: float
    coefficients: dict[str, CoefficientEstimate]
    residual_diagnostics: ResidualDiagnostics


@dataclass(frozen=True)
class GroupAnalysis:
    group_value: str
    row_count: int
    correlations: tuple[CorrelationEstimate, ...]
    regression: RegressionEstimate | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class FlexibleAnalysisResult:
    request: FlexibleAnalysisRequest
    dataset: DatasetSummary
    correlations: tuple[CorrelationEstimate, ...]
    regression: RegressionEstimate | None
    groups: tuple[GroupAnalysis, ...]
    units: dict[str, str]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe result without silently serializing NaN."""

        def clean(value: Any) -> Any:
            if isinstance(value, dict):
                return {str(key): clean(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [clean(item) for item in value]
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value

        payload = cast(dict[str, Any], clean(asdict(self)))
        payload["contract_version"] = CONTRACT_VERSION
        return payload


def _string_values(frame: pd.DataFrame, column: str) -> tuple[str, ...]:
    if column not in frame:
        return ()
    values = frame[column].dropna().astype(str)
    return tuple(sorted(value for value in values.unique() if value.strip()))


def _fingerprint(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    identity = tuple(
        column
        for column in ("shot_id", "session_id", "source_row", "monitor_vendor")
        if column in frame and column not in columns
    )
    selected = identity + columns
    records = [
        {
            column: (
                None
                if pd.isna(value)
                else value.item()
                if hasattr(value, "item")
                else value
            )
            for column, value in row.items()
        }
        for row in frame[list(selected)].to_dict(orient="records")
    ]
    serialized = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


def _adjust_p_values(values: list[float]) -> list[float]:
    adjusted = [float("nan")] * len(values)
    finite = [
        (index, value) for index, value in enumerate(values) if np.isfinite(value)
    ]
    if not finite:
        return adjusted
    ordered = sorted(finite, key=lambda item: item[1])
    raw = np.asarray([item[1] for item in ordered], dtype=float)
    ranks = np.arange(1, len(raw) + 1)
    corrected = np.minimum.accumulate((raw * len(raw) / ranks)[::-1])[::-1]
    for (index, _), value in zip(ordered, np.clip(corrected, 0.0, 1.0), strict=True):
        adjusted[index] = float(value)
    return adjusted


def _correlation_interval(
    coefficient: float, sample_count: int, confidence_level: float
) -> tuple[float, float]:
    if sample_count <= 3 or not np.isfinite(coefficient):
        return float("nan"), float("nan")
    clipped = float(np.clip(coefficient, -0.999999, 0.999999))
    z_value = np.arctanh(clipped)
    margin = stats.norm.ppf(0.5 + confidence_level / 2.0) / sqrt(sample_count - 3)
    return float(np.tanh(z_value - margin)), float(np.tanh(z_value + margin))


def _correlations(
    frame: pd.DataFrame, request: FlexibleAnalysisRequest
) -> tuple[CorrelationEstimate, ...]:
    columns = (request.outcome, *request.predictors)
    input_frame = frame
    if request.missing_policy == "listwise":
        input_frame = frame.dropna(subset=list(columns))
    result = compute_correlations(
        input_frame, metrics=columns, method=request.correlation_method
    )
    raw_p = [
        float(result.p_values.loc[request.outcome, item]) for item in request.predictors
    ]
    counts = [
        int(result.pair_counts.loc[request.outcome, item])
        for item in request.predictors
    ]
    # D17 (ADR-0048 owner ruling, Tools#4901): compute_correlations already
    # labels a boolean-projected metric one layer down; carry that label
    # through rather than recomputing it.
    boolean_projected = {
        predictor: predictor in result.boolean_projected
        for predictor in request.predictors
    }
    # D15 (ADR-0048 owner ruling): exclude under-sampled predictors from the
    # Benjamini-Hochberg denominator *before* correcting, rather than
    # correcting over every requested predictor and only afterwards blanking
    # the ones whose pair count fell below min_samples (UD's behaviour,
    # which inflates the adjusted p value of every predictor that *does*
    # survive). _adjust_p_values already excludes non-finite entries from its
    # pool, so replacing an under-sampled predictor's raw p with nan here is
    # sufficient - no change to _adjust_p_values itself.
    correction_input = [
        raw if count >= request.min_samples else float("nan")
        for raw, count in zip(raw_p, counts, strict=True)
    ]
    adjusted = _adjust_p_values(correction_input)
    estimates: list[CorrelationEstimate] = []
    for predictor, p_value, adjusted_p, count in zip(
        request.predictors, raw_p, adjusted, counts, strict=True
    ):
        coefficient = float(result.coefficients.loc[request.outcome, predictor])
        if count < request.min_samples:
            coefficient = p_value = adjusted_p = float("nan")
        if request.correlation_method == "pearson":
            lower, upper = _correlation_interval(
                coefficient, count, request.confidence_level
            )
        else:
            lower = upper = float("nan")
        estimates.append(
            CorrelationEstimate(
                predictor,
                coefficient,
                p_value,
                adjusted_p,
                lower,
                upper,
                count,
                request.correlation_method,
                boolean_projected[predictor],
            )
        )
    return tuple(estimates)


def _regression(
    frame: pd.DataFrame, request: FlexibleAnalysisRequest
) -> RegressionEstimate:
    columns = (request.outcome, *request.predictors)
    numeric = frame[list(columns)].apply(pd.to_numeric, errors="coerce").dropna()
    count = len(numeric)
    parameter_count = len(request.predictors) + 1
    if count < max(request.min_samples, parameter_count + 2):
        raise ValueError("Too few complete observations for regression")
    y = numeric[request.outcome].to_numpy(float)
    x = numeric[list(request.predictors)].to_numpy(float)
    design = np.column_stack((np.ones(count), x))
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < parameter_count:
        raise ValueError("Regression design matrix is rank deficient")
    fitted = design @ beta
    residuals = y - fitted
    residual_sum = float(residuals @ residuals)
    diff = y - y.mean()
    # ⚡ Bolt: dot product avoids temporary allocation and is ~1.3x faster
    # than sum(**2)
    total_sum = float(diff @ diff)
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else float("nan")
    degrees_freedom = count - parameter_count
    adjusted = 1.0 - (1.0 - r_squared) * (count - 1) / degrees_freedom
    variance = residual_sum / degrees_freedom
    covariance = variance * np.linalg.inv(design.T @ design)
    standard_errors = np.sqrt(np.diag(covariance))
    t_values = beta / standard_errors
    p_values = 2.0 * stats.t.sf(np.abs(t_values), degrees_freedom)
    critical = stats.t.ppf(0.5 + request.confidence_level / 2.0, degrees_freedom)
    names = ("intercept", *request.predictors)
    coefficients = {
        name: CoefficientEstimate(
            float(beta[index]),
            float(standard_errors[index]),
            float(t_values[index]),
            float(p_values[index]),
            float(beta[index] - critical * standard_errors[index]),
            float(beta[index] + critical * standard_errors[index]),
        )
        for index, name in enumerate(names)
    }
    leverage = np.einsum(
        "ij,jk,ik->i", design, np.linalg.inv(design.T @ design), design
    )
    cooks = (residuals**2 / (parameter_count * variance)) * (
        leverage / np.maximum((1.0 - leverage) ** 2, np.finfo(float).eps)
    )
    jarque = stats.jarque_bera(residuals)
    denominator = float(residuals @ residuals)
    diagnostics = ResidualDiagnostics(
        rmse=float(np.sqrt(np.mean(residuals**2))),
        mae=float(np.mean(np.abs(residuals))),
        residual_mean=float(np.mean(residuals)),
        residual_std=float(np.std(residuals, ddof=parameter_count)),
        durbin_watson=(
            float(np.diff(residuals) @ np.diff(residuals) / denominator)
            if denominator > 0
            else float("nan")
        ),
        jarque_bera_p_value=float(jarque.pvalue),
        influential_count=int(np.count_nonzero(cooks > 4.0 / count)),
    )
    return RegressionEstimate(count, r_squared, adjusted, coefficients, diagnostics)


def analyze_variables(
    frame: pd.DataFrame, request: FlexibleAnalysisRequest
) -> FlexibleAnalysisResult:
    """Analyze arbitrary numeric variables while retaining dataset lineage."""

    if request.outcome in request.predictors:
        raise ValueError("outcome cannot also be a predictor")
    selected = (request.outcome, *request.predictors)
    required = set(selected)
    if request.group_by:
        required.add(request.group_by)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Columns not present: {sorted(missing)}")
    numeric = frame[list(selected)].apply(pd.to_numeric, errors="coerce")
    constants = [
        column for column in selected if numeric[column].dropna().nunique() < 2
    ]
    if constants:
        raise ValueError(f"Constant variables cannot be analyzed: {constants}")
    if request.missing_policy == "fail" and numeric.isna().any().any():
        raise ValueError("Selected variables contain missing or non-numeric values")

    vendors = _string_values(frame, "monitor_vendor")
    if any(column.startswith("source::") for column in selected) and len(vendors) > 1:
        raise ValueError("source fields cannot be pooled across multiple monitors")
    kinds = _string_values(frame, "observation_kind") or ("shot",)
    is_aggregate = any(kind.lower() != "shot" for kind in kinds)
    if is_aggregate and request.analysis_mode in {"regression", "comprehensive"}:
        raise ValueError("Aggregate observations cannot enter regression")
    if is_aggregate and not request.allow_aggregate:
        raise ValueError("Aggregate observations require allow_aggregate=True")

    warnings: list[str] = []
    if is_aggregate:
        warnings.append(
            "Aggregate correlations are descriptive only and may exhibit "
            "ecological bias."
        )
    if (
        request.correlation_method != "pearson"
        and request.analysis_mode != "regression"
    ):
        warnings.append(
            "Analytical confidence intervals are only reported for Pearson correlation."
        )
    complete = int(numeric.dropna().shape[0])
    dataset = DatasetSummary(
        len(frame),
        complete,
        selected,
        vendors,
        _string_values(frame, "session_id"),
        kinds,
        _fingerprint(frame, selected),
    )
    correlations = (
        _correlations(frame, request)
        if request.analysis_mode in {"correlation", "comprehensive"}
        else ()
    )
    regression = (
        _regression(frame, request)
        if request.analysis_mode in {"regression", "comprehensive"}
        else None
    )

    groups: list[GroupAnalysis] = []
    if request.group_by:
        ungrouped = replace(request, group_by=None)
        grouped = frame.dropna(subset=[request.group_by]).groupby(
            request.group_by, sort=True, observed=True
        )
        for value, group_frame in grouped:
            try:
                group_result = analyze_variables(group_frame, ungrouped)
                groups.append(
                    GroupAnalysis(
                        str(value),
                        len(group_frame),
                        group_result.correlations,
                        group_result.regression,
                        group_result.warnings,
                    )
                )
            except ValueError as error:
                groups.append(
                    GroupAnalysis(str(value), len(group_frame), (), None, (str(error),))
                )
    units = {
        column: METRICS[column].canonical_unit if column in METRICS else "source"
        for column in selected
    }
    return FlexibleAnalysisResult(
        request,
        dataset,
        correlations,
        regression,
        tuple(groups),
        units,
        tuple(warnings),
    )
