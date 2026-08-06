"""UI-neutral launch-monitor statistics matching the React contract."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass, replace
from hashlib import sha256
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy import stats

AnalysisMode = Literal["correlation", "regression", "comprehensive"]
CorrelationMethod = Literal["pearson", "spearman", "kendall"]
MissingPolicy = Literal["pairwise", "listwise", "fail"]
CONTRACT_VERSION = "1.0.0"


@dataclass(frozen=True)
class AnalysisRequest:
    outcome: str
    predictors: tuple[str, ...]
    analysis_mode: AnalysisMode = "comprehensive"
    correlation_method: CorrelationMethod = "pearson"
    missing_policy: MissingPolicy = "pairwise"
    group_by: str | None = None
    confidence_level: float = 0.95
    min_samples: int = 10
    allow_aggregate: bool = False


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
    predictor: str
    coefficient: float | None
    p_value: float | None
    adjusted_p_value: float | None
    ci_lower: float | None
    ci_upper: float | None
    sample_count: int
    method: str


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
    durbin_watson: float | None
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
class AnalysisResult:
    contract_version: str
    request: AnalysisRequest
    dataset: DatasetSummary
    correlations: tuple[CorrelationEstimate, ...]
    regression: RegressionEstimate | None
    groups: tuple[GroupAnalysis, ...]
    warnings: tuple[str, ...]

    def to_wire(self) -> dict[str, Any]:
        """Return the camel-case JSON structure consumed by the React twin."""

        def camel(name: str) -> str:
            head, *tail = name.split("_")
            return head + "".join(word.title() for word in tail)

        def convert(value: Any) -> Any:
            if is_dataclass(value) and not isinstance(value, type):
                return {
                    camel(item.name): convert(getattr(value, item.name))
                    for item in fields(value)
                }
            if isinstance(value, dict):
                return {str(key): convert(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(item) for item in value]
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value

        return cast(dict[str, Any], convert(self))


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns with at least three numeric values, including source fields."""

    return sorted(
        str(column)
        for column in frame.columns
        if pd.to_numeric(frame[column], errors="coerce").notna().sum() >= 3
    )


def _strings(frame: pd.DataFrame, column: str) -> tuple[str, ...]:
    if column not in frame:
        return ()
    return tuple(sorted(frame[column].dropna().astype(str).unique()))


def _fingerprint(frame: pd.DataFrame, selected: tuple[str, ...]) -> str:
    identity = tuple(
        column
        for column in ("shot_id", "session_id", "source_row", "monitor_vendor")
        if column in frame and column not in selected
    )
    columns = identity + selected
    records = [
        {
            column: None
            if pd.isna(value)
            else value.item()
            if hasattr(value, "item")
            else value
            for column, value in row.items()
        }
        for row in frame[list(columns)].to_dict(orient="records")
    ]
    serialized = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


def _adjust(values: list[float | None]) -> list[float | None]:
    finite = sorted(
        ((index, value) for index, value in enumerate(values) if value is not None),
        key=lambda item: item[1],
    )
    output: list[float | None] = [None] * len(values)
    previous = 1.0
    for rank in range(len(finite), 0, -1):
        index, value = finite[rank - 1]
        corrected = min(previous, value * len(finite) / rank)
        output[index] = min(1.0, corrected)
        previous = corrected
    return output


def _correlations(
    frame: pd.DataFrame, request: AnalysisRequest
) -> tuple[CorrelationEstimate, ...]:
    selected = (request.outcome, *request.predictors)
    working = (
        frame.dropna(subset=list(selected))
        if request.missing_policy == "listwise"
        else frame
    )
    provisional: list[CorrelationEstimate] = []
    for predictor in request.predictors:
        pair = (
            working[[request.outcome, predictor]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )
        count = len(pair)
        if count < request.min_samples:
            provisional.append(
                CorrelationEstimate(
                    predictor,
                    None,
                    None,
                    None,
                    None,
                    None,
                    count,
                    request.correlation_method,
                )
            )
            continue
        left = pair[request.outcome].to_numpy(float)
        right = pair[predictor].to_numpy(float)
        if request.correlation_method == "pearson":
            estimate = stats.pearsonr(left, right)
        elif request.correlation_method == "spearman":
            estimate = stats.spearmanr(left, right)
        else:
            estimate = stats.kendalltau(left, right)
        coefficient = float(estimate.statistic)
        lower: float | None = None
        upper: float | None = None
        if request.correlation_method == "pearson" and count > 3:
            transformed = np.arctanh(np.clip(coefficient, -0.999999, 0.999999))
            margin = stats.norm.ppf(0.5 + request.confidence_level / 2) / np.sqrt(
                count - 3
            )
            lower, upper = (
                float(np.tanh(transformed - margin)),
                float(np.tanh(transformed + margin)),
            )
        provisional.append(
            CorrelationEstimate(
                predictor,
                coefficient,
                float(estimate.pvalue),
                None,
                lower,
                upper,
                count,
                request.correlation_method,
            )
        )
    adjusted = _adjust([item.p_value for item in provisional])
    return tuple(
        replace(item, adjusted_p_value=adjusted[index])
        for index, item in enumerate(provisional)
    )


def _regression(frame: pd.DataFrame, request: AnalysisRequest) -> RegressionEstimate:
    columns = (request.outcome, *request.predictors)
    numeric = frame[list(columns)].apply(pd.to_numeric, errors="coerce").dropna()
    count = len(numeric)
    parameter_count = len(request.predictors) + 1
    if count < max(request.min_samples, parameter_count + 2):
        raise ValueError("Too few complete observations for regression")
    y = numeric[request.outcome].to_numpy(float)
    design = np.column_stack(
        (np.ones(count), numeric[list(request.predictors)].to_numpy(float))
    )
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < parameter_count:
        raise ValueError("Regression design matrix is rank deficient")
    fitted = design @ beta
    residuals = y - fitted
    residual_sum = float(residuals @ residuals)
    total_sum = float(((y - y.mean()) ** 2).sum())
    r_squared = 1 - residual_sum / total_sum
    degrees = count - parameter_count
    covariance = residual_sum / degrees * np.linalg.inv(design.T @ design)
    standard_errors = np.sqrt(np.diag(covariance))
    t_values = beta / standard_errors
    p_values = 2 * stats.t.sf(np.abs(t_values), degrees)
    critical = stats.t.ppf(0.5 + request.confidence_level / 2, degrees)
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
    variance = residual_sum / degrees
    cooks = residuals**2 / max(np.finfo(float).eps, parameter_count * variance)
    cooks *= leverage / np.maximum((1 - leverage) ** 2, np.finfo(float).eps)
    return RegressionEstimate(
        count,
        r_squared,
        1 - (1 - r_squared) * (count - 1) / degrees,
        coefficients,
        ResidualDiagnostics(
            float(np.sqrt(np.mean(residuals**2))),
            float(np.mean(np.abs(residuals))),
            float(np.mean(residuals)),
            float(np.std(residuals, ddof=parameter_count)),
            (
                float(np.diff(residuals) @ np.diff(residuals) / residual_sum)
                if residual_sum > 0
                else None
            ),
            int(np.sum(cooks > 4 / count)),
        ),
    )


def analyze_launch_monitor_data(
    frame: pd.DataFrame, request: AnalysisRequest
) -> AnalysisResult:
    """Analyze arbitrary numeric columns with explicit scientific boundaries."""

    selected = (request.outcome, *request.predictors)
    if not request.outcome or not request.predictors:
        raise ValueError("Select an outcome and predictors")
    if request.outcome in request.predictors:
        raise ValueError("outcome cannot also be a predictor")
    if len(set(request.predictors)) != len(request.predictors):
        raise ValueError("predictors must be unique")
    if not 0.5 < request.confidence_level < 1 or request.min_samples < 3:
        raise ValueError("Invalid confidence level or minimum sample count")
    required = set(selected) | ({request.group_by} if request.group_by else set())
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
    vendors = _strings(frame, "monitor_vendor")
    if any(column.startswith("source::") for column in selected) and len(vendors) > 1:
        raise ValueError("source fields cannot be pooled across multiple monitors")
    kinds = _strings(frame, "observation_kind") or ("shot",)
    aggregate = any(kind.lower() != "shot" for kind in kinds)
    if aggregate and request.analysis_mode != "correlation":
        raise ValueError("Aggregate observations cannot enter regression")
    if aggregate and not request.allow_aggregate:
        raise ValueError("Aggregate observations require allow_aggregate=True")
    warnings: list[str] = []
    if aggregate:
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
    correlations = (
        _correlations(frame, request) if request.analysis_mode != "regression" else ()
    )
    regression = (
        _regression(frame, request) if request.analysis_mode != "correlation" else None
    )
    groups: list[GroupAnalysis] = []
    if request.group_by:
        ungrouped = replace(request, group_by=None)
        for value, group in frame.dropna(subset=[request.group_by]).groupby(
            request.group_by, sort=True, observed=True
        ):
            try:
                result = analyze_launch_monitor_data(group, ungrouped)
                groups.append(
                    GroupAnalysis(
                        str(value),
                        len(group),
                        result.correlations,
                        result.regression,
                        result.warnings,
                    )
                )
            except ValueError as error:
                groups.append(
                    GroupAnalysis(str(value), len(group), (), None, (str(error),))
                )
    return AnalysisResult(
        CONTRACT_VERSION,
        request,
        DatasetSummary(
            len(frame),
            int(numeric.dropna().shape[0]),
            selected,
            vendors,
            _strings(frame, "session_id"),
            kinds,
            _fingerprint(frame, selected),
        ),
        correlations,
        regression,
        tuple(groups),
        tuple(warnings),
    )


__all__ = [
    "CONTRACT_VERSION",
    "AnalysisRequest",
    "AnalysisResult",
    "analyze_launch_monitor_data",
    "numeric_columns",
]
