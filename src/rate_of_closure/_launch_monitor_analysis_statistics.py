"""Correlation and regression calculations for launch-monitor analysis."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats

from ._launch_monitor_analysis_types import (
    AnalysisRequest,
    CoefficientEstimate,
    CorrelationEstimate,
    RegressionEstimate,
    ResidualDiagnostics,
)


@dataclass(frozen=True)
class _RegressionWork:
    design: np.ndarray
    residuals: np.ndarray
    residual_sum: float
    inverse_information: np.ndarray
    parameter_count: int


def correlations(
    frame: pd.DataFrame, request: AnalysisRequest
) -> tuple[CorrelationEstimate, ...]:
    """Calculate requested pairwise correlations and adjusted p-values."""

    selected = (request.outcome, *request.predictors)
    working = (
        frame.dropna(subset=list(selected))
        if request.missing_policy == "listwise"
        else frame
    )
    provisional = tuple(
        _correlation_for_predictor(working, request, predictor)
        for predictor in request.predictors
    )
    adjusted = _adjust_p_values([item.p_value for item in provisional])
    return tuple(
        replace(item, adjusted_p_value=adjusted[index])
        for index, item in enumerate(provisional)
    )


def _correlation_for_predictor(
    frame: pd.DataFrame, request: AnalysisRequest, predictor: str
) -> CorrelationEstimate:
    pair = (
        frame[[request.outcome, predictor]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    count = len(pair)
    if count < request.min_samples:
        return CorrelationEstimate(
            predictor, None, None, None, None, None, count, request.correlation_method
        )
    left = pair[request.outcome].to_numpy(float)
    right = pair[predictor].to_numpy(float)
    estimate = _correlation_estimate(left, right, request)
    lower, upper = _pearson_interval(float(estimate.statistic), count, request)
    return CorrelationEstimate(
        predictor,
        float(estimate.statistic),
        float(estimate.pvalue),
        None,
        lower,
        upper,
        count,
        request.correlation_method,
    )


def _correlation_estimate(
    left: np.ndarray, right: np.ndarray, request: AnalysisRequest
) -> stats.SignificanceResult:
    if request.correlation_method == "pearson":
        return stats.pearsonr(left, right)
    if request.correlation_method == "spearman":
        return stats.spearmanr(left, right)
    return stats.kendalltau(left, right)


def _pearson_interval(
    coefficient: float, count: int, request: AnalysisRequest
) -> tuple[float | None, float | None]:
    if request.correlation_method != "pearson" or count <= 3:
        return None, None
    transformed = np.arctanh(np.clip(coefficient, -0.999999, 0.999999))
    margin = stats.norm.ppf(0.5 + request.confidence_level / 2) / np.sqrt(count - 3)
    return float(np.tanh(transformed - margin)), float(np.tanh(transformed + margin))


def _adjust_p_values(values: list[float | None]) -> list[float | None]:
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


def regression(frame: pd.DataFrame, request: AnalysisRequest) -> RegressionEstimate:
    """Calculate ordinary least squares with uncertainty and diagnostics."""

    columns = (request.outcome, *request.predictors)
    numeric = frame[list(columns)].apply(pd.to_numeric, errors="coerce").dropna()
    count = len(numeric)
    parameter_count = len(request.predictors) + 1
    if count < max(request.min_samples, parameter_count + 2):
        raise ValueError("Too few complete observations for regression")
    outcome = numeric[request.outcome].to_numpy(float)
    design = np.column_stack(
        (np.ones(count), numeric[list(request.predictors)].to_numpy(float))
    )
    beta, _, rank, _ = np.linalg.lstsq(design, outcome, rcond=None)
    if rank < parameter_count:
        raise ValueError("Regression design matrix is rank deficient")
    fitted = design @ beta
    residuals = outcome - fitted
    residual_sum = float(residuals @ residuals)
    total_sum = float(((outcome - outcome.mean()) ** 2).sum())
    r_squared = 1 - residual_sum / total_sum
    degrees = count - parameter_count
    inverse_information = np.linalg.inv(design.T @ design)
    standard_errors = np.sqrt(np.diag(residual_sum / degrees * inverse_information))
    coefficients = _coefficient_estimates(beta, standard_errors, degrees, request)
    diagnostics = _residual_diagnostics(
        _RegressionWork(
            design, residuals, residual_sum, inverse_information, parameter_count
        )
    )
    return RegressionEstimate(
        count,
        r_squared,
        1 - (1 - r_squared) * (count - 1) / degrees,
        coefficients,
        diagnostics,
    )


def _coefficient_estimates(
    beta: np.ndarray,
    standard_errors: np.ndarray,
    degrees: int,
    request: AnalysisRequest,
) -> dict[str, CoefficientEstimate]:
    t_values = beta / standard_errors
    p_values = 2 * stats.t.sf(np.abs(t_values), degrees)
    critical = stats.t.ppf(0.5 + request.confidence_level / 2, degrees)
    names = ("intercept", *request.predictors)
    return {
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


def _residual_diagnostics(work: _RegressionWork) -> ResidualDiagnostics:
    count = len(work.residuals)
    degrees = count - work.parameter_count
    leverage = np.einsum(
        "ij,jk,ik->i", work.design, work.inverse_information, work.design
    )
    variance = work.residual_sum / degrees
    cooks = work.residuals**2 / max(
        np.finfo(float).eps, work.parameter_count * variance
    )
    cooks *= leverage / np.maximum((1 - leverage) ** 2, np.finfo(float).eps)
    durbin_watson = (
        float(np.diff(work.residuals) @ np.diff(work.residuals) / work.residual_sum)
        if work.residual_sum > 0
        else None
    )
    return ResidualDiagnostics(
        float(np.sqrt(np.mean(work.residuals**2))),
        float(np.mean(np.abs(work.residuals))),
        float(np.mean(work.residuals)),
        float(np.std(work.residuals, ddof=work.parameter_count)),
        durbin_watson,
        int(np.sum(cooks > 4 / count)),
    )
