"""UI-neutral launch-monitor statistics matching the React contract."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from typing import cast

import pandas as pd

from rate_of_closure.launch_monitor_numeric import finite_launch_monitor_scalar

from ._launch_monitor_analysis_statistics import correlations, regression
from ._launch_monitor_analysis_types import (
    CONTRACT_VERSION,
    AnalysisRequest,
    AnalysisResult,
    DatasetSummary,
    GroupAnalysis,
)
from ._launch_monitor_analysis_types import (
    AnalysisMode as AnalysisMode,
)
from ._launch_monitor_analysis_types import (
    CoefficientEstimate as CoefficientEstimate,
)
from ._launch_monitor_analysis_types import (
    CorrelationEstimate as CorrelationEstimate,
)
from ._launch_monitor_analysis_types import (
    CorrelationMethod as CorrelationMethod,
)
from ._launch_monitor_analysis_types import (
    MissingPolicy as MissingPolicy,
)
from ._launch_monitor_analysis_types import (
    RegressionEstimate as RegressionEstimate,
)
from ._launch_monitor_analysis_types import (
    ResidualDiagnostics as ResidualDiagnostics,
)


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns with at least three numeric values, including source fields."""

    return sorted(
        str(column)
        for column in frame.columns
        if frame[column].map(finite_launch_monitor_scalar).notna().sum() >= 3
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
            column: (
                None
                if pd.isna(value)
                else value.item()
                if hasattr(value, "item")
                else value
            )
            for column, value in row.items()
        }
        for row in frame[list(columns)].to_dict(orient="records")
    ]
    serialized = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


def _validate_request(frame: pd.DataFrame, request: AnalysisRequest) -> pd.DataFrame:
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
    projected = frame[list(selected)]
    # pandas spelled the elementwise map `applymap` before 2.1 and removed it in
    # 3.0. Resolve the legacy name lazily: on 3.x the attribute does not exist,
    # so binding it eagerly raised AttributeError even though `.map` was there.
    mapper = getattr(projected, "map", None)
    if not callable(mapper):
        mapper = getattr(projected, "applymap", None)
    if not callable(mapper):
        raise AttributeError("pandas DataFrame exposes neither .map nor .applymap")
    numeric = cast(pd.DataFrame, mapper(finite_launch_monitor_scalar))
    constants = [
        column for column in selected if numeric[column].dropna().nunique() < 2
    ]
    if constants:
        raise ValueError(f"Constant variables cannot be analyzed: {constants}")
    if request.missing_policy == "fail" and numeric.isna().any().any():
        raise ValueError("Selected variables contain missing or non-numeric values")
    return numeric


def _validate_observation_scope(
    frame: pd.DataFrame, request: AnalysisRequest, selected: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...], list[str]]:
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
    return vendors, kinds, warnings


def _group_analyses(
    frame: pd.DataFrame, request: AnalysisRequest
) -> tuple[GroupAnalysis, ...]:
    if not request.group_by:
        return ()
    groups: list[GroupAnalysis] = []
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
    return tuple(groups)


def analyze_launch_monitor_data(
    frame: pd.DataFrame, request: AnalysisRequest
) -> AnalysisResult:
    """Analyze arbitrary numeric columns with explicit scientific boundaries."""

    selected = (request.outcome, *request.predictors)
    numeric = _validate_request(frame, request)
    vendors, kinds, warnings = _validate_observation_scope(frame, request, selected)
    projected = frame.copy()
    projected[list(selected)] = numeric
    correlation_results = (
        correlations(projected, request)
        if request.analysis_mode != "regression"
        else ()
    )
    regression_result = (
        regression(projected, request)
        if request.analysis_mode != "correlation"
        else None
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
        correlation_results,
        regression_result,
        _group_analyses(frame, request),
        tuple(warnings),
    )


__all__ = [
    "CONTRACT_VERSION",
    "AnalysisRequest",
    "AnalysisResult",
    "analyze_launch_monitor_data",
    "numeric_columns",
]
