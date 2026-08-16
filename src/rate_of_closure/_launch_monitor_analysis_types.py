"""Immutable data contracts for launch-monitor statistical analysis."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Literal, cast

import numpy as np

AnalysisMode = Literal["correlation", "regression", "comprehensive"]
CorrelationMethod = Literal["pearson", "spearman", "kendall"]
MissingPolicy = Literal["pairwise", "listwise", "fail"]
CONTRACT_VERSION = "1.0.0"


@dataclass(frozen=True)
class AnalysisRequest:
    """Configuration for one launch-monitor analysis run."""

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
    """Provenance and completeness metadata for analyzed observations."""

    row_count: int
    complete_row_count: int
    selected_columns: tuple[str, ...]
    monitor_vendors: tuple[str, ...]
    session_ids: tuple[str, ...]
    observation_kinds: tuple[str, ...]
    fingerprint_sha256: str


@dataclass(frozen=True)
class CorrelationEstimate:
    """One predictor's correlation estimate and uncertainty metadata."""

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
    """One ordinary least-squares coefficient estimate."""

    estimate: float
    standard_error: float
    t_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True)
class ResidualDiagnostics:
    """Residual quality and influence diagnostics for a regression."""

    rmse: float
    mae: float
    residual_mean: float
    residual_std: float
    durbin_watson: float | None
    influential_count: int


@dataclass(frozen=True)
class RegressionEstimate:
    """Ordinary least-squares result and diagnostics."""

    sample_count: int
    r_squared: float
    adjusted_r_squared: float
    coefficients: dict[str, CoefficientEstimate]
    residual_diagnostics: ResidualDiagnostics


@dataclass(frozen=True)
class GroupAnalysis:
    """Analysis result for one group value."""

    group_value: str
    row_count: int
    correlations: tuple[CorrelationEstimate, ...]
    regression: RegressionEstimate | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisResult:
    """Complete, serializable launch-monitor analysis result."""

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
