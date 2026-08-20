"""Immutable contracts for player covariation analysis."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

MIN_FISHER_SAMPLES = 4


@dataclass(frozen=True)
class CovariationRequest:
    """Select columns and sampling rules for one pairwise analysis."""

    x_column: str
    y_column: str
    player_column: str
    min_samples: int = MIN_FISHER_SAMPLES
    confidence_level: float = 0.95


@dataclass(frozen=True)
class PairScanRequest:
    """Select columns and sampling rules for an exploratory all-pairs scan."""

    player_column: str
    numeric_columns: tuple[str, ...] = ()
    min_samples: int = MIN_FISHER_SAMPLES
    confidence_level: float = 0.95


@dataclass(frozen=True)
class AssociationEstimate:
    """A descriptive bivariate estimate for one analysis scope."""

    sample_count: int
    group_count: int
    pearson_r: float | None
    spearman_r: float | None
    slope: float | None
    intercept: float | None
    r_squared: float | None
    ci_lower: float | None
    ci_upper: float | None
    status: str


@dataclass(frozen=True)
class MetaAnalysisSummary:
    """Fixed- and random-effects synthesis of within-player Pearson r."""

    contributor_count: int
    total_sample_count: int
    fixed_effect_r: float | None
    fixed_ci_lower: float | None
    fixed_ci_upper: float | None
    random_effect_r: float | None
    random_ci_lower: float | None
    random_ci_upper: float | None
    tau_squared: float | None
    q_statistic: float | None
    i_squared_pct: float | None


@dataclass(frozen=True)
class PlayerCovariationAnalysis:
    """Export-friendly tables and summaries for one selected variable pair."""

    request: CovariationRequest
    per_player: pd.DataFrame
    pooled: AssociationEstimate
    within_player: AssociationEstimate
    between_player: AssociationEstimate
    meta_analysis: MetaAnalysisSummary
    backing_data: pd.DataFrame
    units: dict[str, str]
    definitions: dict[str, str]
    warnings: tuple[str, ...]
    method_description: str


@dataclass(frozen=True)
class PairScanAnalysis:
    """Deterministically ranked, explicitly exploratory pair scan."""

    ranking: pd.DataFrame
    warnings: tuple[str, ...]
    method_description: str


__all__ = [
    "AssociationEstimate",
    "CovariationRequest",
    "MetaAnalysisSummary",
    "PairScanAnalysis",
    "PairScanRequest",
    "PlayerCovariationAnalysis",
]
