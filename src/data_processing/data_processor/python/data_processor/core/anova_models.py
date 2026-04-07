"""Shared ANOVA result models and reporting helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class PostHocMethod(Enum):
    """Available post-hoc test methods."""

    TUKEY_HSD = "tukey_hsd"
    BONFERRONI = "bonferroni"
    SCHEFFE = "scheffe"
    DUNNETT = "dunnett"
    HOLM = "holm"
    SIDAK = "sidak"


@dataclass
class AssumptionTestResult:
    """Result of assumption testing."""

    test_name: str
    statistic: float
    p_value: float
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PostHocComparison:
    """Result of a single post-hoc comparison."""

    group1: str
    group2: str
    mean_diff: float
    std_error: float
    t_statistic: float
    p_value: float
    p_adjusted: float
    ci_lower: float
    ci_upper: float
    significant: bool


@dataclass
class ANOVATable:
    """Standard ANOVA summary table."""

    source: list[str]
    sum_of_squares: list[float]
    df: list[int]
    mean_square: list[float]
    f_statistic: list[float | None]
    p_value: list[float | None]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a pandas DataFrame for display."""
        return pd.DataFrame(
            {
                "Source": self.source,
                "SS": self.sum_of_squares,
                "df": self.df,
                "MS": self.mean_square,
                "F": self.f_statistic,
                "p-value": self.p_value,
            }
        )


@dataclass
class OneWayANOVAResult:
    """Results from one-way ANOVA."""

    f_statistic: float
    p_value: float
    df_between: int
    df_within: int
    anova_table: ANOVATable
    eta_squared: float
    omega_squared: float
    cohens_f: float
    group_means: dict[str, float]
    group_stds: dict[str, float]
    group_counts: dict[str, int]
    grand_mean: float
    post_hoc_results: list[PostHocComparison] = field(default_factory=list)
    assumption_tests: list[AssumptionTestResult] = field(default_factory=list)
    observed_power: float = 0.0


@dataclass
class TwoWayANOVAResult:
    """Results from two-way ANOVA."""

    factor_a_f: float
    factor_a_p: float
    factor_a_df: int
    factor_b_f: float
    factor_b_p: float
    factor_b_df: int
    interaction_f: float
    interaction_p: float
    interaction_df: int
    df_error: int
    ms_error: float
    anova_table: ANOVATable
    eta_squared_a: float
    eta_squared_b: float
    eta_squared_ab: float
    partial_eta_squared_a: float
    partial_eta_squared_b: float
    partial_eta_squared_ab: float
    cell_means: pd.DataFrame
    marginal_means_a: dict[str, float] = field(default_factory=dict)
    marginal_means_b: dict[str, float] = field(default_factory=dict)
    assumption_tests: list[AssumptionTestResult] = field(default_factory=list)


@dataclass
class RepeatedMeasuresResult:
    """Results from repeated measures ANOVA."""

    f_statistic: float
    p_value: float
    df_effect: int
    df_error: int
    mauchly_w: float
    mauchly_p: float
    sphericity_assumed: bool
    greenhouse_geisser_epsilon: float
    huynh_feldt_epsilon: float
    corrected_p_gg: float
    corrected_p_hf: float
    eta_squared: float
    partial_eta_squared: float
    anova_table: ANOVATable


def format_anova_report(result: OneWayANOVAResult | TwoWayANOVAResult) -> str:
    """Format ANOVA results as a text report."""
    lines = ["=" * 60, "ANOVA Results", "=" * 60, ""]

    lines.append("ANOVA Table:")
    lines.append("-" * 60)
    lines.append(result.anova_table.to_dataframe().to_string(index=False))
    lines.append("")

    if isinstance(result, OneWayANOVAResult):
        lines.append("Effect Sizes:")
        lines.append(f"  η² (eta-squared):     {result.eta_squared:.4f}")
        lines.append(f"  ω² (omega-squared):   {result.omega_squared:.4f}")
        lines.append(f"  Cohen's f:            {result.cohens_f:.4f}")
        lines.append(f"  Observed Power:       {result.observed_power:.4f}")
        lines.append("")
        lines.append("Group Statistics:")
        lines.extend(
            [
                f"  {name}: M = {result.group_means[name]:.4f}, SD = {result.group_stds[name]:.4f}, n = {result.group_counts[name]}"
                for name in result.group_means
            ]
        )
        lines.append("")

        if result.post_hoc_results:
            lines.append("Post-hoc Comparisons:")
            lines.append("-" * 60)
            for comparison in result.post_hoc_results:
                significance = "*" if comparison.significant else ""
                lines.append(
                    f"  {comparison.group1} vs {comparison.group2}: "
                    f"Δ = {comparison.mean_diff:.4f}, p = {comparison.p_adjusted:.4f} {significance}"
                )

    elif isinstance(result, TwoWayANOVAResult):
        lines.append("Effect Sizes (Partial η²):")
        lines.append(f"  Factor A:     {result.partial_eta_squared_a:.4f}")
        lines.append(f"  Factor B:     {result.partial_eta_squared_b:.4f}")
        lines.append(f"  Interaction:  {result.partial_eta_squared_ab:.4f}")

    if hasattr(result, "assumption_tests") and result.assumption_tests:
        lines.append("")
        lines.append("Assumption Tests:")
        lines.append("-" * 60)
        for test in result.assumption_tests:
            status = "PASS" if test.passed else "FAIL"
            lines.append(f"  [{status}] {test.test_name}: {test.message}")

    return "\n".join(lines)
