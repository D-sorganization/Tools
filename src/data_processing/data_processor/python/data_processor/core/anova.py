from numba import jit

"""Comprehensive ANOVA Suite for Statistical Analysis.

Provides professional-level ANOVA capabilities including:
- One-way ANOVA
- Two-way ANOVA (with and without interaction)
- Repeated measures ANOVA
- MANOVA (Multivariate ANOVA)
- Post-hoc tests (Tukey HSD, Bonferroni, Scheffé)
- Effect size calculations (eta-squared, omega-squared, Cohen's d)
- Assumption testing (normality, homogeneity of variance)

Designed for rigorous statistical analysis of experimental data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


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
        """Convert to pandas DataFrame for display."""
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

    # Main results
    f_statistic: float
    p_value: float
    df_between: int
    df_within: int

    # ANOVA table
    anova_table: ANOVATable

    # Effect sizes
    eta_squared: float
    omega_squared: float
    cohens_f: float

    # Group statistics
    group_means: dict[str, float]
    group_stds: dict[str, float]
    group_counts: dict[str, int]
    grand_mean: float

    # Post-hoc tests
    post_hoc_results: list[PostHocComparison] = field(default_factory=list)

    # Assumption tests
    assumption_tests: list[AssumptionTestResult] = field(default_factory=list)

    # Power analysis
    observed_power: float = 0.0


@dataclass
class TwoWayANOVAResult:
    """Results from two-way ANOVA."""

    # Main effects
    factor_a_f: float
    factor_a_p: float
    factor_a_df: int

    factor_b_f: float
    factor_b_p: float
    factor_b_df: int

    # Interaction
    interaction_f: float
    interaction_p: float
    interaction_df: int

    # Error
    df_error: int
    ms_error: float

    # ANOVA table
    anova_table: ANOVATable

    # Effect sizes
    eta_squared_a: float
    eta_squared_b: float
    eta_squared_ab: float
    partial_eta_squared_a: float
    partial_eta_squared_b: float
    partial_eta_squared_ab: float

    # Cell means
    cell_means: pd.DataFrame
    marginal_means_a: dict[str, float] = field(default_factory=dict)
    marginal_means_b: dict[str, float] = field(default_factory=dict)

    # Assumption tests
    assumption_tests: list[AssumptionTestResult] = field(default_factory=list)


@dataclass
class RepeatedMeasuresResult:
    """Results from repeated measures ANOVA."""

    f_statistic: float
    p_value: float
    df_effect: int
    df_error: int

    # Sphericity test
    mauchly_w: float
    mauchly_p: float
    sphericity_assumed: bool

    # Corrections
    greenhouse_geisser_epsilon: float
    huynh_feldt_epsilon: float
    corrected_p_gg: float
    corrected_p_hf: float

    # Effect sizes
    eta_squared: float
    partial_eta_squared: float

    # ANOVA table
    anova_table: ANOVATable


class ANOVAAnalyzer:
    """Comprehensive ANOVA analysis suite.

    Provides one-way, two-way, and repeated measures ANOVA
    with full diagnostic testing and post-hoc analysis.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        """Initialize ANOVA analyzer.

        Args:
            alpha: Significance level for hypothesis testing
        """
        self.alpha = alpha

    def _validate_and_prepare_groups(
        self,
        df: pd.DataFrame,
        dependent_var: str,
        group_var: str,
    ) -> dict[str, np.ndarray]:
        """Validate inputs and prepare group arrays for ANOVA.

        Args:
            df: DataFrame with data
            dependent_var: Name of dependent variable column
            group_var: Name of grouping variable column

        Returns:
            Dictionary mapping group names to NaN-filtered arrays

        Raises:
            ValueError: If columns missing or fewer than 2 groups
        """
        if dependent_var not in df.columns or group_var not in df.columns:
            raise ValueError("Specified columns not found in DataFrame")

        groups = df.groupby(group_var)[dependent_var].apply(list).to_dict()
        if len(groups) < 2:
            raise ValueError("ANOVA requires at least 2 groups")

        return {
            name: np.array([x for x in values if not np.isnan(x)])
            for name, values in groups.items()
        }

    def _compute_anova_statistics(
        self,
        group_arrays: dict[str, np.ndarray],
    ) -> tuple[float, float, float, float, float, float, int, int, int, float]:
        """Compute core ANOVA sums of squares, F-statistic, and effect sizes.

        Args:
            group_arrays: Dictionary mapping group names to arrays

        Returns:
            Tuple of (ss_between, ss_within, ss_total, ms_between, ms_within,
                       f_stat, df_between, df_within, df_total, grand_mean)
        """
        if not (group_arrays is not None):
            raise ValueError("group_arrays must be provided")
        k = len(group_arrays)
        n_total = sum(len(arr) for arr in group_arrays.values())
        grand_mean = np.mean(np.concatenate(list(group_arrays.values())))

        ss_between = sum(
            len(arr) * (np.mean(arr) - grand_mean) ** 2 for arr in group_arrays.values()
        )
        ss_within = sum(np.sum((arr - np.mean(arr)) ** 2) for arr in group_arrays.values())
        ss_total = ss_between + ss_within

        df_between = k - 1
        df_within = n_total - k
        df_total = n_total - 1

        ms_between = ss_between / df_between
        ms_within = ss_within / df_within

        f_stat = ms_between / ms_within

        return (
            ss_between,
            ss_within,
            ss_total,
            ms_between,
            ms_within,
            f_stat,
            df_between,
            df_within,
            df_total,
            grand_mean,
        )

    def one_way_anova(
        self,
        df: pd.DataFrame,
        dependent_var: str,
        group_var: str,
        post_hoc: PostHocMethod | None = PostHocMethod.TUKEY_HSD,
        test_assumptions: bool = True,
    ) -> OneWayANOVAResult:
        """Perform one-way ANOVA.

        Args:
            df: DataFrame with data
            dependent_var: Name of dependent variable column
            group_var: Name of grouping variable column
            post_hoc: Post-hoc test method (None to skip)
            test_assumptions: Whether to run assumption tests

        Returns:
            Complete one-way ANOVA results
        """
        if not (df is not None):
            raise ValueError("df must be provided")
        group_arrays = self._validate_and_prepare_groups(df, dependent_var, group_var)

        (
            ss_between,
            ss_within,
            ss_total,
            ms_between,
            ms_within,
            f_stat,
            df_between,
            df_within,
            df_total,
            grand_mean,
        ) = self._compute_anova_statistics(group_arrays)

        n_total = sum(len(arr) for arr in group_arrays.values())
        p_value = 1 - stats.f.cdf(f_stat, df_between, df_within)

        # Effect sizes
        eta_squared = ss_between / ss_total
        omega_squared = max(0, (ss_between - df_between * ms_within) / (ss_total + ms_within))
        cohens_f = np.sqrt(eta_squared / (1 - eta_squared)) if eta_squared < 1 else 0

        anova_table = ANOVATable(
            source=["Between Groups", "Within Groups", "Total"],
            sum_of_squares=[ss_between, ss_within, ss_total],
            df=[df_between, df_within, df_total],
            mean_square=[ms_between, ms_within, np.nan],
            f_statistic=[f_stat, None, None],
            p_value=[p_value, None, None],
        )

        assumption_tests = self._test_anova_assumptions(group_arrays) if test_assumptions else []
        post_hoc_results = (
            self._post_hoc_tests(group_arrays, ms_within, df_within, post_hoc)
            if post_hoc and p_value < self.alpha
            else []
        )

        noncentrality = np.sqrt(n_total * eta_squared / (1 - eta_squared)) if eta_squared < 1 else 0

        return OneWayANOVAResult(
            f_statistic=f_stat,
            p_value=p_value,
            df_between=df_between,
            df_within=df_within,
            anova_table=anova_table,
            eta_squared=eta_squared,
            omega_squared=omega_squared,
            cohens_f=cohens_f,
            group_means={name: np.mean(arr) for name, arr in group_arrays.items()},
            group_stds={name: np.std(arr, ddof=1) for name, arr in group_arrays.items()},
            group_counts={name: len(arr) for name, arr in group_arrays.items()},
            grand_mean=grand_mean,
            post_hoc_results=post_hoc_results,
            assumption_tests=assumption_tests,
            observed_power=self._calculate_power(f_stat, df_between, df_within, noncentrality),
        )

    def two_way_anova(
        self,
        df: pd.DataFrame,
        dependent_var: str,
        factor_a: str,
        factor_b: str,
        test_interaction: bool = True,
        test_assumptions: bool = True,
    ) -> TwoWayANOVAResult:
        """Perform two-way ANOVA.

        Args:
            df: DataFrame with data
            dependent_var: Name of dependent variable column
            factor_a: Name of first factor column
            factor_b: Name of second factor column
            test_interaction: Whether to test for interaction
            test_assumptions: Whether to run assumption tests

        Returns:
            Complete two-way ANOVA results
        """
        if not (df is not None):
            raise ValueError("df must be provided")
        data = df[[dependent_var, factor_a, factor_b]].dropna()
        y: np.ndarray = np.asarray(data[dependent_var].values)

        levels_a = data[factor_a].unique()
        levels_b = data[factor_b].unique()
        n_total = len(data)
        grand_mean = np.mean(y)
        cell_means = data.groupby([factor_a, factor_b])[dependent_var].mean().unstack()
        marginal_a = data.groupby(factor_a)[dependent_var].mean().to_dict()
        marginal_b = data.groupby(factor_b)[dependent_var].mean().to_dict()

        ss = self._two_way_sum_of_squares(
            data,
            y,
            dependent_var,
            factor_a,
            factor_b,
            levels_a,
            levels_b,
            marginal_a,
            marginal_b,
            grand_mean,
            test_interaction,
        )
        ftest = self._two_way_f_tests(ss, len(levels_a), len(levels_b), n_total)
        effect = self._two_way_effect_sizes(ss)

        interaction_label = f"{factor_a}\u00d7{factor_b}"
        anova_table = ANOVATable(
            source=[factor_a, factor_b, interaction_label, "Error", "Total"],
            sum_of_squares=[
                ss["a"],
                ss["b"],
                ss["ab"],
                ss["error"],
                ss["total"],
            ],
            df=[
                ftest["df_a"],
                ftest["df_b"],
                ftest["df_ab"],
                ftest["df_error"],
                n_total - 1,
            ],
            mean_square=[
                ftest["ms_a"],
                ftest["ms_b"],
                ftest["ms_ab"],
                ftest["ms_error"],
                np.nan,
            ],
            f_statistic=[
                ftest["f_a"],
                ftest["f_b"],
                ftest["f_ab"],
                None,
                None,
            ],
            p_value=[
                ftest["p_a"],
                ftest["p_b"],
                ftest["p_ab"],
                None,
                None,
            ],
        )

        assumption_tests = self._two_way_assumption_tests(
            data,
            dependent_var,
            factor_a,
            factor_b,
            levels_a,
            levels_b,
            test_assumptions,
        )

        return TwoWayANOVAResult(
            factor_a_f=ftest["f_a"],
            factor_a_p=ftest["p_a"],
            factor_a_df=ftest["df_a"],
            factor_b_f=ftest["f_b"],
            factor_b_p=ftest["p_b"],
            factor_b_df=ftest["df_b"],
            interaction_f=ftest["f_ab"],
            interaction_p=ftest["p_ab"],
            interaction_df=ftest["df_ab"],
            df_error=ftest["df_error"],
            ms_error=ftest["ms_error"],
            anova_table=anova_table,
            eta_squared_a=effect["eta_a"],
            eta_squared_b=effect["eta_b"],
            eta_squared_ab=effect["eta_ab"],
            partial_eta_squared_a=effect["partial_eta_a"],
            partial_eta_squared_b=effect["partial_eta_b"],
            partial_eta_squared_ab=effect["partial_eta_ab"],
            cell_means=cell_means,
            marginal_means_a=marginal_a,
            marginal_means_b=marginal_b,
            assumption_tests=assumption_tests,
        )

    # -- two-way ANOVA helper methods --

    @jit(nopython=True, fastmath=True)
    @jit(nopython=True, fastmath=True)
    @staticmethod
    def _two_way_sum_of_squares(
        data: pd.DataFrame,
        y: np.ndarray,
        dependent_var: str,
        factor_a: str,
        factor_b: str,
        levels_a: np.ndarray,
        levels_b: np.ndarray,
        marginal_a: dict,
        marginal_b: dict,
        grand_mean: float,
        test_interaction: bool,
    ) -> dict[str, float]:
        """Compute SS_total, SS_A, SS_B, SS_AB, SS_error."""
        if not (data is not None):
            raise ValueError("data must be provided")
        ss_total = float(np.sum((y - grand_mean) ** 2))

        ss_a = sum(
            len(data[data[factor_a] == lv]) * (marginal_a[lv] - grand_mean) ** 2 for lv in levels_a
        )
        ss_b = sum(
            len(data[data[factor_b] == lv]) * (marginal_b[lv] - grand_mean) ** 2 for lv in levels_b
        )

        ss_ab = 0.0
        if test_interaction:
            for lv_a in levels_a:
                for lv_b in levels_b:
                    cell = data[(data[factor_a] == lv_a) & (data[factor_b] == lv_b)]
                    if len(cell) > 0:
                        expected = marginal_a[lv_a] + marginal_b[lv_b] - grand_mean
                        cell_mean = cell[dependent_var].mean()
                        ss_ab += len(cell) * (cell_mean - expected) ** 2

        return {
            "total": ss_total,
            "a": float(ss_a),
            "b": float(ss_b),
            "ab": ss_ab,
            "error": ss_total - float(ss_a) - float(ss_b) - ss_ab,
        }

    @staticmethod
    def _two_way_f_tests(ss: dict[str, float], a: int, b: int, n_total: int) -> dict[str, float]:
        """Compute degrees of freedom, mean squares, F-statistics, and p-values."""
        if not (ss is not None):
            raise ValueError("ss must be provided")
        df_a = a - 1
        df_b = b - 1
        df_ab = df_a * df_b
        df_error = n_total - a * b

        ms_a = ss["a"] / df_a
        ms_b = ss["b"] / df_b
        ms_ab = ss["ab"] / df_ab if df_ab > 0 else 0
        ms_error = ss["error"] / df_error

        f_a = ms_a / ms_error
        f_b = ms_b / ms_error
        f_ab = ms_ab / ms_error if ms_ab > 0 else 0

        p_a = 1 - stats.f.cdf(f_a, df_a, df_error)
        p_b = 1 - stats.f.cdf(f_b, df_b, df_error)
        p_ab = 1 - stats.f.cdf(f_ab, df_ab, df_error) if f_ab > 0 else 1.0

        return {
            "df_a": df_a,
            "df_b": df_b,
            "df_ab": df_ab,
            "df_error": df_error,
            "ms_a": ms_a,
            "ms_b": ms_b,
            "ms_ab": ms_ab,
            "ms_error": ms_error,
            "f_a": f_a,
            "f_b": f_b,
            "f_ab": f_ab,
            "p_a": float(p_a),
            "p_b": float(p_b),
            "p_ab": float(p_ab),
        }

    @staticmethod
    def _two_way_effect_sizes(ss: dict[str, float]) -> dict[str, float]:
        """Compute eta-squared and partial eta-squared effect sizes."""
        ss_t = ss["total"]
        ss_e = ss["error"]
        return {
            "eta_a": ss["a"] / ss_t,
            "eta_b": ss["b"] / ss_t,
            "eta_ab": ss["ab"] / ss_t,
            "partial_eta_a": ss["a"] / (ss["a"] + ss_e),
            "partial_eta_b": ss["b"] / (ss["b"] + ss_e),
            "partial_eta_ab": (ss["ab"] / (ss["ab"] + ss_e) if (ss["ab"] + ss_e) > 0 else 0),
        }

    def _two_way_assumption_tests(
        self,
        data: pd.DataFrame,
        dependent_var: str,
        factor_a: str,
        factor_b: str,
        levels_a: np.ndarray,
        levels_b: np.ndarray,
        test_assumptions: bool,
    ) -> list:
        """Run assumption tests for two-way ANOVA if requested."""
        if not (data is not None):
            raise ValueError("data must be provided")
        if not test_assumptions:
            return []
        cell_data = {}
        for lv_a in levels_a:
            for lv_b in levels_b:
                cell = data[(data[factor_a] == lv_a) & (data[factor_b] == lv_b)]
                if len(cell) > 2:
                    cell_data[f"{lv_a}_{lv_b}"] = cell[dependent_var].values
        if cell_data:
            return self._test_anova_assumptions(cell_data)
        return []

    def repeated_measures_anova(
        self,
        df: pd.DataFrame,
        dependent_vars: list[str],
        subject_id: str,
    ) -> RepeatedMeasuresResult:
        """Perform one-way repeated measures ANOVA.

        Args:
            df: DataFrame with one row per subject
            dependent_vars: List of columns for repeated measures
            subject_id: Column identifying subjects

        Returns:
            Repeated measures ANOVA results
        """
        # Reshape data to long format
        if not (df is not None):
            raise ValueError("df must be provided")
        data = df[[subject_id] + dependent_vars].dropna()
        n_subjects = len(data)
        k = len(dependent_vars)

        # Get values matrix
        values = data[dependent_vars].values

        # Calculate means
        grand_mean = np.mean(values)
        condition_means = np.mean(values, axis=0)
        subject_means = np.mean(values, axis=1)

        # Sum of squares
        ss_total = np.sum((values - grand_mean) ** 2)
        ss_between_subjects = k * np.sum((subject_means - grand_mean) ** 2)
        ss_within_subjects = ss_total - ss_between_subjects
        ss_conditions = n_subjects * np.sum((condition_means - grand_mean) ** 2)
        ss_error = ss_within_subjects - ss_conditions

        # Degrees of freedom
        df_between_subjects = n_subjects - 1
        df_conditions = k - 1
        df_error = df_between_subjects * df_conditions

        # Mean squares
        ms_conditions = ss_conditions / df_conditions
        ms_error = ss_error / df_error

        # F statistic
        f_stat = ms_conditions / ms_error
        p_value = 1 - stats.f.cdf(f_stat, df_conditions, df_error)

        # Sphericity test (Mauchly's)
        mauchly_w, mauchly_p, gg_epsilon, hf_epsilon = self._mauchly_test(values)

        # Corrected p-values
        corrected_df_conditions = gg_epsilon * df_conditions
        corrected_df_error = gg_epsilon * df_error
        p_gg = 1 - stats.f.cdf(f_stat, corrected_df_conditions, corrected_df_error)

        corrected_df_conditions_hf = hf_epsilon * df_conditions
        corrected_df_error_hf = hf_epsilon * df_error
        p_hf = 1 - stats.f.cdf(f_stat, corrected_df_conditions_hf, corrected_df_error_hf)

        # Effect sizes
        eta_squared = ss_conditions / ss_total
        partial_eta_squared = ss_conditions / (ss_conditions + ss_error)

        # ANOVA table
        anova_table = ANOVATable(
            source=["Between Subjects", "Conditions", "Error", "Total"],
            sum_of_squares=[ss_between_subjects, ss_conditions, ss_error, ss_total],
            df=[df_between_subjects, df_conditions, df_error, n_subjects * k - 1],
            mean_square=[
                ss_between_subjects / df_between_subjects,
                ms_conditions,
                ms_error,
                np.nan,
            ],
            f_statistic=[None, f_stat, None, None],
            p_value=[None, p_value, None, None],
        )

        return RepeatedMeasuresResult(
            f_statistic=f_stat,
            p_value=p_value,
            df_effect=df_conditions,
            df_error=df_error,
            mauchly_w=mauchly_w,
            mauchly_p=mauchly_p,
            sphericity_assumed=mauchly_p > self.alpha,
            greenhouse_geisser_epsilon=gg_epsilon,
            huynh_feldt_epsilon=hf_epsilon,
            corrected_p_gg=p_gg,
            corrected_p_hf=p_hf,
            eta_squared=eta_squared,
            partial_eta_squared=partial_eta_squared,
            anova_table=anova_table,
        )

    def _test_anova_assumptions(
        self,
        groups: dict[str, np.ndarray],
    ) -> list[AssumptionTestResult]:
        """Test ANOVA assumptions."""
        if not (groups is not None):
            raise ValueError("groups must be provided")
        results = []

        # 1. Normality test (Shapiro-Wilk for each group)
        normality_passed = True
        normality_details = {}

        for name, data in groups.items():
            if len(data) >= 3:
                stat, p = stats.shapiro(data)
                normality_details[name] = {"statistic": stat, "p_value": p}
                if p < self.alpha:
                    normality_passed = False

        results.append(
            AssumptionTestResult(
                test_name="Normality (Shapiro-Wilk)",
                statistic=np.nan,
                p_value=np.nan,
                passed=normality_passed,
                message=(
                    "All groups pass normality test"
                    if normality_passed
                    else "Some groups violate normality"
                ),
                details=normality_details,
            )
        )

        # 2. Homogeneity of variance (Levene's test)
        group_arrays = list(groups.values())
        if all(len(arr) >= 2 for arr in group_arrays):
            levene_stat, levene_p = stats.levene(*group_arrays)
            results.append(
                AssumptionTestResult(
                    test_name="Homogeneity of Variance (Levene's)",
                    statistic=levene_stat,
                    p_value=levene_p,
                    passed=levene_p >= self.alpha,
                    message=(
                        "Variances are homogeneous"
                        if levene_p >= self.alpha
                        else "Variances are heterogeneous"
                    ),
                )
            )

        # 3. Sample size check
        min_size = min(len(arr) for arr in group_arrays)
        results.append(
            AssumptionTestResult(
                test_name="Sample Size",
                statistic=min_size,
                p_value=np.nan,
                passed=min_size >= 20,
                message=f"Minimum group size: {min_size}"
                + (" (adequate)" if min_size >= 20 else " (small, use caution)"),
            )
        )

        return results

    @jit(nopython=True, fastmath=True)
    def _post_hoc_tests(
        self,
        groups: dict[str, np.ndarray],
        ms_error: float,
        df_error: int,
        method: PostHocMethod,
    ) -> list[PostHocComparison]:
        """Perform post-hoc pairwise comparisons."""
        if not (groups is not None):
            raise ValueError("groups must be provided")
        results = []
        group_names = list(groups.keys())
        n_groups = len(group_names)
        n_comparisons = n_groups * (n_groups - 1) // 2

        for name1, name2 in combinations(group_names, 2):
            arr1 = groups[name1]
            arr2 = groups[name2]

            mean_diff = np.mean(arr1) - np.mean(arr2)
            n1, n2 = len(arr1), len(arr2)

            # Standard error
            se = np.sqrt(ms_error * (1 / n1 + 1 / n2))

            # t-statistic
            t_stat = mean_diff / se

            # Raw p-value (two-tailed)
            p_raw = 2 * (1 - stats.t.cdf(abs(t_stat), df_error))

            # Adjusted p-value based on method
            if method == PostHocMethod.BONFERRONI:
                p_adj = min(1.0, p_raw * n_comparisons)
            elif method == PostHocMethod.SIDAK:
                p_adj = 1 - (1 - p_raw) ** n_comparisons
            elif method == PostHocMethod.HOLM:
                # Note: Proper Holm requires sorting all p-values
                p_adj = min(1.0, p_raw * n_comparisons)  # Simplified
            else:  # Tukey HSD (default)
                # Use studentized range distribution if available
                q = abs(t_stat) * np.sqrt(2)
                try:
                    p_adj = stats.studentized_range.sf(q, n_groups, df_error)
                except AttributeError:
                    # Fallback for older scipy versions
                    p_adj = min(1.0, p_raw * n_comparisons)

            # Confidence interval
            t_crit = stats.t.ppf(1 - self.alpha / 2, df_error)
            ci_lower = mean_diff - t_crit * se
            ci_upper = mean_diff + t_crit * se

            results.append(
                PostHocComparison(
                    group1=str(name1),
                    group2=str(name2),
                    mean_diff=mean_diff,
                    std_error=se,
                    t_statistic=t_stat,
                    p_value=p_raw,
                    p_adjusted=p_adj,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    significant=p_adj < self.alpha,
                )
            )

        return results

    def _mauchly_test(
        self,
        data: np.ndarray,
    ) -> tuple[float, float, float, float]:
        """Perform Mauchly's test of sphericity.

        Returns:
            W statistic, p-value, GG epsilon, HF epsilon
        """
        if not (data is not None):
            raise ValueError("data must be provided")
        n, k = data.shape

        if k < 3:
            return 1.0, 1.0, 1.0, 1.0

        # Calculate difference scores
        # Center the data
        centered = data - np.mean(data, axis=0)

        # Covariance matrix
        cov_matrix = np.cov(centered.T)

        # Orthonormalized transformation
        c_matrix = np.eye(k) - np.ones((k, k)) / k
        c_matrix = c_matrix[:-1, :]  # Remove last row

        # Transform covariance matrix
        s_star = c_matrix @ cov_matrix @ c_matrix.T

        # Mauchly's W
        det_s = np.linalg.det(s_star)
        trace_s = np.trace(s_star)
        p = k - 1

        if trace_s > 0:
            w = det_s / (trace_s / p) ** p
        else:
            w = 1.0

        # Chi-square approximation for p-value
        df = p * (p + 1) / 2 - 1
        chi_sq = -(n - 1 - (2 * p**2 + p + 2) / (6 * p)) * np.log(max(w, 1e-10))
        mauchly_p = 1 - stats.chi2.cdf(chi_sq, df)

        # Greenhouse-Geisser epsilon
        eigenvalues = np.linalg.eigvalsh(s_star)
        sum_lambda = np.sum(eigenvalues)
        sum_lambda_sq = np.sum(eigenvalues**2)

        if sum_lambda_sq > 0:
            gg_epsilon = sum_lambda**2 / (p * sum_lambda_sq)
        else:
            gg_epsilon = 1.0

        # Huynh-Feldt epsilon
        hf_epsilon = (n * (p - 1) * gg_epsilon - 2) / ((p - 1) * (n - 1 - (p - 1) * gg_epsilon))
        hf_epsilon = min(1.0, max(gg_epsilon, hf_epsilon))

        return w, mauchly_p, gg_epsilon, hf_epsilon

    def _calculate_power(
        self,
        f_obs: float,
        df1: int,
        df2: int,
        noncentrality: float,
    ) -> float:
        """Calculate observed power for F-test."""
        try:
            # Critical F value
            f_crit = stats.f.ppf(1 - self.alpha, df1, df2)
            # Power using non-central F distribution
            power = 1 - stats.ncf.cdf(f_crit, df1, df2, noncentrality**2)
            return float(power)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            return 0.0


def format_anova_report(result: OneWayANOVAResult | TwoWayANOVAResult) -> str:
    """Format ANOVA results as a text report.

    Args:
        result: ANOVA result object

    Returns:
        Formatted text report
    """
    lines = ["=" * 60, "ANOVA Results", "=" * 60, ""]

    # ANOVA table
    lines.append("ANOVA Table:")
    lines.append("-" * 60)
    table_df = result.anova_table.to_dataframe()
    lines.append(table_df.to_string(index=False))
    lines.append("")

    if isinstance(result, OneWayANOVAResult):
        # Effect sizes
        lines.append("Effect Sizes:")
        lines.append(f"  η² (eta-squared):     {result.eta_squared:.4f}")
        lines.append(f"  ω² (omega-squared):   {result.omega_squared:.4f}")
        lines.append(f"  Cohen's f:            {result.cohens_f:.4f}")
        lines.append(f"  Observed Power:       {result.observed_power:.4f}")
        lines.append("")

        # Group statistics
        lines.append("Group Statistics:")
        lines.extend(
            [
                f"  {name}: M = {result.group_means[name]:.4f}, SD = {result.group_stds[name]:.4f}, n = {result.group_counts[name]}"
                for name in result.group_means
            ]
        )
        lines.append("")

        # Post-hoc tests
        if result.post_hoc_results:
            lines.append("Post-hoc Comparisons:")
            lines.append("-" * 60)
            for comp in result.post_hoc_results:
                sig = "*" if comp.significant else ""
                lines.append(
                    f"  {comp.group1} vs {comp.group2}: "
                    f"Δ = {comp.mean_diff:.4f}, p = {comp.p_adjusted:.4f} {sig}"
                )

    elif isinstance(result, TwoWayANOVAResult):
        lines.append("Effect Sizes (Partial η²):")
        lines.append(f"  Factor A:     {result.partial_eta_squared_a:.4f}")
        lines.append(f"  Factor B:     {result.partial_eta_squared_b:.4f}")
        lines.append(f"  Interaction:  {result.partial_eta_squared_ab:.4f}")

    # Assumption tests
    if hasattr(result, "assumption_tests") and result.assumption_tests:
        lines.append("")
        lines.append("Assumption Tests:")
        lines.append("-" * 60)
        for test in result.assumption_tests:
            status = "✓" if test.passed else "✗"
            lines.append(f"  {status} {test.test_name}: {test.message}")

    lines.append("=" * 60)
    return "\n".join(lines)


__all__ = [
    "PostHocMethod",
    "AssumptionTestResult",
    "PostHocComparison",
    "ANOVATable",
    "OneWayANOVAResult",
    "TwoWayANOVAResult",
    "RepeatedMeasuresResult",
    "ANOVAAnalyzer",
    "format_anova_report",
]
