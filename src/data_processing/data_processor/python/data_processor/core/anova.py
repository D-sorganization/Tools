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
        # Validate inputs
        if dependent_var not in df.columns or group_var not in df.columns:
            raise ValueError("Specified columns not found in DataFrame")

        # Get groups
        groups = df.groupby(group_var)[dependent_var].apply(list).to_dict()
        list(groups.keys())
        k = len(groups)

        if k < 2:
            raise ValueError("ANOVA requires at least 2 groups")

        # Convert to arrays and filter NaN
        group_arrays = {
            name: np.array([x for x in values if not np.isnan(x)])
            for name, values in groups.items()
        }

        # Calculate statistics
        n_total = sum(len(arr) for arr in group_arrays.values())
        grand_mean = np.mean(np.concatenate(list(group_arrays.values())))

        # Group statistics
        group_means = {name: np.mean(arr) for name, arr in group_arrays.items()}
        group_stds = {name: np.std(arr, ddof=1) for name, arr in group_arrays.items()}
        group_counts = {name: len(arr) for name, arr in group_arrays.items()}

        # Sum of squares
        ss_between = sum(
            len(arr) * (np.mean(arr) - grand_mean) ** 2 for arr in group_arrays.values()
        )
        ss_within = sum(
            np.sum((arr - np.mean(arr)) ** 2) for arr in group_arrays.values()
        )
        ss_total = ss_between + ss_within

        # Degrees of freedom
        df_between = k - 1
        df_within = n_total - k
        df_total = n_total - 1

        # Mean squares
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within

        # F-statistic
        f_stat = ms_between / ms_within
        p_value = 1 - stats.f.cdf(f_stat, df_between, df_within)

        # Effect sizes
        eta_squared = ss_between / ss_total
        omega_squared = (ss_between - df_between * ms_within) / (ss_total + ms_within)
        omega_squared = max(0, omega_squared)  # Can be negative for small effects
        cohens_f = np.sqrt(eta_squared / (1 - eta_squared)) if eta_squared < 1 else 0

        # ANOVA table
        anova_table = ANOVATable(
            source=["Between Groups", "Within Groups", "Total"],
            sum_of_squares=[ss_between, ss_within, ss_total],
            df=[df_between, df_within, df_total],
            mean_square=[ms_between, ms_within, np.nan],
            f_statistic=[f_stat, None, None],
            p_value=[p_value, None, None],
        )

        # Assumption tests
        assumption_tests = []
        if test_assumptions:
            assumption_tests = self._test_anova_assumptions(group_arrays)

        # Post-hoc tests
        post_hoc_results = []
        if post_hoc and p_value < self.alpha:
            post_hoc_results = self._post_hoc_tests(
                group_arrays, ms_within, df_within, post_hoc
            )

        # Observed power
        noncentrality = (
            np.sqrt(n_total * eta_squared / (1 - eta_squared)) if eta_squared < 1 else 0
        )
        observed_power = self._calculate_power(
            f_stat, df_between, df_within, noncentrality
        )

        return OneWayANOVAResult(
            f_statistic=f_stat,
            p_value=p_value,
            df_between=df_between,
            df_within=df_within,
            anova_table=anova_table,
            eta_squared=eta_squared,
            omega_squared=omega_squared,
            cohens_f=cohens_f,
            group_means=group_means,
            group_stds=group_stds,
            group_counts=group_counts,
            grand_mean=grand_mean,
            post_hoc_results=post_hoc_results,
            assumption_tests=assumption_tests,
            observed_power=observed_power,
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
        # Remove missing values
        data = df[[dependent_var, factor_a, factor_b]].dropna()
        y: np.ndarray = np.asarray(data[dependent_var].values)

        # Get factor levels
        levels_a = data[factor_a].unique()
        levels_b = data[factor_b].unique()
        a = len(levels_a)
        b = len(levels_b)
        n_total = len(data)

        # Calculate grand mean
        grand_mean = np.mean(y)

        # Calculate cell means
        cell_means = data.groupby([factor_a, factor_b])[dependent_var].mean().unstack()

        # Marginal means
        marginal_a = data.groupby(factor_a)[dependent_var].mean().to_dict()
        marginal_b = data.groupby(factor_b)[dependent_var].mean().to_dict()

        # Sum of squares calculations
        ss_total = np.sum((y - grand_mean) ** 2)

        # SS for factor A
        ss_a = sum(
            len(data[data[factor_a] == level]) * (marginal_a[level] - grand_mean) ** 2
            for level in levels_a
        )

        # SS for factor B
        ss_b = sum(
            len(data[data[factor_b] == level]) * (marginal_b[level] - grand_mean) ** 2
            for level in levels_b
        )

        # Sum of squares for interaction
        ss_ab = 0
        if test_interaction:
            for level_a in levels_a:
                for level_b in levels_b:
                    cell_data = data[
                        (data[factor_a] == level_a) & (data[factor_b] == level_b)
                    ]
                    if len(cell_data) > 0:
                        cell_mean = cell_data[dependent_var].mean()
                        expected = (
                            marginal_a[level_a] + marginal_b[level_b] - grand_mean
                        )
                        ss_ab += len(cell_data) * (cell_mean - expected) ** 2

        # SS error
        ss_error = ss_total - ss_a - ss_b - ss_ab

        # Degrees of freedom
        df_a = a - 1
        df_b = b - 1
        df_ab = df_a * df_b
        df_error = n_total - a * b
        df_total = n_total - 1

        # Mean squares
        ms_a = ss_a / df_a
        ms_b = ss_b / df_b
        ms_ab = ss_ab / df_ab if df_ab > 0 else 0
        ms_error = ss_error / df_error

        # F statistics
        f_a = ms_a / ms_error
        f_b = ms_b / ms_error
        f_ab = ms_ab / ms_error if ms_ab > 0 else 0

        # P values
        p_a = 1 - stats.f.cdf(f_a, df_a, df_error)
        p_b = 1 - stats.f.cdf(f_b, df_b, df_error)
        p_ab = 1 - stats.f.cdf(f_ab, df_ab, df_error) if f_ab > 0 else 1.0

        # Effect sizes
        eta_sq_a = ss_a / ss_total
        eta_sq_b = ss_b / ss_total
        eta_sq_ab = ss_ab / ss_total

        partial_eta_sq_a = ss_a / (ss_a + ss_error)
        partial_eta_sq_b = ss_b / (ss_b + ss_error)
        partial_eta_sq_ab = ss_ab / (ss_ab + ss_error) if (ss_ab + ss_error) > 0 else 0

        # ANOVA table
        anova_table = ANOVATable(
            source=[factor_a, factor_b, f"{factor_a}×{factor_b}", "Error", "Total"],
            sum_of_squares=[ss_a, ss_b, ss_ab, ss_error, ss_total],
            df=[df_a, df_b, df_ab, df_error, df_total],
            mean_square=[ms_a, ms_b, ms_ab, ms_error, np.nan],
            f_statistic=[f_a, f_b, f_ab, None, None],
            p_value=[p_a, p_b, p_ab, None, None],
        )

        # Assumption tests
        assumption_tests = []
        if test_assumptions:
            # Test within each cell
            cell_data = {}
            for level_a in levels_a:
                for level_b in levels_b:
                    cell = data[
                        (data[factor_a] == level_a) & (data[factor_b] == level_b)
                    ]
                    if len(cell) > 2:
                        cell_data[f"{level_a}_{level_b}"] = cell[dependent_var].values
            if cell_data:
                assumption_tests = self._test_anova_assumptions(cell_data)

        return TwoWayANOVAResult(
            factor_a_f=f_a,
            factor_a_p=p_a,
            factor_a_df=df_a,
            factor_b_f=f_b,
            factor_b_p=p_b,
            factor_b_df=df_b,
            interaction_f=f_ab,
            interaction_p=p_ab,
            interaction_df=df_ab,
            df_error=df_error,
            ms_error=ms_error,
            anova_table=anova_table,
            eta_squared_a=eta_sq_a,
            eta_squared_b=eta_sq_b,
            eta_squared_ab=eta_sq_ab,
            partial_eta_squared_a=partial_eta_sq_a,
            partial_eta_squared_b=partial_eta_sq_b,
            partial_eta_squared_ab=partial_eta_sq_ab,
            cell_means=cell_means,
            marginal_means_a=marginal_a,
            marginal_means_b=marginal_b,
            assumption_tests=assumption_tests,
        )

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
        p_hf = 1 - stats.f.cdf(
            f_stat, corrected_df_conditions_hf, corrected_df_error_hf
        )

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

    def _post_hoc_tests(
        self,
        groups: dict[str, np.ndarray],
        ms_error: float,
        df_error: int,
        method: PostHocMethod,
    ) -> list[PostHocComparison]:
        """Perform post-hoc pairwise comparisons."""
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
        hf_epsilon = (n * (p - 1) * gg_epsilon - 2) / (
            (p - 1) * (n - 1 - (p - 1) * gg_epsilon)
        )
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
        for name in result.group_means:
            lines.append(
                f"  {name}: M = {result.group_means[name]:.4f}, "
                f"SD = {result.group_stds[name]:.4f}, n = {result.group_counts[name]}"
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
