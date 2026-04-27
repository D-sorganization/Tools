"""One-way ANOVA helpers and diagnostics."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from .anova_models import (
    ANOVATable,
    AssumptionTestResult,
    OneWayANOVAResult,
    PostHocComparison,
    PostHocMethod,
)


def validate_and_prepare_groups(
    data: pd.DataFrame | dict[str, np.ndarray],
    dependent_var: str | None,
    group_var: str | None,
) -> dict[str, np.ndarray]:
    """Validate one-way ANOVA input and return cleaned group arrays."""
    if isinstance(data, dict):
        if len(data) < 2:
            raise ValueError("ANOVA requires at least 2 groups")
        return {
            str(name): np.asarray([value for value in values if not np.isnan(value)])
            for name, values in data.items()
        }

    if dependent_var is None or group_var is None:
        raise ValueError(
            "dependent_var and group_var must be provided for DataFrame input"
        )
    if dependent_var not in data.columns or group_var not in data.columns:
        raise ValueError("Specified columns not found in DataFrame")

    groups = data.groupby(group_var)[dependent_var].apply(list).to_dict()
    if len(groups) < 2:
        raise ValueError("ANOVA requires at least 2 groups")
    return {
        str(name): np.asarray([value for value in values if not np.isnan(value)])
        for name, values in groups.items()
    }


def compute_anova_statistics(
    group_arrays: dict[str, np.ndarray],
) -> tuple[float, float, float, float, float, float, int, int, int, float]:
    """Compute sums of squares, mean squares, and the F statistic."""
    k = len(group_arrays)
    n_total = sum(len(array) for array in group_arrays.values())
    grand_mean = float(np.mean(np.concatenate(list(group_arrays.values()))))

    ss_between = sum(
        len(array) * (float(np.mean(array)) - grand_mean) ** 2
        for array in group_arrays.values()
    )
    ss_within = sum(
        float(np.sum((array - float(np.mean(array))) ** 2))
        for array in group_arrays.values()
    )
    ss_total = ss_between + ss_within

    df_between = k - 1
    df_within = n_total - k
    df_total = n_total - 1

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_statistic = ms_between / ms_within

    return (
        ss_between,
        ss_within,
        ss_total,
        ms_between,
        ms_within,
        f_statistic,
        df_between,
        df_within,
        df_total,
        grand_mean,
    )


def test_anova_assumptions(
    alpha: float,
    groups: dict[str, np.ndarray],
) -> list[AssumptionTestResult]:
    """Run normality, variance, and sample-size checks for ANOVA inputs."""
    results: list[AssumptionTestResult] = []

    normality_passed = True
    normality_details: dict[str, dict[str, float]] = {}
    for name, data in groups.items():
        if len(data) >= 3:
            statistic, p_value = stats.shapiro(data)
            normality_details[name] = {
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
            if p_value < alpha:
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

    group_arrays = list(groups.values())
    if all(len(array) >= 2 for array in group_arrays):
        levene_stat, levene_p = stats.levene(*group_arrays)
        results.append(
            AssumptionTestResult(
                test_name="Homogeneity of Variance (Levene's)",
                statistic=float(levene_stat),
                p_value=float(levene_p),
                passed=levene_p >= alpha,
                message=(
                    "Variances are homogeneous"
                    if levene_p >= alpha
                    else "Variances are heterogeneous"
                ),
            )
        )

    min_size = min(len(array) for array in group_arrays)
    results.append(
        AssumptionTestResult(
            test_name="Sample Size",
            statistic=float(min_size),
            p_value=np.nan,
            passed=min_size >= 20,
            message=f"Minimum group size: {min_size}"
            + (" (adequate)" if min_size >= 20 else " (small, use caution)"),
        )
    )

    return results


def post_hoc_tests(
    alpha: float,
    groups: dict[str, np.ndarray],
    ms_error: float,
    df_error: int,
    method: PostHocMethod,
) -> list[PostHocComparison]:
    """Perform pairwise post-hoc comparisons."""
    results: list[PostHocComparison] = []
    group_names = list(groups.keys())
    n_groups = len(group_names)
    n_comparisons = n_groups * (n_groups - 1) // 2

    for name1, name2 in combinations(group_names, 2):
        arr1 = groups[name1]
        arr2 = groups[name2]

        mean_diff = float(np.mean(arr1) - np.mean(arr2))
        n1, n2 = len(arr1), len(arr2)
        std_error = float(np.sqrt(ms_error * (1 / n1 + 1 / n2)))
        t_statistic = mean_diff / std_error
        p_raw = float(2 * (1 - stats.t.cdf(abs(t_statistic), df_error)))

        if method == PostHocMethod.BONFERRONI:
            p_adjusted = min(1.0, p_raw * n_comparisons)
        elif method == PostHocMethod.SIDAK:
            p_adjusted = 1 - (1 - p_raw) ** n_comparisons
        elif method == PostHocMethod.HOLM:
            p_adjusted = min(1.0, p_raw * n_comparisons)
        else:
            q_statistic = abs(t_statistic) * np.sqrt(2)
            try:
                p_adjusted = float(
                    stats.studentized_range.sf(q_statistic, n_groups, df_error)
                )
            except AttributeError:
                p_adjusted = min(1.0, p_raw * n_comparisons)

        t_critical = float(stats.t.ppf(1 - alpha / 2, df_error))
        ci_lower = mean_diff - t_critical * std_error
        ci_upper = mean_diff + t_critical * std_error

        results.append(
            PostHocComparison(
                group1=name1,
                group2=name2,
                mean_diff=mean_diff,
                std_error=std_error,
                t_statistic=float(t_statistic),
                p_value=p_raw,
                p_adjusted=float(p_adjusted),
                ci_lower=float(ci_lower),
                ci_upper=float(ci_upper),
                significant=p_adjusted < alpha,
            )
        )

    return results


def calculate_power(alpha: float, df1: int, df2: int, noncentrality: float) -> float:
    """Calculate observed power for an F-test."""
    try:
        f_critical = stats.f.ppf(1 - alpha, df1, df2)
        return float(1 - stats.ncf.cdf(f_critical, df1, df2, noncentrality**2))
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return 0.0


def perform_one_way_anova(
    alpha: float,
    data: pd.DataFrame | dict[str, np.ndarray],
    dependent_var: str | None,
    group_var: str | None,
    post_hoc: PostHocMethod | None,
    test_assumptions: bool,
) -> OneWayANOVAResult:
    """Perform one-way ANOVA and return a complete result object."""
    group_arrays = validate_and_prepare_groups(data, dependent_var, group_var)
    (
        ss_between,
        ss_within,
        ss_total,
        ms_between,
        ms_within,
        f_statistic,
        df_between,
        df_within,
        df_total,
        grand_mean,
    ) = compute_anova_statistics(group_arrays)

    n_total = sum(len(array) for array in group_arrays.values())
    p_value = float(1 - stats.f.cdf(f_statistic, df_between, df_within))

    eta_squared = ss_between / ss_total
    omega_squared = max(
        0.0, (ss_between - df_between * ms_within) / (ss_total + ms_within)
    )
    cohens_f = float(np.sqrt(eta_squared / (1 - eta_squared)) if eta_squared < 1 else 0)

    anova_table = ANOVATable(
        source=["Between Groups", "Within Groups", "Total"],
        sum_of_squares=[ss_between, ss_within, ss_total],
        df=[df_between, df_within, df_total],
        mean_square=[ms_between, ms_within, np.nan],
        f_statistic=[f_statistic, None, None],
        p_value=[p_value, None, None],
    )

    assumption_results = (
        test_anova_assumptions(alpha, group_arrays) if test_assumptions else []
    )
    post_hoc_results = (
        post_hoc_tests(alpha, group_arrays, ms_within, df_within, post_hoc)
        if post_hoc and p_value < alpha
        else []
    )
    noncentrality = (
        np.sqrt(n_total * eta_squared / (1 - eta_squared)) if eta_squared < 1 else 0.0
    )

    return OneWayANOVAResult(
        f_statistic=float(f_statistic),
        p_value=p_value,
        df_between=df_between,
        df_within=df_within,
        anova_table=anova_table,
        eta_squared=float(eta_squared),
        omega_squared=float(omega_squared),
        cohens_f=cohens_f,
        group_means={
            name: float(np.mean(array)) for name, array in group_arrays.items()
        },
        group_stds={
            name: float(np.std(array, ddof=1)) for name, array in group_arrays.items()
        },
        group_counts={name: len(array) for name, array in group_arrays.items()},
        grand_mean=float(grand_mean),
        post_hoc_results=post_hoc_results,
        assumption_tests=assumption_results,
        observed_power=calculate_power(alpha, df_between, df_within, noncentrality),
    )
