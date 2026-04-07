"""Two-way ANOVA helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .anova_models import ANOVATable, AssumptionTestResult, TwoWayANOVAResult
from .anova_one_way import test_anova_assumptions

try:
    from numba import jit
except ImportError:

    def jit(*_args: Any, **_kwargs: Any):  # type: ignore[no-redef]
        """Fallback no-op decorator when numba is unavailable."""

        def decorator(func: Any) -> Any:
            return func

        return decorator


@jit(nopython=True, fastmath=True)
def two_way_sum_of_squares(
    data: pd.DataFrame,
    y: np.ndarray,
    dependent_var: str,
    factor_a: str,
    factor_b: str,
    levels_a: np.ndarray,
    levels_b: np.ndarray,
    marginal_a: dict[Any, float],
    marginal_b: dict[Any, float],
    grand_mean: float,
    test_interaction: bool,
) -> dict[str, float]:
    """Compute SS_total, SS_A, SS_B, SS_AB, and SS_error."""
    ss_total = float(np.sum((y - grand_mean) ** 2))

    ss_a = sum(
        len(data[data[factor_a] == level]) * (marginal_a[level] - grand_mean) ** 2
        for level in levels_a
    )
    ss_b = sum(
        len(data[data[factor_b] == level]) * (marginal_b[level] - grand_mean) ** 2
        for level in levels_b
    )

    ss_ab = 0.0
    if test_interaction:
        for level_a in levels_a:
            for level_b in levels_b:
                cell = data[(data[factor_a] == level_a) & (data[factor_b] == level_b)]
                if len(cell) > 0:
                    expected = marginal_a[level_a] + marginal_b[level_b] - grand_mean
                    cell_mean = float(cell[dependent_var].mean())
                    ss_ab += len(cell) * (cell_mean - expected) ** 2

    return {
        "total": ss_total,
        "a": float(ss_a),
        "b": float(ss_b),
        "ab": float(ss_ab),
        "error": ss_total - float(ss_a) - float(ss_b) - float(ss_ab),
    }


def two_way_f_tests(
    ss: dict[str, float],
    a: int,
    b: int,
    n_total: int,
) -> dict[str, float]:
    """Compute degrees of freedom, mean squares, F-statistics, and p-values."""
    df_a = a - 1
    df_b = b - 1
    df_ab = df_a * df_b
    df_error = n_total - a * b

    ms_a = ss["a"] / df_a
    ms_b = ss["b"] / df_b
    ms_ab = ss["ab"] / df_ab if df_ab > 0 else 0.0
    ms_error = ss["error"] / df_error

    f_a = ms_a / ms_error
    f_b = ms_b / ms_error
    f_ab = ms_ab / ms_error if ms_ab > 0 else 0.0

    return {
        "df_a": float(df_a),
        "df_b": float(df_b),
        "df_ab": float(df_ab),
        "df_error": float(df_error),
        "ms_a": float(ms_a),
        "ms_b": float(ms_b),
        "ms_ab": float(ms_ab),
        "ms_error": float(ms_error),
        "f_a": float(f_a),
        "f_b": float(f_b),
        "f_ab": float(f_ab),
        "p_a": float(1 - stats.f.cdf(f_a, df_a, df_error)),
        "p_b": float(1 - stats.f.cdf(f_b, df_b, df_error)),
        "p_ab": float(1 - stats.f.cdf(f_ab, df_ab, df_error) if f_ab > 0 else 1.0),
    }


def two_way_effect_sizes(ss: dict[str, float]) -> dict[str, float]:
    """Compute eta-squared and partial eta-squared effect sizes."""
    ss_total = ss["total"]
    ss_error = ss["error"]
    return {
        "eta_a": ss["a"] / ss_total,
        "eta_b": ss["b"] / ss_total,
        "eta_ab": ss["ab"] / ss_total,
        "partial_eta_a": ss["a"] / (ss["a"] + ss_error),
        "partial_eta_b": ss["b"] / (ss["b"] + ss_error),
        "partial_eta_ab": (
            ss["ab"] / (ss["ab"] + ss_error) if (ss["ab"] + ss_error) > 0 else 0.0
        ),
    }


def two_way_assumption_tests(
    alpha: float,
    data: pd.DataFrame,
    dependent_var: str,
    factor_a: str,
    factor_b: str,
    levels_a: np.ndarray,
    levels_b: np.ndarray,
    test_assumptions: bool,
) -> list[AssumptionTestResult]:
    """Run cell-wise assumption tests for a two-way ANOVA."""
    if not test_assumptions:
        return []

    cell_data: dict[str, np.ndarray] = {}
    for level_a in levels_a:
        for level_b in levels_b:
            cell = data[(data[factor_a] == level_a) & (data[factor_b] == level_b)]
            if len(cell) > 2:
                cell_data[f"{level_a}_{level_b}"] = cell[dependent_var].values
    return test_anova_assumptions(alpha, cell_data) if cell_data else []


def perform_two_way_anova(
    alpha: float,
    df: pd.DataFrame,
    dependent_var: str,
    factor_a: str,
    factor_b: str,
    test_interaction: bool,
    test_assumptions: bool,
) -> TwoWayANOVAResult:
    """Perform two-way ANOVA and return a complete result object."""
    data = df[[dependent_var, factor_a, factor_b]].dropna()
    y = np.asarray(data[dependent_var].values)

    levels_a = data[factor_a].unique()
    levels_b = data[factor_b].unique()
    n_total = len(data)
    grand_mean = float(np.mean(y))
    cell_means = data.groupby([factor_a, factor_b])[dependent_var].mean().unstack()
    marginal_a = data.groupby(factor_a)[dependent_var].mean().to_dict()
    marginal_b = data.groupby(factor_b)[dependent_var].mean().to_dict()

    ss = two_way_sum_of_squares(
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
    ftest = two_way_f_tests(ss, len(levels_a), len(levels_b), n_total)
    effect = two_way_effect_sizes(ss)

    interaction_label = f"{factor_a}×{factor_b}"
    anova_table = ANOVATable(
        source=[factor_a, factor_b, interaction_label, "Error", "Total"],
        sum_of_squares=[ss["a"], ss["b"], ss["ab"], ss["error"], ss["total"]],
        df=[
            int(ftest["df_a"]),
            int(ftest["df_b"]),
            int(ftest["df_ab"]),
            int(ftest["df_error"]),
            n_total - 1,
        ],
        mean_square=[
            ftest["ms_a"],
            ftest["ms_b"],
            ftest["ms_ab"],
            ftest["ms_error"],
            np.nan,
        ],
        f_statistic=[ftest["f_a"], ftest["f_b"], ftest["f_ab"], None, None],
        p_value=[ftest["p_a"], ftest["p_b"], ftest["p_ab"], None, None],
    )

    return TwoWayANOVAResult(
        factor_a_f=ftest["f_a"],
        factor_a_p=ftest["p_a"],
        factor_a_df=int(ftest["df_a"]),
        factor_b_f=ftest["f_b"],
        factor_b_p=ftest["p_b"],
        factor_b_df=int(ftest["df_b"]),
        interaction_f=ftest["f_ab"],
        interaction_p=ftest["p_ab"],
        interaction_df=int(ftest["df_ab"]),
        df_error=int(ftest["df_error"]),
        ms_error=ftest["ms_error"],
        anova_table=anova_table,
        eta_squared_a=effect["eta_a"],
        eta_squared_b=effect["eta_b"],
        eta_squared_ab=effect["eta_ab"],
        partial_eta_squared_a=effect["partial_eta_a"],
        partial_eta_squared_b=effect["partial_eta_b"],
        partial_eta_squared_ab=effect["partial_eta_ab"],
        cell_means=cell_means,
        marginal_means_a={str(key): float(value) for key, value in marginal_a.items()},
        marginal_means_b={str(key): float(value) for key, value in marginal_b.items()},
        assumption_tests=two_way_assumption_tests(
            alpha,
            data,
            dependent_var,
            factor_a,
            factor_b,
            levels_a,
            levels_b,
            test_assumptions,
        ),
    )
