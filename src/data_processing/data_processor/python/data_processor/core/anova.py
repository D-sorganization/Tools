"""Facade for the decomposed ANOVA analysis modules."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .anova_models import (
    ANOVATable,
    AssumptionTestResult,
    OneWayANOVAResult,
    PostHocComparison,
    PostHocMethod,
    RepeatedMeasuresResult,
    TwoWayANOVAResult,
    format_anova_report,
)
from .anova_one_way import (
    calculate_power,
    compute_anova_statistics,
    perform_one_way_anova,
    post_hoc_tests,
    test_anova_assumptions,
    validate_and_prepare_groups,
)
from .anova_repeated import mauchly_test, perform_repeated_measures_anova
from .anova_two_way import (
    perform_two_way_anova,
    two_way_assumption_tests,
    two_way_effect_sizes,
    two_way_f_tests,
    two_way_sum_of_squares,
)

logger = logging.getLogger(__name__)


class ANOVAAnalyzer:
    """Facade over the decomposed one-way, two-way, and repeated-measures ANOVA helpers."""

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def _validate_and_prepare_groups(
        self,
        data: pd.DataFrame | dict[str, np.ndarray],
        dependent_var: str | None,
        group_var: str | None,
    ) -> dict[str, np.ndarray]:
        """Validate inputs and prepare group arrays for one-way ANOVA."""
        return validate_and_prepare_groups(data, dependent_var, group_var)

    def _compute_anova_statistics(
        self,
        group_arrays: dict[str, np.ndarray],
    ) -> tuple[float, float, float, float, float, float, int, int, int, float]:
        """Compute core one-way ANOVA sums of squares and F statistic."""
        return compute_anova_statistics(group_arrays)

    def one_way_anova(
        self,
        data: pd.DataFrame | dict[str, np.ndarray],
        dependent_var: str | None = None,
        group_var: str | None = None,
        post_hoc: PostHocMethod | None = PostHocMethod.TUKEY_HSD,
        test_assumptions: bool = True,
    ) -> OneWayANOVAResult:
        """Perform one-way ANOVA on either a grouped dict or a DataFrame."""
        return perform_one_way_anova(
            self.alpha,
            data,
            dependent_var,
            group_var,
            post_hoc,
            test_assumptions,
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
        """Perform two-way ANOVA."""
        return perform_two_way_anova(
            self.alpha,
            df,
            dependent_var,
            factor_a,
            factor_b,
            test_interaction,
            test_assumptions,
        )

    def _two_way_sum_of_squares(
        self,
        data: pd.DataFrame,
        y: np.ndarray,
        dependent_var: str,
        factor_a: str,
        factor_b: str,
        levels_a: np.ndarray,
        levels_b: np.ndarray,
        marginal_a: dict[str, float],
        marginal_b: dict[str, float],
        grand_mean: float,
        test_interaction: bool,
    ) -> dict[str, float]:
        """Backward-compatible wrapper for the extracted two-way SS helper."""
        return two_way_sum_of_squares(
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

    def _two_way_f_tests(
        self,
        ss: dict[str, float],
        a: int,
        b: int,
        n_total: int,
    ) -> dict[str, float]:
        """Backward-compatible wrapper for the extracted F-test helper."""
        return two_way_f_tests(ss, a, b, n_total)

    def _two_way_effect_sizes(self, ss: dict[str, float]) -> dict[str, float]:
        """Backward-compatible wrapper for extracted effect-size helpers."""
        return two_way_effect_sizes(ss)

    def _two_way_assumption_tests(
        self,
        data: pd.DataFrame,
        dependent_var: str,
        factor_a: str,
        factor_b: str,
        levels_a: np.ndarray,
        levels_b: np.ndarray,
        test_assumptions: bool,
    ) -> list[AssumptionTestResult]:
        """Backward-compatible wrapper for extracted two-way diagnostics."""
        return two_way_assumption_tests(
            self.alpha,
            data,
            dependent_var,
            factor_a,
            factor_b,
            levels_a,
            levels_b,
            test_assumptions,
        )

    def repeated_measures_anova(
        self,
        df: pd.DataFrame,
        dependent_vars: list[str],
        subject_id: str,
    ) -> RepeatedMeasuresResult:
        """Perform one-way repeated-measures ANOVA."""
        return perform_repeated_measures_anova(
            self.alpha, df, dependent_vars, subject_id
        )

    def _test_anova_assumptions(
        self,
        groups: dict[str, np.ndarray],
    ) -> list[AssumptionTestResult]:
        """Backward-compatible wrapper for extracted assumption tests."""
        return test_anova_assumptions(self.alpha, groups)

    def _post_hoc_tests(
        self,
        groups: dict[str, np.ndarray],
        ms_error: float,
        df_error: int,
        method: PostHocMethod,
    ) -> list[PostHocComparison]:
        """Backward-compatible wrapper for extracted post-hoc helpers."""
        return post_hoc_tests(self.alpha, groups, ms_error, df_error, method)

    def _mauchly_test(self, data: np.ndarray) -> tuple[float, float, float, float]:
        """Backward-compatible wrapper for the extracted sphericity helper."""
        return mauchly_test(data)

    def _calculate_power(
        self,
        f_obs: float,
        df1: int,
        df2: int,
        noncentrality: float,
    ) -> float:
        """Backward-compatible wrapper for observed power calculation."""
        return calculate_power(self.alpha, df1, df2, noncentrality)


__all__ = [
    "ANOVAAnalyzer",
    "ANOVATable",
    "AssumptionTestResult",
    "OneWayANOVAResult",
    "PostHocComparison",
    "PostHocMethod",
    "RepeatedMeasuresResult",
    "TwoWayANOVAResult",
    "format_anova_report",
]
