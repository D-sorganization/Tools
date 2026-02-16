"""Tests for data_processor.core.regression_types module."""

from __future__ import annotations

import numpy as np
import pytest
from data_processor.core.regression_types import (
    CoefficientInfo,
    RegressionConfig,
    RegressionDiagnostics,
    RegressionResult,
    RegularizationType,
    SelectionMethod,
)


class TestRegularizationType:
    """Tests for RegularizationType enum."""

    def test_values(self) -> None:
        assert RegularizationType.NONE.value == "none"
        assert RegularizationType.RIDGE.value == "ridge"
        assert RegularizationType.LASSO.value == "lasso"
        assert RegularizationType.ELASTIC_NET.value == "elastic_net"


class TestSelectionMethod:
    """Tests for SelectionMethod enum."""

    def test_values(self) -> None:
        assert SelectionMethod.NONE.value == "none"
        assert SelectionMethod.FORWARD.value == "forward"
        assert SelectionMethod.BACKWARD.value == "backward"
        assert SelectionMethod.STEPWISE.value == "stepwise"


class TestCoefficientInfo:
    """Tests for CoefficientInfo dataclass."""

    def test_construction(self) -> None:
        ci = CoefficientInfo(
            name="x1",
            estimate=1.5,
            std_error=0.2,
            t_statistic=7.5,
            p_value=0.001,
            ci_lower=1.1,
            ci_upper=1.9,
        )
        assert ci.name == "x1"
        assert ci.estimate == 1.5
        assert ci.vif == 1.0  # default

    def test_significant_true(self) -> None:
        ci = CoefficientInfo(
            name="x",
            estimate=1.0,
            std_error=0.1,
            t_statistic=10.0,
            p_value=0.001,
            ci_lower=0.8,
            ci_upper=1.2,
        )
        assert ci.significant is True

    def test_significant_false(self) -> None:
        ci = CoefficientInfo(
            name="x",
            estimate=0.1,
            std_error=0.5,
            t_statistic=0.2,
            p_value=0.8,
            ci_lower=-0.9,
            ci_upper=1.1,
        )
        assert ci.significant is False

    def test_significance_boundary(self) -> None:
        """p=0.05 is NOT significant (< 0.05 required)."""
        ci = CoefficientInfo(
            name="x",
            estimate=1.0,
            std_error=0.5,
            t_statistic=2.0,
            p_value=0.05,
            ci_lower=0.0,
            ci_upper=2.0,
        )
        assert ci.significant is False

    def test_custom_vif(self) -> None:
        ci = CoefficientInfo(
            name="x",
            estimate=1.0,
            std_error=0.1,
            t_statistic=10.0,
            p_value=0.001,
            ci_lower=0.8,
            ci_upper=1.2,
            vif=5.5,
        )
        assert ci.vif == 5.5


class TestRegressionDiagnostics:
    """Tests for RegressionDiagnostics dataclass."""

    def test_construction(self) -> None:
        diag = RegressionDiagnostics(
            residuals=np.array([0.1, -0.2, 0.05]),
            standardized_residuals=np.array([0.5, -1.0, 0.25]),
            studentized_residuals=np.array([0.6, -1.1, 0.3]),
            leverage=np.array([0.1, 0.2, 0.15]),
            cooks_distance=np.array([0.01, 0.05, 0.02]),
            durbin_watson=1.95,
            breusch_pagan_stat=1.2,
            breusch_pagan_p=0.3,
            shapiro_stat=0.98,
            shapiro_p=0.6,
            high_leverage_points=[],
            influential_points=[],
            outlier_points=[1],
        )
        assert diag.durbin_watson == pytest.approx(1.95)
        assert len(diag.outlier_points) == 1


class TestRegressionResult:
    """Tests for RegressionResult dataclass."""

    def test_construction(self) -> None:
        result = RegressionResult(
            model_type="ols",
            n_observations=100,
            n_predictors=2,
            coefficients=[],
            intercept=0.5,
            intercept_se=0.1,
            r_squared=0.85,
            adj_r_squared=0.84,
            rmse=0.5,
            mae=0.4,
            aic=200.0,
            bic=210.0,
            f_statistic=50.0,
            f_p_value=0.0001,
            df_model=2,
            df_residual=97,
            fitted_values=np.zeros(100),
            residuals=np.zeros(100),
            feature_names=["x1", "x2"],
        )
        assert result.r_squared == 0.85
        assert result.n_observations == 100
        assert result.diagnostics is None
        assert result.predict_func is None
        assert result.variable_importance == {}


class TestRegressionConfig:
    """Tests for RegressionConfig dataclass."""

    def test_defaults(self) -> None:
        config = RegressionConfig()
        assert config.regularization == RegularizationType.NONE
        assert config.alpha == 1.0
        assert config.l1_ratio == 0.5
        assert config.polynomial_degree == 1
        assert config.confidence_level == 0.95
        assert config.compute_diagnostics is True

    def test_custom_config(self) -> None:
        config = RegressionConfig(
            regularization=RegularizationType.RIDGE,
            alpha=0.5,
            standardize=True,
        )
        assert config.regularization == RegularizationType.RIDGE
        assert config.alpha == 0.5
        assert config.standardize is True
