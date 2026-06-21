"""Tests for data_processor.core.state_space.

Covers the previously-untested SeasonalModel/ARIMAStateSpace fit paths
(issue #3664) and guards the robustness fixes in sibling issues:

- #3697 SeasonalModel enforces positive variances (squared params)
- #3698 _normal_ppf raises on invalid p instead of returning 0.0
- #3699 _kalman_filter guards against zero innovation covariance (ARIMA H=0)
"""

from __future__ import annotations

import numpy as np
import pytest
from data_processor.core.state_space import (
    ARIMAStateSpace,
    ForecastResult,
    LocalLevelModel,
    LocalLinearTrendModel,
    OptimizationMethod,
    SeasonalModel,
    StateSpaceConfig,
    StateSpaceModelType,
    fit_state_space,
)

pytestmark = pytest.mark.unit


def _seasonal_series(n: int = 60, period: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    trend = np.linspace(0, 5, n)
    season = np.tile(np.array([1.0, -0.5, 0.2, -0.7])[:period], n // period + 1)[:n]
    return trend + season + rng.randn(n) * 0.2


def _arma_series(n: int = 80, seed: int = 1) -> np.ndarray:
    rng = np.random.RandomState(seed)
    e = rng.randn(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.6 * y[t - 1] + e[t]
    return y


class TestLocalLevelModel:
    """#3664 baseline -- the simplest model should fit cleanly."""

    def test_fit_runs(self) -> None:
        y = np.cumsum(np.random.RandomState(0).randn(50))
        result = LocalLevelModel().fit(y)
        assert result.fitted_values.shape == (50,)
        assert np.isfinite(result.log_likelihood)
        assert result.n_states == 1

    def test_fit_rejects_too_short(self) -> None:
        with pytest.raises(ValueError):
            LocalLevelModel().fit(np.array([1.0]))

    def test_fit_rejects_nan(self) -> None:
        with pytest.raises(ValueError):
            LocalLevelModel().fit(np.array([1.0, np.nan, 3.0]))


class TestSeasonalModelFit:
    """#3664 + #3697 — SeasonalModel fit path and positive-variance guard."""

    def test_fit_runs(self) -> None:
        y = _seasonal_series(period=4)
        cfg = StateSpaceConfig(
            model_type=StateSpaceModelType.SEASONAL, seasonal_period=4
        )
        result = SeasonalModel(cfg).fit(y)
        assert result.fitted_values.shape == y.shape
        assert np.isfinite(result.log_likelihood)
        assert np.isfinite(result.aic)

    def test_variances_non_negative_after_update(self) -> None:
        # #3697 — even with negative raw params, squared variances stay >= 0.
        cfg = StateSpaceConfig(
            model_type=StateSpaceModelType.SEASONAL, seasonal_period=4
        )
        model = SeasonalModel(cfg)
        model._initialize_matrices(_seasonal_series(period=4))
        model._update_matrices(np.array([-2.0, -3.0, -1.0, -0.5]))
        assert np.all(np.diag(model.Q) >= 0.0)
        assert model.H[0, 0] >= 0.0
        # And specifically the squares of the inputs.
        assert model.Q[0, 0] == pytest.approx(4.0)
        assert model.H[0, 0] == pytest.approx(0.25)

    def test_reported_parameters_are_variances(self) -> None:
        cfg = StateSpaceConfig(
            model_type=StateSpaceModelType.SEASONAL, seasonal_period=4
        )
        model = SeasonalModel(cfg)
        params = model._parameters_to_dict(np.array([2.0, 3.0, 1.0, 0.5]))
        assert params["sigma_level_sq"] == pytest.approx(4.0)
        assert params["sigma_obs_sq"] == pytest.approx(0.25)

    def test_fit_via_convenience(self) -> None:
        result = fit_state_space(
            _seasonal_series(period=4), model_type="seasonal", seasonal_period=4
        )
        assert result.model_type == StateSpaceModelType.SEASONAL


class TestARIMAStateSpaceFit:
    """#3664 + #3699 — ARIMA fit path and zero innovation covariance guard."""

    def test_fit_runs(self) -> None:
        cfg = StateSpaceConfig(
            model_type=StateSpaceModelType.ARIMA, ar_order=1, ma_order=0
        )
        result = ARIMAStateSpace(cfg).fit(_arma_series())
        assert result.fitted_values.shape == (80,)
        assert np.isfinite(result.log_likelihood)

    def test_fit_does_not_divide_by_zero_with_h_zero(self) -> None:
        # #3699 — pure ARIMA sets H=0; filter must stay finite.
        cfg = StateSpaceConfig(
            model_type=StateSpaceModelType.ARIMA, ar_order=2, ma_order=1
        )
        model = ARIMAStateSpace(cfg)
        model.fit(_arma_series())
        assert model.H[0, 0] == 0.0  # pure ARIMA, no observation noise
        filtered, cov, ll = model._kalman_filter(_arma_series())
        assert np.all(np.isfinite(filtered))
        assert np.isfinite(ll)

    def test_arima_with_ma(self) -> None:
        result = fit_state_space(
            _arma_series(), model_type="arima", ar_order=1, ma_order=1
        )
        assert "ar_1" in result.parameters
        assert "ma_1" in result.parameters
        assert "sigma_sq" in result.parameters


class TestNormalPpf:
    """#3698 — invalid p must raise, not collapse intervals to zero width."""

    def test_valid_p(self) -> None:
        model = LocalLevelModel()
        assert model._normal_ppf(0.975) == pytest.approx(1.96, abs=0.01)
        # The rational approximation is not exact at the median.
        assert model._normal_ppf(0.5) == pytest.approx(0.0, abs=1e-3)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_p_raises(self, bad: float) -> None:
        model = LocalLevelModel()
        with pytest.raises(ValueError):
            model._normal_ppf(bad)

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError):
            LocalLevelModel()._normal_ppf(None)  # type: ignore[arg-type]


class TestForecast:
    """#3664 — forecasting path with non-degenerate intervals (#3698)."""

    def test_forecast_after_fit(self) -> None:
        model = LocalLinearTrendModel()
        model.fit(np.cumsum(np.random.RandomState(2).randn(40)))
        fc = model.forecast(steps=5)
        assert isinstance(fc, ForecastResult)
        assert fc.forecast.shape == (5,)
        # Confidence intervals must have positive width (no zero collapse).
        assert np.all(fc.upper_ci > fc.lower_ci)

    def test_forecast_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            LocalLevelModel().forecast(steps=3)


class TestEMOptimization:
    def test_em_path_runs(self) -> None:
        cfg = StateSpaceConfig(
            model_type=StateSpaceModelType.LOCAL_LEVEL,
            optimization_method=OptimizationMethod.EM,
            max_iterations=20,
        )
        result = LocalLevelModel(cfg).fit(np.cumsum(np.random.RandomState(3).randn(40)))
        assert np.isfinite(result.log_likelihood)
