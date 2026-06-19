"""Regression tests for methods that must not be numba-dispatched.

Numba nopython mode cannot compile bound methods that depend on ``self``,
Python callables, dictionaries, dataclasses, or instance mutation. These tests
keep those object-oriented orchestration paths as plain Python functions.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest


@pytest.mark.parametrize(
    ("class_path", "method_names"),
    [
        (
            "data_processor.core.uncertainty_quantification.UncertaintyQuantifier",
            (
                "error_propagation",
                "sensitivity_analysis",
                "prediction_intervals",
                "delta_method_ci",
                "_studentized_interval",
                "_sobol_sample",
            ),
        ),
        (
            "data_processor.core.cross_correlation.CrossCorrelationAnalyzer",
            (
                "rolling_cross_correlation",
                "_compute_pvalues",
                "_select_lag_order",
                "_create_lag_matrix",
                "_conditional_entropy",
            ),
        ),
        (
            "data_processor.core.kalman_filter.KalmanFilter",
            ("filter",),
        ),
        (
            "data_processor.core.kalman_filter.ExtendedKalmanFilter",
            ("filter",),
        ),
        (
            "data_processor.core.kalman_filter.UnscentedKalmanFilter",
            ("_sigma_points", "filter"),
        ),
        (
            "data_processor.core.state_space.BaseStateSpaceModel",
            (
                "forecast",
                "_kalman_filter",
                "_kalman_smoother",
                "_em_m_step",
                "_numerical_gradient",
            ),
        ),
        (
            "data_processor.core.state_space.SeasonalModel",
            ("_initialize_matrices",),
        ),
        (
            "data_processor.core.state_space.ARIMAStateSpace",
            ("_update_matrices", "_estimate_ar"),
        ),
        (
            "data_processor.core.dataset_manager.DatasetManager",
            ("save_workspace", "load_workspace"),
        ),
    ],
)
def test_object_oriented_analysis_methods_are_plain_functions(
    class_path: str, method_names: tuple[str, ...]
) -> None:
    """Object-heavy methods should remain Python functions, not CPUDispatchers."""
    module_name, class_name = class_path.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)

    for method_name in method_names:
        method = getattr(cls, method_name)
        assert inspect.isfunction(method), (
            f"{class_path}.{method_name} must be a plain Python function; "
            f"got {type(method)!r}"
        )


def test_error_propagation_accepts_callable_and_dict_inputs() -> None:
    """The callable/dict path exercises the former duplicate-jit failure."""
    from data_processor.core.uncertainty_quantification import UncertaintyQuantifier

    def area(length: float, width: float) -> float:
        return length * width

    value, uncertainty = UncertaintyQuantifier().error_propagation(
        area,
        {"length": 4.0, "width": 3.0},
        {"length": 0.2, "width": 0.1},
    )

    assert value == pytest.approx(12.0)
    assert uncertainty == pytest.approx(np.sqrt((3.0 * 0.2) ** 2 + (4.0 * 0.1) ** 2))


def test_cross_correlation_object_methods_execute_without_numba_dispatch() -> None:
    """Rolling correlation and p-values call instance helpers and stay executable."""
    from data_processor.core.cross_correlation import CrossCorrelationAnalyzer

    analyzer = CrossCorrelationAnalyzer()
    x = np.linspace(-1.0, 1.0, 40)
    y = x + 0.05

    rolling = analyzer.rolling_cross_correlation(x, y, window=8)
    p_values = analyzer._compute_pvalues(np.array([0.0, 0.5]), n=20)

    assert rolling.correlations.shape == x.shape
    assert p_values.shape == (2,)
    assert np.all((0.0 <= p_values) & (p_values <= 1.0))


def test_two_way_anova_sum_of_squares_is_plain_function() -> None:
    """Pandas-backed ANOVA helper uses closures and must not be numba-dispatched."""
    from data_processor.core.anova_two_way import two_way_sum_of_squares

    assert inspect.isfunction(two_way_sum_of_squares)


def test_kalman_filter_variants_execute_without_numba_dispatch() -> None:
    """KF/EKF/UKF filters use Python callables and mutable object state."""
    from data_processor.core.kalman_filter import (
        ExtendedKalmanFilter,
        KalmanFilter,
        KalmanFilterConfig,
        UnscentedKalmanFilter,
    )

    measurements = np.array([0.0, 0.1, 0.2, 0.3])

    kf = KalmanFilter(KalmanFilterConfig(state_dim=1, obs_dim=1))
    kf_result = kf.filter(measurements)

    ekf = ExtendedKalmanFilter(state_dim=1, obs_dim=1)
    ekf_result = ekf.filter(
        measurements,
        transition_func=lambda state: state,
        observation_func=lambda state: state,
    )

    ukf = UnscentedKalmanFilter(
        state_dim=1,
        measurement_dim=1,
        f=lambda state, _control: state,
        h=lambda state: state,
        Q=np.eye(1) * 0.01,
        R=np.eye(1) * 0.1,
    )
    ukf_result = ukf.filter(measurements)

    assert kf_result.filtered_states.shape == (4, 1)
    assert ekf_result.filtered_states.shape == (4, 1)
    assert ukf_result.filtered_states.shape == (4, 1)


def test_state_space_forecast_and_arima_helpers_execute_without_numba_dispatch() -> (
    None
):
    """State-space methods mutate instance matrices and should remain Python."""
    from data_processor.core.state_space import ARIMAStateSpace, LocalLevelModel

    data = np.linspace(0.0, 1.0, 30)

    model = LocalLevelModel()
    model.fit(data)
    forecast = model.forecast(steps=3)

    arima = ARIMAStateSpace()
    arima._initialize_matrices(data)
    parameters = arima._get_initial_parameters()
    arima._update_matrices(parameters)

    assert forecast.forecast.shape == (3,)
    assert arima._estimate_ar(data, 1).shape == (1,)
