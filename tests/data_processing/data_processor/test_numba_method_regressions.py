"""Regression tests for Numba decorators on object-oriented estimators."""

from __future__ import annotations

import inspect

import numpy as np

try:
    from numba.core.registry import CPUDispatcher
except (ImportError, AttributeError, ModuleNotFoundError):
    _CPU_DISPATCHER_TYPES: tuple[type[object], ...] = ()
else:
    _CPU_DISPATCHER_TYPES = (CPUDispatcher,)


def _assert_plain_methods(owner: type, method_names: tuple[str, ...]) -> None:
    for method_name in method_names:
        method = owner.__dict__[method_name]
        assert inspect.isfunction(method)
        assert not isinstance(method, _CPU_DISPATCHER_TYPES)


def test_kalman_filter_module_imports_and_public_filters_run() -> None:
    from data_processor.core.kalman_filter import (
        ExtendedKalmanFilter,
        KalmanFilter,
        KalmanFilterConfig,
        UnscentedKalmanFilter,
    )

    _assert_plain_methods(KalmanFilter, ("filter",))
    _assert_plain_methods(ExtendedKalmanFilter, ("filter",))
    _assert_plain_methods(UnscentedKalmanFilter, ("_sigma_points", "filter"))

    measurements = np.linspace(0.0, 1.0, 8)
    result = KalmanFilter(KalmanFilterConfig(state_dim=1, obs_dim=1)).filter(
        measurements
    )
    assert result.filtered_states.shape == (8, 1)

    def transition(state: np.ndarray, control: np.ndarray | None = None) -> np.ndarray:
        return state if control is None else state + control

    def observation(state: np.ndarray) -> np.ndarray:
        return state

    ekf = ExtendedKalmanFilter(state_dim=1, obs_dim=1)
    ekf_result = ekf.filter(
        measurements,
        transition_func=transition,
        observation_func=observation,
    )
    assert ekf_result.filtered_states.shape == (8, 1)

    ukf = UnscentedKalmanFilter(
        state_dim=1,
        measurement_dim=1,
        f=transition,
        h=observation,
        Q=np.array([[0.01]]),
        R=np.array([[0.1]]),
    )
    ukf_result = ukf.filter(measurements)
    assert ukf_result.filtered_states.shape == (8, 1)


def test_state_space_instance_methods_are_plain_python_and_runtime_paths_work() -> None:
    from data_processor.core.state_space import (
        ARIMAStateSpace,
        BaseStateSpaceModel,
        LocalLevelModel,
        SeasonalModel,
        StateSpaceConfig,
    )

    _assert_plain_methods(
        BaseStateSpaceModel,
        (
            "forecast",
            "_kalman_filter",
            "_kalman_smoother",
            "_em_m_step",
            "_numerical_gradient",
        ),
    )
    _assert_plain_methods(SeasonalModel, ("_initialize_matrices",))
    _assert_plain_methods(ARIMAStateSpace, ("_update_matrices", "_estimate_ar"))

    observations = np.cumsum(np.linspace(-0.2, 0.3, 24))
    model = LocalLevelModel(StateSpaceConfig(max_iterations=5))
    fit_result = model.fit(observations)
    forecast = model.forecast(steps=5)

    assert fit_result.fitted_values.shape == observations.shape
    assert forecast.forecast.shape == (5,)
    assert forecast.lower_ci.shape == (5,)
    assert forecast.upper_ci.shape == (5,)
