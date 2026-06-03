"""Focused coverage for signal_toolkit adaptive filters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from signal_toolkit import adaptive_filter
from signal_toolkit.adaptive_filter import AdaptiveFilter
from signal_toolkit.core import Signal


def _signal(values: np.ndarray, *, name: str = "sensor", units: str = "V") -> Signal:
    return Signal(
        time=np.arange(values.size, dtype=np.float64),
        values=values,
        name=name,
        units=units,
    )


def _expected_lms(
    x: np.ndarray,
    d: np.ndarray,
    *,
    order: int,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.zeros(order)
    filtered = np.zeros(x.size)
    error = np.zeros(x.size)
    for index in range(order, x.size):
        window = x[index - order : index][::-1]
        filtered[index] = np.dot(weights, window)
        error[index] = d[index] - filtered[index]
        weights += step_size * error[index] * window
    return filtered, error


def _expected_rls(
    x: np.ndarray,
    d: np.ndarray,
    *,
    order: int,
    forgetting_factor: float,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.zeros(order)
    inverse_correlation = np.eye(order) / delta
    filtered = np.zeros(x.size)
    error = np.zeros(x.size)
    for index in range(order, x.size):
        window = x[index - order : index][::-1].reshape(-1, 1)
        filtered[index] = np.dot(weights, window.flatten())
        error[index] = d[index] - filtered[index]
        gain = (
            inverse_correlation
            @ window
            / (forgetting_factor + window.T @ inverse_correlation @ window)
        )
        inverse_correlation = (
            inverse_correlation - gain @ window.T @ inverse_correlation
        ) / forgetting_factor
        weights += gain.flatten() * error[index]
    return filtered, error


def test_lms_numpy_fallback_matches_reference_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adaptive_filter, "_rust_lms", None)
    x = np.array([0.2, -0.1, 0.3, 0.7, -0.4, 0.9, 0.1, -0.2], dtype=np.float64)
    desired = np.array([0.0, 0.1, 0.2, 0.5, -0.1, 0.8, 0.3, -0.4], dtype=np.float64)
    signal = _signal(x, name="accelerometer", units="m/s^2")
    reference = _signal(desired)

    filtered, error = AdaptiveFilter.lms(signal, reference, order=3, step_size=0.05)

    expected_filtered, expected_error = _expected_lms(
        x, desired, order=3, step_size=0.05
    )
    np.testing.assert_allclose(filtered.values, expected_filtered)
    np.testing.assert_allclose(error.values, expected_error)
    assert filtered.name == "accelerometer_lms"
    assert error.name == "accelerometer_lms_error"
    assert filtered.units == error.units == "m/s^2"
    np.testing.assert_array_equal(filtered.time, signal.time)


def test_rls_numpy_fallback_matches_reference_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adaptive_filter, "_rust_rls", None)
    x = np.array([0.4, 0.2, -0.3, 0.8, 0.1, -0.5, 0.6, 0.9], dtype=np.float64)
    desired = np.array([0.2, 0.0, -0.2, 0.6, 0.3, -0.4, 0.5, 0.7], dtype=np.float64)
    signal = _signal(x, name="pressure", units="Pa")
    reference = _signal(desired)

    filtered, error = AdaptiveFilter.rls(
        signal,
        reference,
        order=3,
        forgetting_factor=0.97,
        delta=0.2,
    )

    expected_filtered, expected_error = _expected_rls(
        x,
        desired,
        order=3,
        forgetting_factor=0.97,
        delta=0.2,
    )
    np.testing.assert_allclose(filtered.values, expected_filtered)
    np.testing.assert_allclose(error.values, expected_error)
    assert filtered.name == "pressure_rls"
    assert error.name == "pressure_rls_error"
    assert filtered.units == error.units == "Pa"
    np.testing.assert_array_equal(error.time, signal.time)


def test_lms_uses_rust_kernel_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []

    def rust_lms(
        x: np.ndarray, d: np.ndarray, **kwargs: Any
    ) -> tuple[list[float], list[float]]:
        calls.append((x, d, kwargs))
        return [1.0, 2.0, 3.0], [0.3, 0.2, 0.1]

    monkeypatch.setattr(adaptive_filter, "_rust_lms", rust_lms)
    signal = _signal(np.array([1, 2, 3]), name="native")
    reference = _signal(np.array([2, 4, 6]))

    filtered, error = AdaptiveFilter.lms(signal, reference, order=2, step_size=0.125)

    assert calls[0][0].dtype == np.float64
    assert calls[0][1].dtype == np.float64
    assert calls[0][2] == {"order": 2, "step_size": 0.125}
    np.testing.assert_array_equal(filtered.values, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(error.values, np.array([0.3, 0.2, 0.1]))


def test_rls_uses_rust_kernel_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []

    def rust_rls(
        x: np.ndarray, d: np.ndarray, **kwargs: Any
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        calls.append((x, d, kwargs))
        return (1.5, 2.5, 3.5), (0.5, 0.25, 0.125)

    monkeypatch.setattr(adaptive_filter, "_rust_rls", rust_rls)
    signal = _signal(np.array([3, 2, 1]), name="native")
    reference = _signal(np.array([6, 4, 2]))

    filtered, error = AdaptiveFilter.rls(
        signal,
        reference,
        order=2,
        forgetting_factor=0.95,
        delta=0.5,
    )

    assert calls[0][0].dtype == np.float64
    assert calls[0][1].dtype == np.float64
    assert calls[0][2] == {"order": 2, "forgetting_factor": 0.95, "delta": 0.5}
    np.testing.assert_array_equal(filtered.values, np.array([1.5, 2.5, 3.5]))
    np.testing.assert_array_equal(error.values, np.array([0.5, 0.25, 0.125]))


@pytest.mark.parametrize(
    "method",
    [
        AdaptiveFilter.lms,
        AdaptiveFilter.rls,
    ],
)
def test_adaptive_filters_require_signal(
    method: Callable[..., tuple[Signal, Signal]],
) -> None:
    reference = _signal(np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="signal must be provided"):
        method(None, reference)
