"""Tests for signal_toolkit.limits module.

Covers:
- SaturationMode enum
- apply_saturation (hard, tanh, sigmoid, atan, cubic)
- apply_rate_limiter
- apply_deadband
- apply_hysteresis
- apply_backlash
- create_saturation_function
- visualize_saturation_curves
"""

from __future__ import annotations

import numpy as np
import pytest
from signal_toolkit.core import Signal
from signal_toolkit.limits import (
    SaturationMode,
    apply_backlash,
    apply_deadband,
    apply_hysteresis,
    apply_rate_limiter,
    apply_saturation,
    create_saturation_function,
    visualize_saturation_curves,
)


@pytest.fixture()
def ramp_signal() -> Signal:
    """Create a ramp from -3 to 3."""
    t = np.linspace(0, 1, 500)
    values = np.linspace(-3, 3, 500)
    return Signal(time=t, values=values, name="ramp")


@pytest.fixture()
def sine_signal() -> Signal:
    t = np.linspace(0, 2 * np.pi, 1000)
    values = 2.0 * np.sin(t)  # amplitude 2, exceeds ±1
    return Signal(time=t, values=values, name="sine")


# ── SaturationMode ─────────────────────────────────────────────────────


class TestSaturationMode:
    def test_values(self) -> None:
        assert SaturationMode.HARD.value == "hard"
        assert SaturationMode.TANH.value == "tanh"
        assert SaturationMode.SIGMOID.value == "sigmoid"


# ── apply_saturation ───────────────────────────────────────────────────


class TestApplySaturation:
    def test_hard_clipping(self, ramp_signal: Signal) -> None:
        result = apply_saturation(ramp_signal, lower=-1.0, upper=1.0, mode=SaturationMode.HARD)
        assert np.max(result.values) <= 1.0 + 1e-10
        assert np.min(result.values) >= -1.0 - 1e-10

    def test_tanh_reduces_range(self, ramp_signal: Signal) -> None:
        """Tanh saturation should reduce the output range vs the input range."""
        result = apply_saturation(ramp_signal, lower=-1.0, upper=1.0, mode=SaturationMode.TANH)
        # Output range should be smaller than input range [-3, 3]
        assert np.max(result.values) < 3.0
        assert np.min(result.values) > -3.0

    def test_sigmoid_within_limits(self, ramp_signal: Signal) -> None:
        result = apply_saturation(ramp_signal, lower=-1.0, upper=1.0, mode=SaturationMode.SIGMOID)
        assert np.max(result.values) <= 1.0 + 1e-6
        assert np.min(result.values) >= -1.0 - 1e-6

    def test_atan_within_limits(self, ramp_signal: Signal) -> None:
        result = apply_saturation(ramp_signal, lower=-1.0, upper=1.0, mode=SaturationMode.ATAN)
        assert np.max(result.values) <= 1.0 + 1e-6
        assert np.min(result.values) >= -1.0 - 1e-6

    def test_output_is_signal(self, ramp_signal: Signal) -> None:
        result = apply_saturation(ramp_signal, -1.0, 1.0)
        assert isinstance(result, Signal)
        assert len(result.values) == len(ramp_signal.values)

    def test_inside_limits_unchanged(self) -> None:
        """Values within limits should be approximately unchanged for hard clip."""
        t = np.linspace(0, 1, 100)
        values = np.linspace(-0.5, 0.5, 100)
        sig = Signal(time=t, values=values, name="small")
        result = apply_saturation(sig, lower=-1.0, upper=1.0, mode=SaturationMode.HARD)
        np.testing.assert_allclose(result.values, values, atol=1e-10)


# ── apply_rate_limiter ─────────────────────────────────────────────────


class TestApplyRateLimiter:
    def test_limits_rate(self) -> None:
        """A step should be smoothed by the rate limiter."""
        t = np.linspace(0, 1, 1000)
        values = np.where(t < 0.5, 0.0, 10.0)
        sig = Signal(time=t, values=values, name="step")
        result = apply_rate_limiter(sig, max_rate=20.0)
        assert isinstance(result, Signal)
        # The output should not jump instantaneously
        dt = t[1] - t[0]
        rates = np.abs(np.diff(result.values) / dt)
        # Allow some tolerance for smoothing
        assert np.max(rates) < 25.0  # close to 20 but smoothed

    def test_slow_signal_unchanged(self) -> None:
        """A slowly changing signal should pass through nearly unchanged."""
        t = np.linspace(0, 10, 1000)
        values = 0.1 * t  # rate = 0.1
        sig = Signal(time=t, values=values, name="slow")
        result = apply_rate_limiter(sig, max_rate=100.0)
        np.testing.assert_allclose(result.values, values, atol=0.5)


# ── apply_deadband ─────────────────────────────────────────────────────


class TestApplyDeadband:
    def test_small_signals_zeroed(self) -> None:
        """Signals within the deadband should be near zero."""
        t = np.linspace(0, 1, 100)
        values = np.linspace(-0.05, 0.05, 100)
        sig = Signal(time=t, values=values, name="tiny")
        result = apply_deadband(sig, threshold=0.1, smooth=False)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-6)

    def test_large_signals_pass(self) -> None:
        """Signals well outside the deadband should pass through."""
        t = np.linspace(0, 1, 100)
        values = np.full(100, 5.0)
        sig = Signal(time=t, values=values, name="large")
        result = apply_deadband(sig, threshold=0.1, smooth=False)
        assert np.all(result.values > 4.0)


# ── apply_hysteresis ───────────────────────────────────────────────────


class TestApplyHysteresis:
    def test_binary_output(self) -> None:
        """Output should only be output_high or output_low without smoothing."""
        t = np.linspace(0, 2 * np.pi, 1000)
        values = np.sin(t)
        sig = Signal(time=t, values=values, name="sine")
        result = apply_hysteresis(
            sig,
            threshold_up=0.5,
            threshold_down=-0.5,
            output_high=1.0,
            output_low=0.0,
            smooth=False,
        )
        unique = set(np.unique(result.values))
        assert unique <= {0.0, 1.0}


# ── apply_backlash ─────────────────────────────────────────────────────


class TestApplyBacklash:
    def test_output_type(self) -> None:
        t = np.linspace(0, 2 * np.pi, 500)
        values = np.sin(t)
        sig = Signal(time=t, values=values, name="sine")
        result = apply_backlash(sig, backlash_width=0.2, smooth=False)
        assert isinstance(result, Signal)
        assert len(result.values) == len(t)


# ── create_saturation_function ─────────────────────────────────────────


class TestCreateSaturationFunction:
    def test_callable_returned(self) -> None:
        func = create_saturation_function(-1.0, 1.0, SaturationMode.TANH)
        assert callable(func)

    def test_applies_saturation(self) -> None:
        func = create_saturation_function(-1.0, 1.0, SaturationMode.HARD)
        values = np.array([-5.0, -0.5, 0.0, 0.5, 5.0])
        result = func(values)
        assert np.max(result) <= 1.0 + 1e-10
        assert np.min(result) >= -1.0 - 1e-10


# ── visualize_saturation_curves ────────────────────────────────────────


class TestVisualizeSaturationCurves:
    def test_returns_dict(self) -> None:
        curves = visualize_saturation_curves()
        assert isinstance(curves, dict)
        assert len(curves) > 0

    def test_each_curve_is_tuple_of_arrays(self) -> None:
        curves = visualize_saturation_curves(num_points=100)
        for _mode_name, (inp, out) in curves.items():
            assert len(inp) == 100
            assert len(out) == 100
