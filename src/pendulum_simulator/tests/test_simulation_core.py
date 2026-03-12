"""Tests for the shared ODE integration core."""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.simulation_core import integrate_ode


class TestIntegrateOde:
    def test_simple_exponential_decay(self) -> None:
        """y' = -y, y(0) = 1 => y(t) = e^(-t)"""

        def rhs(t, y):
            return -y

        t, states = integrate_ode(rhs, np.array([1.0]), t_end=2.0, dt=0.01)
        assert len(t) >= 2
        assert states.shape[1] == 1
        # y(2) ≈ e^(-2) ≈ 0.1353
        np.testing.assert_allclose(states[-1, 0], np.exp(-2.0), rtol=0.011)

    def test_harmonic_oscillator(self) -> None:
        """x'' + x = 0 => x(t) = cos(t), v(t) = -sin(t)"""

        def rhs(t, y):
            return np.array([y[1], -y[0]])

        t, states = integrate_ode(
            rhs,
            np.array([1.0, 0.0]),
            t_end=np.pi,
            dt=0.01,
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
        )
        # x(pi) ≈ -1, v(pi) ≈ 0
        np.testing.assert_allclose(states[-1, 0], -1.0, atol=1e-2)
        np.testing.assert_allclose(states[-1, 1], 0.0, atol=1e-2)

    def test_rejects_bad_inputs(self) -> None:
        def rhs(t, y):
            return -y

        with pytest.raises(AssertionError):
            integrate_ode(rhs, np.array([float("nan")]), t_end=1.0)
        with pytest.raises(AssertionError):
            integrate_ode(rhs, np.array([1.0]), t_end=-1.0)
        with pytest.raises(AssertionError):
            integrate_ode(rhs, np.array([1.0]), t_end=1.0, dt=2.0)

    def test_returns_correct_shapes(self) -> None:
        def rhs(t, y):
            return -y

        t, states = integrate_ode(rhs, np.array([1.0, 2.0, 3.0]), t_end=1.0, dt=0.1)
        assert t.ndim == 1
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert len(t) == states.shape[0]
