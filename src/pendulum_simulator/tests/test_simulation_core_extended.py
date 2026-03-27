from typing import Any

"""Extended tests for simulation_core.py — covering remaining code paths.

Adds tests for:
- max_step parameter usage
- RuntimeError on integration failure
- Different ODE methods (DOP853)
- Infimum time point assertion (at least 2 points)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from double_pendulum_golf.simulation_core import integrate_ode


class TestIntegrateOdeExtended:
    """Additional coverage for previously uncovered paths in integrate_ode."""

    def test_max_step_parameter_is_used(self) -> None:
        """Passing max_step should include it in solve_ivp kwargs."""

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            return -y

        # Should not raise — max_step overrides adaptive step size selection
        t, states = integrate_ode(
            rhs,
            np.array([1.0]),
            t_end=0.5,
            dt=0.01,
            max_step=0.01,
        )
        assert len(t) >= 2
        assert states.shape[1] == 1

    def test_max_step_none_does_not_set_in_kwargs(self) -> None:
        """When max_step=None (default), solve_ivp is called without max_step kwarg."""
        import scipy.integrate as sci

        calls = []
        orig = sci.solve_ivp

        def capturing_solve_ivp(*args, **kwargs) -> Any:
            calls.append(kwargs)
            return orig(*args, **kwargs)

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            return -y

        with patch(
            "double_pendulum_golf.simulation_core.solve_ivp",
            side_effect=capturing_solve_ivp,
        ):
            integrate_ode(rhs, np.array([1.0]), t_end=0.1, dt=0.01)

        assert len(calls) == 1
        assert "max_step" not in calls[0]

    def test_max_step_provided_is_forwarded(self) -> None:
        """When max_step is specified, solve_ivp should receive it."""
        import scipy.integrate as sci

        calls = []
        orig = sci.solve_ivp

        def capturing_solve_ivp(*args, **kwargs) -> Any:
            calls.append(kwargs)
            return orig(*args, **kwargs)

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            return -y

        with patch(
            "double_pendulum_golf.simulation_core.solve_ivp",
            side_effect=capturing_solve_ivp,
        ):
            integrate_ode(rhs, np.array([1.0]), t_end=0.1, dt=0.01, max_step=0.005)

        assert len(calls) == 1
        assert "max_step" in calls[0]
        assert calls[0]["max_step"] == 0.005

    def test_runtime_error_on_failed_integration(self) -> None:
        """If solve_ivp returns sol.success=False, RuntimeError must be raised."""
        failed_sol = MagicMock()
        failed_sol.success = False
        failed_sol.message = "Integration failed (test)"

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            return -y

        with patch(
            "double_pendulum_golf.simulation_core.solve_ivp",
            return_value=failed_sol,
        ):
            with pytest.raises(RuntimeError, match="Integration failed"):
                integrate_ode(rhs, np.array([1.0]), t_end=1.0, dt=0.1)

    def test_dop853_method(self) -> None:
        """DOP853 method should work for a simple harmonic oscillator."""

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            return np.array([y[1], -y[0]])

        t, states = integrate_ode(
            rhs,
            np.array([1.0, 0.0]),
            t_end=np.pi,
            dt=0.02,
            method="DOP853",
        )
        # After π seconds, position should be ≈ -1
        np.testing.assert_allclose(states[-1, 0], -1.0, atol=0.05)

    def test_output_states_are_finite(self) -> None:
        """All output states must be finite."""

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            return -y

        t, states = integrate_ode(rhs, np.array([1.0, 2.0]), t_end=1.0, dt=0.05)
        assert np.all(np.isfinite(t))
        assert np.all(np.isfinite(states))

    def test_multi_dimensional_rhs(self) -> None:
        """Four-dimensional ODE should work correctly."""

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            return -0.5 * y

        y0 = np.array([1.0, 2.0, 3.0, 4.0])
        t, states = integrate_ode(rhs, y0, t_end=2.0, dt=0.1)
        assert states.shape[1] == 4
        # Closed-form: y(t) = y0 * exp(-0.5*t)
        expected_final = y0 * np.exp(-0.5 * t[-1])
        np.testing.assert_allclose(states[-1], expected_final, rtol=0.01)
