"""Tests for upstream_drift_tools.process_calculators.ode_solver module.

Covers:
- ODESolver instantiation
- solve() for simple first-order ODE
- Multi-variable ODE system
- Parameter substitution
"""

from __future__ import annotations

import numpy as np
import pytest
from upstream_drift_tools.process_calculators.ode_solver import ODESolver


class TestODESolverInit:
    def test_single_variable(self) -> None:
        solver = ODESolver(
            derivatives={"T": "k*(T_env - T)"},
            parameters={"k": 0.3, "T_env": 350.0},
        )
        assert len(solver.var_syms) == 1
        assert len(solver.param_syms) == 2

    def test_multi_variable(self) -> None:
        solver = ODESolver(
            derivatives={"x": "v", "v": "-k*x"},
            parameters={"k": 1.0},
        )
        assert len(solver.var_syms) == 2


class TestODESolverSolve:
    def test_heating_convergence(self) -> None:
        """Newton's law of cooling: dT/dt = k*(T_env - T).

        Starting at T=300, with T_env=350 and k=0.3, the temperature
        should approach 350 exponentially.
        """
        solver = ODESolver(
            derivatives={"T": "k*(T_env - T)"},
            parameters={"k": 0.3, "T_env": 350.0},
        )
        t_eval = np.linspace(0.0, 30.0, 200)
        sol = solver.solve((0.0, 30.0), [300.0], t_eval=t_eval, rtol=1e-8, atol=1e-10)
        # After long time, should be near T_env
        assert sol.y[0][-1] == pytest.approx(350.0, abs=0.1)
        # Should be above initial
        assert sol.y[0][-1] > 300.0

    def test_exponential_decay(self) -> None:
        """dy/dt = -y has solution y = y₀ * exp(-t)."""
        solver = ODESolver(
            derivatives={"y": "-y"},
            parameters={},
        )
        t_eval = np.linspace(0.0, 5.0, 100)
        sol = solver.solve((0.0, 5.0), [1.0], t_eval=t_eval, rtol=1e-8, atol=1e-10)
        expected = np.exp(-t_eval)
        np.testing.assert_allclose(sol.y[0], expected, rtol=1e-3)

    def test_harmonic_oscillator(self) -> None:
        """dx/dt = v, dv/dt = -x. Solution is x = cos(t), v = -sin(t)."""
        solver = ODESolver(
            derivatives={"x": "v", "v": "-x"},
            parameters={},
        )
        t_eval = np.linspace(0.0, 2 * np.pi, 200)
        sol = solver.solve((0.0, 2 * np.pi), [1.0, 0.0], t_eval=t_eval)
        # x should return near 1.0 after one period
        assert sol.y[0][-1] == pytest.approx(1.0, abs=0.01)
        # v should return near 0.0 after one period
        assert sol.y[1][-1] == pytest.approx(0.0, abs=0.05)

    def test_solution_shape(self) -> None:
        solver = ODESolver(
            derivatives={"y": "-y"},
            parameters={},
        )
        t_eval = np.linspace(0.0, 1.0, 50)
        sol = solver.solve((0.0, 1.0), [1.0], t_eval=t_eval)
        assert len(sol.t) == 50
        assert sol.y.shape[0] == 1  # 1 variable
        assert sol.y.shape[1] == 50
