# ruff: noqa: E501
"""Tests for the simulation module.

Focuses on integration accuracy, energy conservation (for
undriven systems), and correct handling of polynomial torques.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from double_pendulum_golf.physics import PendulumParams, total_energy
from double_pendulum_golf.simulation import (
    SimulationResult,
    make_polynomial_torque,
    run_simulation,
)

# ======================================================================
# Polynomial torque builder
# ======================================================================


class TestPolynomialTorque:
    """Verify polynomial torque function construction."""

    def test_constant_torque(self) -> None:
        func = make_polynomial_torque([5.0], [0.0])
        assert func(0.0) == (5.0, 0.0)
        assert func(10.0) == (5.0, 0.0)

    def test_linear_torque(self) -> None:
        # tau(t) = 2 + 3*t
        func = make_polynomial_torque([2.0, 3.0], [0.0])
        tau1, tau2 = func(1.0)
        assert np.isclose(tau1, 5.0)  # 2 + 3*1
        assert np.isclose(tau2, 0.0)

    def test_quadratic_torque(self) -> None:
        # tau(t) = 1 + 0*t + 2*t^2  =>  tau(3) = 1 + 18 = 19
        func = make_polynomial_torque([1.0, 0.0, 2.0], [0.0])
        tau1, _ = func(3.0)
        assert np.isclose(tau1, 19.0)

    def test_empty_coefficients_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            make_polynomial_torque([], [1.0])


# ======================================================================
# Simulation execution
# ======================================================================


class TestSimulationBasics:
    """Basic simulation sanity checks."""

    def test_produces_result(
        self,
        default_params: PendulumParams,
        zero_torque: Callable[[float], tuple[float, float]],
        aligned_state: np.ndarray,
    ) -> None:
        result = run_simulation(
            default_params,
            aligned_state,
            t_end=0.5,
            torque_func=zero_torque,
            dt=0.01,
        )
        assert isinstance(result, SimulationResult)
        assert result.n_steps > 10

    def test_initial_state_preserved(
        self,
        default_params: PendulumParams,
        zero_torque: Callable[[float], tuple[float, float]],
        cocked_state: np.ndarray,
    ) -> None:
        result = run_simulation(
            default_params,
            cocked_state,
            t_end=0.1,
            torque_func=zero_torque,
            dt=0.01,
        )
        assert np.allclose(result.states[0], cocked_state, atol=1e-6)

    def test_time_monotonically_increases(
        self,
        default_params: PendulumParams,
        zero_torque: Callable[[float], tuple[float, float]],
        aligned_state: np.ndarray,
    ) -> None:
        result = run_simulation(
            default_params,
            aligned_state,
            t_end=1.0,
            torque_func=zero_torque,
            dt=0.01,
        )
        assert all(np.diff(result.t) > 0)

    def test_native_backend_integration(self, default_params: PendulumParams) -> None:
        import unittest.mock as mock

        with (
            mock.patch(
                "double_pendulum_golf.simulation.double_native_enabled",
                return_value=True,
            ),
            mock.patch("double_pendulum_golf.simulation.simulate_double") as mock_sim,
        ):
            # Setup mock to return a coarse trajectory
            # Use 3 points to ensure interpolation is triggered
            t_res = np.array([0.0, 0.5, 1.0])
            states_res = np.zeros((3, 4))
            mock_sim.return_value = (t_res, states_res)

            result = run_simulation(
                default_params,
                np.array([0.0, 0.0, 0.0, 0.0]),
                t_end=1.0,
                torque_func=lambda t: (0.0, 0.0),
                dt=0.1,  # 10 steps
                coeffs=[0.0] * 10,
                n_coeffs_per_joint=5,
            )
            assert mock_sim.called
            # The result should be interpolated to dt=0.1, so length is 10 (from 0.0 to 0.9)
            assert len(result.t) == 10
            assert np.isclose(result.t[1] - result.t[0], 0.1)

    def test_native_backend_too_few_points(self, default_params: PendulumParams) -> None:
        import unittest.mock as mock

        with (
            mock.patch(
                "double_pendulum_golf.simulation.double_native_enabled",
                return_value=True,
            ),
            mock.patch(
                "double_pendulum_golf.simulation.simulate_double",
                return_value=(np.array([0.0]), np.zeros((1, 4))),
            ) as mock_sim,
        ):
            # If native returns < 2 points, it falls back to Python ODE solver
            result = run_simulation(
                default_params,
                np.array([0.0, 0.0, 0.0, 0.0]),
                t_end=0.5,
                torque_func=lambda t: (0.0, 0.0),
                dt=0.1,
                coeffs=[0.0] * 10,
                n_coeffs_per_joint=5,
            )
            assert mock_sim.called
            # Fell back, so length should be based on integration
            assert len(result.t) >= 5


class TestSimulationResultContracts:
    """Trajectory-level DbC validation for SimulationResult."""

    def test_constructor_rejects_non_monotonic_time(
        self,
        default_params: PendulumParams,
    ) -> None:
        with pytest.raises((ValueError, TypeError), match="strictly increasing"):
            SimulationResult(
                t=np.array([0.0, 0.0]),
                states=np.zeros((2, 4)),
                params=default_params,
                torque_func=lambda _t: (0.0, 0.0),
            )

    def test_constructor_rejects_wrong_state_width(
        self,
        default_params: PendulumParams,
    ) -> None:
        with pytest.raises((ValueError, TypeError), match="states must have width 4"):
            SimulationResult(
                t=np.array([0.0, 0.1]),
                states=np.zeros((2, 3)),
                params=default_params,
                torque_func=lambda _t: (0.0, 0.0),
            )


class TestEnergyConservation:
    """For an undriven system (zero torque), total energy must be conserved."""

    def test_energy_conserved_free_pendulum(
        self,
        equal_params: PendulumParams,
        zero_torque: Callable[[float], tuple[float, float]],
    ) -> None:
        """Energy drift should be < 0.1% over 2 seconds."""
        state0 = np.array([np.radians(45), np.radians(30), 0.0, 0.0])
        result = run_simulation(
            equal_params,
            state0,
            t_end=2.0,
            torque_func=zero_torque,
            dt=0.005,
        )
        E0 = total_energy(result.states[0], equal_params)
        energies = np.array(
            [total_energy(result.states[i], equal_params) for i in range(result.n_steps)]
        )
        max_drift = np.max(np.abs(energies - E0))
        relative_drift = max_drift / abs(E0) if abs(E0) > 1e-10 else max_drift
        assert relative_drift < 1e-3, (
            f"Energy drift {relative_drift:.2e} exceeds 0.1% threshold"
        )


class TestSimulationAccessors:
    """Test the SimulationResult data access methods."""

    def test_mass_matrix_at(
        self,
        default_params: PendulumParams,
        zero_torque: Callable[[float], tuple[float, float]],
        aligned_state: np.ndarray,
    ) -> None:
        result = run_simulation(
            default_params,
            aligned_state,
            t_end=0.1,
            torque_func=zero_torque,
            dt=0.01,
        )
        mc = result.mass_matrix_at(0)
        assert "M11" in mc
        assert "M12" in mc
        assert "M22" in mc
        assert mc["M12"] == mc["M21"]  # symmetry

    def test_positions_at(
        self,
        default_params: PendulumParams,
        zero_torque: Callable[[float], tuple[float, float]],
        aligned_state: np.ndarray,
    ) -> None:
        result = run_simulation(
            default_params,
            aligned_state,
            t_end=0.1,
            torque_func=zero_torque,
            dt=0.01,
        )
        pos = result.positions_at(0)
        assert "shoulder" in pos
        assert "wrist" in pos
        assert "tip" in pos

    def test_all_positions_and_energies(
        self,
        default_params: PendulumParams,
        zero_torque: Callable[[float], tuple[float, float]],
        aligned_state: np.ndarray,
    ) -> None:
        result = run_simulation(
            default_params,
            aligned_state,
            t_end=0.1,
            torque_func=zero_torque,
            dt=0.01,
        )
        positions = result.all_positions()
        energies = result.all_energies()
        assert len(positions) == result.n_steps
        assert energies["kinetic"].shape == (result.n_steps,)
        assert energies["potential"].shape == (result.n_steps,)
        assert energies["total"].shape == (result.n_steps,)

    def test_out_of_range_index_rejected(
        self,
        default_params: PendulumParams,
        zero_torque: Callable[[float], tuple[float, float]],
        aligned_state: np.ndarray,
    ) -> None:
        result = run_simulation(
            default_params,
            aligned_state,
            t_end=0.1,
            torque_func=zero_torque,
            dt=0.01,
        )
        with pytest.raises((ValueError, TypeError)):
            result.mass_matrix_at(result.n_steps + 10)


class TestSimulationDbC:
    """Contract violations for simulation inputs."""

    def test_negative_duration_rejected(
        self,
        default_params: PendulumParams,
        zero_torque: Callable[[float], tuple[float, float]],
        aligned_state: np.ndarray,
    ) -> None:
        with pytest.raises((ValueError, TypeError)):
            run_simulation(
                default_params,
                aligned_state,
                t_end=-1.0,
                torque_func=zero_torque,
            )

    def test_wrong_state_shape_rejected(
        self,
        default_params: PendulumParams,
        zero_torque: Callable[[float], tuple[float, float]],
    ) -> None:
        with pytest.raises((ValueError, TypeError)):
            run_simulation(
                default_params,
                np.array([0, 0]),
                t_end=1.0,
                torque_func=zero_torque,
            )
