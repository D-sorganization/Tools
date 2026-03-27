"""Tests for the triple pendulum simulation module."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from double_pendulum_golf.physics_triple import TriplePendulumParams, total_energy
from double_pendulum_golf.simulation_triple import (
    TripleSimulationResult,
    make_polynomial_torque,
    run_simulation,
)


@pytest.fixture
def triple_params() -> TriplePendulumParams:
    return TriplePendulumParams(m1=5.0, m2=0.5, m3=0.2, L1=0.6, L2=0.6, L3=0.6)


@pytest.fixture
def zero_torque() -> Callable[[float], tuple[float, float, float]]:
    return lambda t: (0.0, 0.0, 0.0)


@pytest.fixture
def aligned_state() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0, 0.5, 0.2])


class TestTriplePolynomialTorque:
    def test_constant_torque(self) -> None:
        func = make_polynomial_torque([5.0], [0.0], [1.0])
        assert func(0.0) == (5.0, 0.0, 1.0)
        assert func(2.0) == (5.0, 0.0, 1.0)

    def test_empty_coefficients_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            make_polynomial_torque([], [1.0], [0.0])


class TestTripleSimulationBasics:
    def test_produces_result(
        self,
        triple_params: TriplePendulumParams,
        zero_torque: Callable[[float], tuple[float, float, float]],
        aligned_state: np.ndarray,
    ) -> None:
        result = run_simulation(
            triple_params, aligned_state, t_end=0.5, torque_func=zero_torque, dt=0.01
        )
        assert isinstance(result, TripleSimulationResult)
        assert result.n_steps > 10

    def test_initial_state_preserved(
        self,
        triple_params: TriplePendulumParams,
        zero_torque: Callable[[float], tuple[float, float, float]],
        aligned_state: np.ndarray,
    ) -> None:
        result = run_simulation(
            triple_params, aligned_state, t_end=0.1, torque_func=zero_torque, dt=0.01
        )
        assert np.allclose(result.states[0], aligned_state, atol=1e-6)

    def test_time_monotonically_increases(
        self,
        triple_params: TriplePendulumParams,
        zero_torque: Callable[[float], tuple[float, float, float]],
        aligned_state: np.ndarray,
    ) -> None:
        result = run_simulation(
            triple_params, aligned_state, t_end=1.0, torque_func=zero_torque, dt=0.01
        )
        assert all(np.diff(result.t) > 0)

    def test_run_with_joint_limits(
        self,
        triple_params: TriplePendulumParams,
        zero_torque: Callable[[float], tuple[float, float, float]],
        aligned_state: np.ndarray,
    ) -> None:
        from double_pendulum_golf.physics import JointLimitsNDOF

        limits = JointLimitsNDOF(
            angle_min=np.array([-1.0, -1.0, -1.0]),
            angle_max=np.array([1.0, 1.0, 1.0]),
            stiffness=100.0,
            damping=10.0,
        )
        result = run_simulation(
            triple_params,
            aligned_state,
            t_end=0.01,
            torque_func=zero_torque,
            dt=0.005,
            limits=limits,
        )
        assert result.n_steps >= 2


class TestTripleSimulationResultContracts:
    """Trajectory-level DbC validation for TripleSimulationResult."""

    def test_constructor_rejects_non_monotonic_time(
        self,
        triple_params: TriplePendulumParams,
    ) -> None:
        with pytest.raises((ValueError, TypeError), match="strictly increasing"):
            TripleSimulationResult(
                t=np.array([0.0, 0.0]),
                states=np.zeros((2, 6)),
                params=triple_params,
                torque_func=lambda _t: (0.0, 0.0, 0.0),
            )

    def test_constructor_rejects_wrong_state_width(
        self,
        triple_params: TriplePendulumParams,
    ) -> None:
        with pytest.raises((ValueError, TypeError), match="states must have width 6"):
            TripleSimulationResult(
                t=np.array([0.0, 0.1]),
                states=np.zeros((2, 5)),
                params=triple_params,
                torque_func=lambda _t: (0.0, 0.0, 0.0),
            )


class TestTripleEnergyConservation:
    def test_energy_conserved_free_pendulum(
        self,
        triple_params: TriplePendulumParams,
        zero_torque: Callable[[float], tuple[float, float, float]],
    ) -> None:
        """Energy conservation over 1 s with tight tolerances.

        Uses LSODA (adaptive stiffness-aware method) and tight tolerances
        to verify that the solver interface correctly accepts rtol/atol kwargs
        and that energy drift stays below 2% for a 1-second free-pendulum run.
        The 2% bound is appropriate for DOP853 on a chaotic triple pendulum.
        """
        state0 = np.array([np.radians(45), np.radians(30), np.radians(-15), 0.0, 0.0, 0.0])
        result = run_simulation(
            triple_params,
            state0,
            t_end=1.0,
            torque_func=zero_torque,
            dt=0.005,
            method="LSODA",
            rtol=1e-8,
            atol=1e-10,
        )
        E0 = total_energy(result.states[0], triple_params)
        energies = np.array([total_energy(s, triple_params) for s in result.states])
        max_drift = np.max(np.abs(energies - E0))
        relative_drift = max_drift / abs(E0) if abs(E0) > 1e-10 else max_drift
        # 2% relative drift is acceptable for a chaotic 3-link pendulum over 1 s
        assert relative_drift < 0.02


class TestTripleSimulationAccessors:
    """Batch accessors should expose full-trajectory views."""

    def test_all_positions_and_energies(
        self,
        triple_params: TriplePendulumParams,
        zero_torque: Callable[[float], tuple[float, float, float]],
        aligned_state: np.ndarray,
    ) -> None:
        result = run_simulation(
            triple_params, aligned_state, t_end=0.1, torque_func=zero_torque, dt=0.01
        )
        positions = result.all_positions()
        energies = result.all_energies()
        assert len(positions) == result.n_steps
        assert energies["kinetic"].shape == (result.n_steps,)
        assert energies["potential"].shape == (result.n_steps,)
        assert energies["total"].shape == (result.n_steps,)
