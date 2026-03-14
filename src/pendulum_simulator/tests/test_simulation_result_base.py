"""Tests for the TrajectoryResultMixin in simulation_result_base.py.

Covers:
- n_steps property
- _validate_trajectory() preconditions
- _check_idx() bounds checking
- _assert_energy_finite() static method
- all_positions(), all_mass_matrices(), all_energies(), all_accelerations()
- all_torques(), all_friction_torques(), all_total_torques()
- total_torques_at() with mocked friction and drive torques
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from double_pendulum_golf.simulation_result_base import TrajectoryResultMixin


# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing
# ---------------------------------------------------------------------------


class _ConcreteResult(TrajectoryResultMixin):
    """Concrete test double that satisfies TrajectoryResultMixin interface."""

    def __init__(
        self,
        t: np.ndarray,
        states: np.ndarray,
        state_width: int = 4,
    ) -> None:
        self.t = t
        self.states = states
        self._state_width = state_width
        self.torque_func = lambda time: np.zeros(2)

    def positions_at(self, idx: int) -> Any:
        return {"theta1": float(self.states[idx, 0])}

    def mass_matrix_at(self, idx: int) -> np.ndarray:
        return np.eye(2)

    def energy_at(self, idx: int) -> dict[str, float]:
        return {"kinetic": 1.0, "potential": 2.0, "total": 3.0}

    def accelerations_at(self, idx: int) -> np.ndarray:
        return np.zeros(2)

    def torques_at(self, idx: int) -> np.ndarray:
        return np.array([1.0, 0.5])

    def friction_torques_at(self, idx: int) -> np.ndarray:
        return np.zeros(2)


def _make_result(n: int = 5, state_width: int = 4) -> _ConcreteResult:
    t = np.linspace(0.0, 1.0, n)
    states = np.random.default_rng(42).random((n, state_width))
    return _ConcreteResult(t, states, state_width)


# ---------------------------------------------------------------------------
# Tests for n_steps property
# ---------------------------------------------------------------------------


class TestNSteps:
    def test_n_steps_matches_t_length(self) -> None:
        r = _make_result(n=10)
        assert r.n_steps == 10

    def test_n_steps_single(self) -> None:
        r = _make_result(n=1)
        assert r.n_steps == 1

    def test_n_steps_large(self) -> None:
        r = _make_result(n=1000)
        assert r.n_steps == 1000


# ---------------------------------------------------------------------------
# Tests for _validate_trajectory
# ---------------------------------------------------------------------------


class TestValidateTrajectory:
    def test_valid_trajectory_passes(self) -> None:
        r = _make_result(n=5, state_width=4)
        r._validate_trajectory(4)  # should not raise

    def test_single_step_passes(self) -> None:
        r = _make_result(n=1, state_width=4)
        r._validate_trajectory(4)

    def test_wrong_state_width_raises(self) -> None:
        r = _make_result(n=5, state_width=4)
        with pytest.raises(AssertionError, match="width"):
            r._validate_trajectory(6)

    def test_non_finite_t_raises(self) -> None:
        r = _make_result(n=4, state_width=4)
        r.t[2] = float("nan")
        with pytest.raises(AssertionError, match="finite"):
            r._validate_trajectory(4)

    def test_non_finite_states_raises(self) -> None:
        r = _make_result(n=4, state_width=4)
        r.states[1, 0] = float("inf")
        with pytest.raises(AssertionError, match="finite"):
            r._validate_trajectory(4)

    def test_non_monotonic_t_raises(self) -> None:
        r = _make_result(n=4, state_width=4)
        r.t = np.array([0.0, 0.5, 0.3, 1.0])  # out of order
        with pytest.raises(AssertionError, match="strictly increasing"):
            r._validate_trajectory(4)

    def test_t_size_mismatch_with_states_raises(self) -> None:
        r = _make_result(n=5, state_width=4)
        r.t = np.linspace(0, 1, 3)  # length 3 but states has 5 rows
        with pytest.raises(AssertionError):
            r._validate_trajectory(4)

    def test_1d_t_required(self) -> None:
        r = _make_result(n=5, state_width=4)
        r.t = r.t.reshape(1, -1)  # wrong shape
        with pytest.raises(AssertionError, match="1D"):
            r._validate_trajectory(4)

    def test_2d_states_required(self) -> None:
        r = _make_result(n=5, state_width=4)
        r.states = r.states.flatten()  # wrong shape
        with pytest.raises(AssertionError, match="2D"):
            r._validate_trajectory(4)


# ---------------------------------------------------------------------------
# Tests for _check_idx
# ---------------------------------------------------------------------------


class TestCheckIdx:
    def test_valid_first_index(self) -> None:
        r = _make_result(n=5)
        r._check_idx(0)  # should not raise

    def test_valid_last_index(self) -> None:
        r = _make_result(n=5)
        r._check_idx(4)

    def test_valid_middle_index(self) -> None:
        r = _make_result(n=5)
        r._check_idx(2)

    def test_negative_index_raises(self) -> None:
        r = _make_result(n=5)
        with pytest.raises(AssertionError):
            r._check_idx(-1)

    def test_index_equal_to_n_raises(self) -> None:
        r = _make_result(n=5)
        with pytest.raises(AssertionError):
            r._check_idx(5)

    def test_large_index_raises(self) -> None:
        r = _make_result(n=5)
        with pytest.raises(AssertionError):
            r._check_idx(100)


# ---------------------------------------------------------------------------
# Tests for _assert_energy_finite
# ---------------------------------------------------------------------------


class TestAssertEnergyFinite:
    def test_all_finite_passes(self) -> None:
        TrajectoryResultMixin._assert_energy_finite(
            {"kinetic": 1.0, "potential": 2.0, "total": 3.0}, idx=0
        )

    def test_nan_raises(self) -> None:
        with pytest.raises(AssertionError, match="Non-finite"):
            TrajectoryResultMixin._assert_energy_finite(
                {"kinetic": float("nan"), "potential": 2.0, "total": 3.0}, idx=1
            )

    def test_inf_raises(self) -> None:
        with pytest.raises(AssertionError, match="Non-finite"):
            TrajectoryResultMixin._assert_energy_finite(
                {"kinetic": 1.0, "potential": float("inf"), "total": 3.0}, idx=2
            )


# ---------------------------------------------------------------------------
# Tests for batch accessors
# ---------------------------------------------------------------------------


class TestBatchAccessors:
    @pytest.fixture
    def result(self) -> _ConcreteResult:
        return _make_result(n=4)

    def test_all_positions_length(self, result: _ConcreteResult) -> None:
        positions = result.all_positions()
        assert len(positions) == 4

    def test_all_positions_each_is_dict(self, result: _ConcreteResult) -> None:
        for pos in result.all_positions():
            assert isinstance(pos, dict)

    def test_all_mass_matrices_length(self, result: _ConcreteResult) -> None:
        matrices = result.all_mass_matrices()
        assert len(matrices) == 4

    def test_all_mass_matrices_each_is_ndarray(self, result: _ConcreteResult) -> None:
        for M in result.all_mass_matrices():
            assert isinstance(M, np.ndarray)

    def test_all_energies_keys(self, result: _ConcreteResult) -> None:
        energies = result.all_energies()
        assert set(energies.keys()) == {"kinetic", "potential", "total"}

    def test_all_energies_array_length(self, result: _ConcreteResult) -> None:
        energies = result.all_energies()
        for arr in energies.values():
            assert len(arr) == 4

    def test_all_accelerations_shape(self, result: _ConcreteResult) -> None:
        accels = result.all_accelerations()
        assert accels.shape == (4, 2)

    def test_all_torques_shape(self, result: _ConcreteResult) -> None:
        torques = result.all_torques()
        assert torques.shape == (4, 2)

    def test_all_friction_torques_shape(self, result: _ConcreteResult) -> None:
        friction = result.all_friction_torques()
        assert friction.shape == (4, 2)
        np.testing.assert_allclose(friction, 0.0)

    def test_all_total_torques_shape(self, result: _ConcreteResult) -> None:
        total = result.all_total_torques()
        assert total.shape == (4, 2)

    def test_all_total_torques_equals_drive_plus_friction(
        self, result: _ConcreteResult
    ) -> None:
        """total_torques uses torque_func + friction_torques_at.

        Note: torques_at() and total_torques_at() are separate paths.
        total_torques_at() calls torque_func (returns zeros by default) + friction (zeros).
        all_total_torques() calls total_torques_at() for each step.
        """
        total = result.all_total_torques()
        # torque_func returns zeros(2), friction returns zeros(2)
        # so totals should all be zeros
        np.testing.assert_allclose(total, 0.0)


# ---------------------------------------------------------------------------
# Tests for total_torques_at
# ---------------------------------------------------------------------------


class TestTotalTorquesAt:
    def test_total_torques_at_valid_index(self) -> None:
        r = _make_result(n=5)
        result_arr = r.total_torques_at(2)
        assert isinstance(result_arr, np.ndarray)

    def test_total_torques_at_invalid_index_raises(self) -> None:
        r = _make_result(n=5)
        with pytest.raises(AssertionError):
            r.total_torques_at(10)

    def test_total_torques_at_zero_friction(self) -> None:
        """total = drive + 0 = drive."""
        r = _make_result(n=3)
        # torque_func returns zeros, friction is zeros → total should be zeros
        r.torque_func = lambda t: np.zeros(2)
        total = r.total_torques_at(0)
        np.testing.assert_allclose(total, 0.0)

    def test_total_torques_at_combines_correctly(self) -> None:
        r = _make_result(n=3)
        r.torque_func = lambda t: np.array([10.0, 5.0])
        total = r.total_torques_at(0)
        # friction_torques_at returns zeros; total = [10, 5] + [0, 0] = [10, 5]
        np.testing.assert_allclose(total, [10.0, 5.0])
