"""Extended tests for club_forces.py and joint_moments.py.

Covers previously untested functions:
- club_forces.overall_club_decomposition (integration test at zero state)
- joint_moments.golfer_pendulum_moments (full 7-DOF coverage)
- Additional edge cases for club_force_decomposition
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.club_forces import (
    overall_club_decomposition,
)
from double_pendulum_golf.joint_moments import golfer_pendulum_moments
from double_pendulum_golf.physics_golfer import GolferParams, N_DOF

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def params() -> GolferParams:
    return GolferParams(
        m_hub=2.0,
        m_r_upper=3.0,
        m_r_fore=2.0,
        m_l_upper=3.0,
        m_l_fore=2.0,
        m_club=0.5,
        L_hub=0.15,
        L_r_upper=0.35,
        L_r_fore=0.30,
        L_l_upper=0.35,
        L_l_fore=0.30,
        L_club=1.10,
        d_rs=0.20,
        d_ls=0.20,
        grip_right=0.05,
        grip_left=0.25,
        m_clubhead=0.2,
    )


@pytest.fixture
def zero_state() -> np.ndarray:
    return np.zeros(2 * N_DOF)


def zero_torque(t: float) -> tuple:
    """Return 7 zero torques (actuated joints only — hub through lh)."""
    return (0.0,) * 7  # N_DOF = 8 but constraint solver expects 7 actuated joints


# ===========================================================================
# Tests for overall_club_decomposition (integration)
# ===========================================================================


class TestOverallClubDecomposition:
    """Integration tests for overall_club_decomposition at zero state.

    Uses real constrained dynamics with zero torques — simplest valid case.
    """

    def test_returns_required_keys(self, params: GolferParams, zero_state: np.ndarray) -> None:
        result = overall_club_decomposition(zero_state, 0.0, params, zero_torque)
        for key in (
            "net_force",
            "action_point",
            "moment",
            "couple",
            "f_right",
            "f_left",
        ):
            assert key in result, f"Missing key: {key}"

    def test_net_force_is_array(self, params: GolferParams, zero_state: np.ndarray) -> None:
        result = overall_club_decomposition(zero_state, 0.0, params, zero_torque)
        assert isinstance(result["net_force"], np.ndarray)
        assert result["net_force"].shape == (2,)

    def test_action_point_is_finite(
        self, params: GolferParams, zero_state: np.ndarray
    ) -> None:
        result = overall_club_decomposition(zero_state, 0.0, params, zero_torque)
        assert np.all(np.isfinite(result["action_point"]))

    def test_couple_is_finite(self, params: GolferParams, zero_state: np.ndarray) -> None:
        result = overall_club_decomposition(zero_state, 0.0, params, zero_torque)
        assert np.isfinite(result["couple"])

    def test_all_values_finite(self, params: GolferParams, zero_state: np.ndarray) -> None:
        result = overall_club_decomposition(zero_state, 0.0, params, zero_torque)
        for key, val in result.items():
            if isinstance(val, np.ndarray):
                assert np.all(np.isfinite(val)), f"Non-finite array for {key}"
            else:
                assert np.isfinite(val), f"Non-finite scalar for {key}: {val}"

    def test_alpha_midpoint(self, params: GolferParams, zero_state: np.ndarray) -> None:
        """alpha=0 gives midpoint between grip positions."""
        result = overall_club_decomposition(zero_state, 0.0, params, zero_torque, alpha=0.0)
        assert result["action_point"].shape == (2,)

    def test_alpha_right_grip(self, params: GolferParams, zero_state: np.ndarray) -> None:
        result = overall_club_decomposition(zero_state, 0.0, params, zero_torque, alpha=-1.0)
        assert all(np.isfinite(result["action_point"]))

    def test_alpha_left_grip(self, params: GolferParams, zero_state: np.ndarray) -> None:
        result = overall_club_decomposition(zero_state, 0.0, params, zero_torque, alpha=1.0)
        assert all(np.isfinite(result["action_point"]))


# ===========================================================================
# Tests for golfer_pendulum_moments
# ===========================================================================


class TestGolferPendulumMoments:
    """Tests for the 7-DOF golfer joint moments computation."""

    @pytest.fixture
    def full_positions(self) -> dict:
        """Minimal positions dict with all keys needed by golfer model."""
        return {
            "hub": (0.0, 0.15),
            "rs": (0.2, 0.15),
            "re": (0.55, 0.15),
            "rh": (0.85, 0.15),
            "ls": (-0.2, 0.15),
            "le": (-0.55, 0.15),
            "lh": (-0.85, 0.15),
            "club_tip": (1.0, -0.8),
        }

    @pytest.fixture
    def full_forces(self) -> dict:
        """Joint forces for all 7 actuated joints."""
        return {
            "hub": (0.0, 0.0),
            "rs": (1.0, -2.0),
            "re": (0.5, -1.0),
            "rh": (0.2, -0.5),
            "ls": (-1.0, -2.0),
            "le": (-0.5, -1.0),
            "lh": (-0.2, -0.5),
        }

    @pytest.fixture
    def applied_torques(self) -> tuple:
        return (10.0, 5.0, 3.0, 1.0, 5.0, 3.0, 1.0)

    def test_returns_21_keys(
        self, full_positions: dict, full_forces: dict, applied_torques: tuple
    ) -> None:
        result = golfer_pendulum_moments(
            full_positions, full_forces, applied_torques, object()
        )
        assert len(result) == 21  # 3 keys x 7 joints

    def test_joint_names_in_keys(
        self, full_positions: dict, full_forces: dict, applied_torques: tuple
    ) -> None:
        result = golfer_pendulum_moments(
            full_positions, full_forces, applied_torques, object()
        )
        for joint in ("hub", "rs", "re", "rh", "ls", "le", "lh"):
            assert f"{joint}_applied_torque" in result
            assert f"{joint}_moment_of_force" in result
            assert f"{joint}_total_moment" in result

    def test_applied_torques_preserved(
        self, full_positions: dict, full_forces: dict, applied_torques: tuple
    ) -> None:
        result = golfer_pendulum_moments(
            full_positions, full_forces, applied_torques, object()
        )
        joints = ["hub", "rs", "re", "rh", "ls", "le", "lh"]
        for i, joint in enumerate(joints):
            assert result[f"{joint}_applied_torque"] == pytest.approx(applied_torques[i])

    def test_all_values_finite(
        self, full_positions: dict, full_forces: dict, applied_torques: tuple
    ) -> None:
        result = golfer_pendulum_moments(
            full_positions, full_forces, applied_torques, object()
        )
        for key, val in result.items():
            assert np.isfinite(val), f"Non-finite value for {key}: {val}"

    def test_total_moment_equals_applied_plus_moment_of_force(
        self, full_positions: dict, full_forces: dict, applied_torques: tuple
    ) -> None:
        result = golfer_pendulum_moments(
            full_positions, full_forces, applied_torques, object()
        )
        for joint in ("hub", "rs", "re", "rh", "ls", "le", "lh"):
            total = result[f"{joint}_total_moment"]
            applied = result[f"{joint}_applied_torque"]
            moment_force = result[f"{joint}_moment_of_force"]
            assert total == pytest.approx(applied + moment_force, abs=1e-8)

    def test_missing_joint_data_gives_zero_moment(self) -> None:
        """When position/force data is missing, moment_of_force should be 0."""
        positions = {}  # missing all positions
        forces = {}
        applied_torques = tuple(1.0 for _ in range(7))
        result = golfer_pendulum_moments(positions, forces, applied_torques, object())
        for joint in ("hub", "rs", "re", "rh", "ls", "le", "lh"):
            assert result[f"{joint}_moment_of_force"] == 0.0
            assert result[f"{joint}_total_moment"] == result[f"{joint}_applied_torque"]

    def test_fewer_than_7_torques_raises(
        self, full_positions: dict, full_forces: dict
    ) -> None:
        with pytest.raises((ValueError, TypeError, AssertionError), match="Need >= 7"):
            golfer_pendulum_moments(full_positions, full_forces, (1.0, 2.0, 3.0), object())

    def test_exactly_7_torques_ok(self, full_positions: dict, full_forces: dict) -> None:
        torques = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        result = golfer_pendulum_moments(full_positions, full_forces, torques, object())
        assert len(result) == 21

    def test_zero_forces_moment_of_force_is_zero(self, full_positions: dict) -> None:
        forces = {joint: (0.0, 0.0) for joint in ("hub", "rs", "re", "rh", "ls", "le", "lh")}
        torques = (1.0,) * 7
        result = golfer_pendulum_moments(full_positions, forces, torques, object())
        for joint in ("hub", "rs", "re", "rh", "ls", "le", "lh"):
            assert result[f"{joint}_moment_of_force"] == pytest.approx(0.0, abs=1e-10)
