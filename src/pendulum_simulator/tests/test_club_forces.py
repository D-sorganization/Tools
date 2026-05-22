"""
Tests for club_forces module — net force & equivalent couple on the club.

Covers:
- Net force calculation (sum of two hand forces)
- Moment of net force about action point
- Equivalent couple computation
- Action-point parameterisation (0 = midpoint, custom values)
- All three decompositions: overall, ZTCF, DELTA
- Edge cases: zero forces, collinear forces, single hand dominant
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.club_forces import (
    club_action_point,
    equivalent_couple,
    moment_of_net_force,
    net_force_on_club,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_params():
    """Create default GolferParams for testing."""
    from double_pendulum_golf.physics_golfer import GolferParams

    return GolferParams(
        m_hub=40.0,
        m_r_upper=3.0,
        m_r_fore=1.5,
        m_l_upper=3.0,
        m_l_fore=1.5,
        m_club=0.5,
        L_hub=0.5,
        L_r_upper=0.3,
        L_r_fore=0.25,
        L_l_upper=0.3,
        L_l_fore=0.25,
        L_club=1.0,
        d_rs=0.2,
        d_ls=0.2,
        grip_right=0.3,
        grip_left=0.3,
        g=9.81,
    )


# ---------------------------------------------------------------------------
# club_action_point
# ---------------------------------------------------------------------------


class TestClubActionPoint:
    """Test the configurable action point on the club."""

    def test_default_midpoint(self, default_params):
        """alpha=0 (default) gives the midpoint between grips."""
        q = np.zeros(8)
        pos = club_action_point(q, default_params, alpha=0.0)
        assert len(pos) == 2
        # Midpoint between grip_right and grip_left positions
        assert np.isfinite(pos[0]) and np.isfinite(pos[1])

    def test_alpha_zero_is_midpoint(self, default_params):
        """alpha=0 means midpoint between two grip locations."""
        q = np.zeros(8)
        from double_pendulum_golf.golfer_kinematics import forward_kinematics

        fk = forward_kinematics(q, default_params)
        grip_r = np.array(fk["grip_right"])
        grip_l = np.array(fk["grip_left"])
        expected_mid = 0.5 * (grip_r + grip_l)

        pos = club_action_point(q, default_params, alpha=0.0)
        np.testing.assert_allclose(pos, expected_mid, atol=1e-10)

    def test_alpha_minus_one_is_right_grip(self, default_params):
        """alpha=-1 means the right grip position."""
        q = np.zeros(8)
        from double_pendulum_golf.golfer_kinematics import forward_kinematics

        fk = forward_kinematics(q, default_params)
        expected = np.array(fk["grip_right"])

        pos = club_action_point(q, default_params, alpha=-1.0)
        np.testing.assert_allclose(pos, expected, atol=1e-10)

    def test_alpha_plus_one_is_left_grip(self, default_params):
        """alpha=+1 means the left grip position."""
        q = np.zeros(8)
        from double_pendulum_golf.golfer_kinematics import forward_kinematics

        fk = forward_kinematics(q, default_params)
        expected = np.array(fk["grip_left"])

        pos = club_action_point(q, default_params, alpha=1.0)
        np.testing.assert_allclose(pos, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# net_force_on_club
# ---------------------------------------------------------------------------


class TestNetForceOnClub:
    """Test the net force vector from two hands acting on the club."""

    def test_returns_2d_vector(self, default_params):
        """Net force is a 2D vector (fx, fy)."""
        f_right = (10.0, -5.0)
        f_left = (3.0, 2.0)
        net = net_force_on_club(f_right, f_left)
        assert len(net) == 2

    def test_sum_of_forces(self, default_params):
        """Net force = F_right + F_left."""
        f_right = (10.0, -5.0)
        f_left = (3.0, 2.0)
        net = net_force_on_club(f_right, f_left)
        np.testing.assert_allclose(net, [13.0, -3.0])

    def test_zero_forces(self, default_params):
        """Zero hand forces give zero net force."""
        net = net_force_on_club((0.0, 0.0), (0.0, 0.0))
        np.testing.assert_allclose(net, [0.0, 0.0])

    def test_equal_opposite_cancel(self, default_params):
        """Equal and opposite forces cancel to zero net force."""
        net = net_force_on_club((5.0, 3.0), (-5.0, -3.0))
        np.testing.assert_allclose(net, [0.0, 0.0], atol=1e-14)


# ---------------------------------------------------------------------------
# moment_of_net_force
# ---------------------------------------------------------------------------


class TestMomentOfNetForce:
    """Test the moment (scalar in 2D) of the net force about the action point."""

    def test_zero_force_zero_moment(self):
        """Zero net force produces zero moment regardless of position."""
        m = moment_of_net_force(
            net_force=np.array([0.0, 0.0]),
            force_point=np.array([1.0, 0.0]),
            action_point=np.array([0.0, 0.0]),
        )
        assert m == pytest.approx(0.0)

    def test_force_through_action_point(self):
        """Force acting at the action point has zero moment arm, hence zero moment."""
        m = moment_of_net_force(
            net_force=np.array([10.0, 5.0]),
            force_point=np.array([1.0, 2.0]),
            action_point=np.array([1.0, 2.0]),
        )
        assert m == pytest.approx(0.0)

    def test_known_moment(self):
        """Known cross-product: r = (1,0), F = (0,1) -> moment = 1*1 - 0*0 = 1."""
        m = moment_of_net_force(
            net_force=np.array([0.0, 1.0]),
            force_point=np.array([1.0, 0.0]),
            action_point=np.array([0.0, 0.0]),
        )
        assert m == pytest.approx(1.0)

    def test_negative_moment(self):
        """r = (0,1), F = (1,0) -> moment = 0*0 - 1*1 = -1."""
        m = moment_of_net_force(
            net_force=np.array([1.0, 0.0]),
            force_point=np.array([0.0, 1.0]),
            action_point=np.array([0.0, 0.0]),
        )
        assert m == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# equivalent_couple
# ---------------------------------------------------------------------------


class TestEquivalentCouple:
    """Test the equivalent couple calculation."""

    def test_couple_definition(self):
        """Couple = total moment from two hands - moment of net force at action point.

        If both hands apply (1, 0) at (0, 1) and (0, -1) respectively about (0, 0):
        - Net force = (2, 0)
        - Net force "acts at" COP which we place at (0, 0) for simplicity
        - Total moment from hands = r1 x F1 + r2 x F2
          = (0,1) x (1,0) + (0,-1) x (1,0) = -1 + 1 = 0
        - Moment of net force at (0,0) = r_cop x F_net = 0 (force through origin)
        - Couple = 0 - 0 = 0
        """
        couple = equivalent_couple(
            f_right=np.array([1.0, 0.0]),
            pos_right=np.array([0.0, 1.0]),
            f_left=np.array([1.0, 0.0]),
            pos_left=np.array([0.0, -1.0]),
            action_point=np.array([0.0, 0.0]),
        )
        assert couple == pytest.approx(0.0)

    def test_pure_couple(self):
        """Equal and opposite forces at different points produce a pure couple.

        F_right = (1, 0) at (0, 1)
        F_left = (-1, 0) at (0, -1)
        Net force = (0, 0)
        Total moment = (0,1) x (1,0) + (0,-1) x (-1,0) = -1 + (-1) = -2
        Moment of net force = 0 (net force is zero)
        Couple = -2 - 0 = -2
        """
        couple = equivalent_couple(
            f_right=np.array([1.0, 0.0]),
            pos_right=np.array([0.0, 1.0]),
            f_left=np.array([-1.0, 0.0]),
            pos_left=np.array([0.0, -1.0]),
            action_point=np.array([0.0, 0.0]),
        )
        assert couple == pytest.approx(-2.0)

    def test_zero_forces_zero_couple(self):
        """Zero forces on both hands produce zero couple."""
        couple = equivalent_couple(
            f_right=np.array([0.0, 0.0]),
            pos_right=np.array([1.0, 0.0]),
            f_left=np.array([0.0, 0.0]),
            pos_left=np.array([0.0, 1.0]),
            action_point=np.array([0.0, 0.0]),
        )
        assert couple == pytest.approx(0.0)

    def test_collinear_forces_no_couple(self):
        """Same direction forces at same height produce zero couple.

        F_right = (1, 0) at (0, 0)
        F_left  = (1, 0) at (1, 0)
        Net force = (2, 0) at action point (0.5, 0)
        M_rh = (0,0)x(1,0)=0, M_lh = (1,0)x(1,0)=0 about (0,0)
        But action_point = (0.5, 0)
        M_rh about AP: (-0.5,0) x (1,0) = 0
        M_lh about AP: (0.5,0) x (1,0) = 0
        Total moment = 0
        M_net at AP: force through AP => 0
        Couple = 0
        """
        couple = equivalent_couple(
            f_right=np.array([1.0, 0.0]),
            pos_right=np.array([0.0, 0.0]),
            f_left=np.array([1.0, 0.0]),
            pos_left=np.array([1.0, 0.0]),
            action_point=np.array([0.5, 0.0]),
        )
        assert couple == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# High-level decomposition integration tests
# ---------------------------------------------------------------------------


class TestClubForceDecomposition:
    """Test the club_force_decomposition function with real physics."""

    @pytest.fixture
    def decomposition_result(self, default_params):
        """Compute zero-state club force decomposition once for all tests."""
        from double_pendulum_golf.club_forces import club_force_decomposition
        from double_pendulum_golf.golfer_constraints import net_joint_forces

        q = np.zeros(8)
        qdot = np.zeros(8)
        qddot = np.zeros(8)
        forces = net_joint_forces(q, qdot, qddot, default_params)
        result = club_force_decomposition(q, qdot, qddot, default_params, forces)
        return {
            "q": q,
            "qdot": qdot,
            "qddot": qddot,
            "forces": forces,
            "result": result,
        }

    def test_overall_decomposition_returns_all_keys(self, decomposition_result):
        """Overall decomposition returns all expected keys."""
        result = decomposition_result["result"]
        expected_keys = {
            "net_force",
            "action_point",
            "moment",
            "couple",
            "f_right",
            "f_left",
            "pos_right",
            "pos_left",
        }
        assert set(result.keys()) == expected_keys

    def test_net_force_is_sum_of_hands(self, decomposition_result):
        """Net force in decomposition equals sum of hand forces."""
        forces = decomposition_result["forces"]
        result = decomposition_result["result"]
        expected = np.array(forces["rh"]) + np.array(forces["lh"])
        np.testing.assert_allclose(result["net_force"], expected)

    def test_all_values_finite(self, decomposition_result):
        """All returned values should be finite."""
        result = decomposition_result["result"]
        for key, val in result.items():
            if isinstance(val, np.ndarray):
                assert np.all(np.isfinite(val)), f"{key} has non-finite values"
            else:
                assert np.isfinite(val), f"{key} is not finite"


class TestZTCFClubDecomposition:
    """Test ZTCF club force decomposition."""

    def test_ztcf_returns_dict(self, default_params):
        from double_pendulum_golf.club_forces import ztcf_club_decomposition

        state = np.zeros(16)
        result = ztcf_club_decomposition(state, default_params)
        assert isinstance(result, dict)
        assert "net_force" in result
        assert "couple" in result

    def test_ztcf_finite_values(self, default_params):
        from double_pendulum_golf.club_forces import ztcf_club_decomposition

        state = np.zeros(16)
        result = ztcf_club_decomposition(state, default_params)
        assert np.all(np.isfinite(result["net_force"]))
        assert np.isfinite(result["couple"])


class TestDELTAClubDecomposition:
    """Test DELTA club force decomposition."""

    def test_delta_returns_dict(self, default_params):
        from double_pendulum_golf.club_forces import delta_club_decomposition

        state = np.zeros(16)
        tau = np.zeros(8)
        result = delta_club_decomposition(state, tau, default_params)
        assert isinstance(result, dict)
        assert "net_force" in result
        assert "couple" in result

    def test_delta_zero_torque_zero_forces(self, default_params):
        """With zero torque, DELTA gives zero accelerations, hence only gravity."""
        from double_pendulum_golf.club_forces import delta_club_decomposition

        state = np.zeros(16)
        tau = np.zeros(8)
        result = delta_club_decomposition(state, tau, default_params)
        # With zero tau and zero velocity, accelerations from DELTA = M+ @ 0 = 0
        # Forces are F = m*a - m*g_vec where g_vec = (0, -g)
        # So F = m*0 - m*(0, -g) = (0, m*g)
        # Net force should be +(m_rh + m_lh)*g in the y direction
        net_fy = result["net_force"][1]
        expected_fy = (
            default_params.m_r_fore + default_params.m_l_fore
        ) * default_params.g
        assert net_fy == pytest.approx(expected_fy, rel=0.01)


# ---------------------------------------------------------------------------
# DbC precondition tests for decomposition functions (GH1478)
# ---------------------------------------------------------------------------


class TestDecompositionDbc:
    """Validate TypeError/ValueError preconditions on decomposition functions."""

    def test_overall_state_wrong_type(self, default_params):
        from double_pendulum_golf.club_forces import overall_club_decomposition

        with pytest.raises(TypeError, match="state must be a numpy ndarray"):
            overall_club_decomposition(
                state=[0.0] * 16,
                t=0.0,
                p=default_params,
                torque_func=lambda t: (0.0,) * 7,
            )

    def test_overall_state_wrong_shape(self, default_params):
        from double_pendulum_golf.club_forces import overall_club_decomposition

        with pytest.raises(ValueError, match="state must have shape"):
            overall_club_decomposition(
                state=np.zeros(8),
                t=0.0,
                p=default_params,
                torque_func=lambda t: (0.0,) * 7,
            )

    def test_overall_t_wrong_type(self, default_params):
        from double_pendulum_golf.club_forces import overall_club_decomposition

        with pytest.raises(TypeError, match="t must be a number"):
            overall_club_decomposition(
                state=np.zeros(16),
                t="now",
                p=default_params,
                torque_func=lambda t: (0.0,) * 7,
            )

    def test_ztcf_state_wrong_type(self, default_params):
        from double_pendulum_golf.club_forces import ztcf_club_decomposition

        with pytest.raises(TypeError, match="state must be a numpy ndarray"):
            ztcf_club_decomposition(state=list(range(16)), p=default_params)

    def test_ztcf_state_wrong_shape(self, default_params):
        from double_pendulum_golf.club_forces import ztcf_club_decomposition

        with pytest.raises(ValueError, match="state must have shape"):
            ztcf_club_decomposition(state=np.zeros(10), p=default_params)

    def test_delta_state_wrong_type(self, default_params):
        from double_pendulum_golf.club_forces import delta_club_decomposition

        with pytest.raises(TypeError, match="state must be a numpy ndarray"):
            delta_club_decomposition(
                state=list(range(16)), tau=np.zeros(8), p=default_params
            )

    def test_delta_tau_wrong_type(self, default_params):
        from double_pendulum_golf.club_forces import delta_club_decomposition

        with pytest.raises(TypeError, match="tau must be a numpy ndarray"):
            delta_club_decomposition(
                state=np.zeros(16), tau=[0.0] * 8, p=default_params
            )

    def test_delta_tau_wrong_shape(self, default_params):
        from double_pendulum_golf.club_forces import delta_club_decomposition

        with pytest.raises(ValueError, match="tau must have shape"):
            delta_club_decomposition(
                state=np.zeros(16), tau=np.zeros(4), p=default_params
            )
