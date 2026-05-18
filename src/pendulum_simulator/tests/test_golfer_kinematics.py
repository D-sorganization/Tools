# ruff: noqa: E501
"""Tests for golfer_kinematics.py — forward kinematics for golfer upper-body model.

Covers:
- _hub_position: position of hub joint
- _shoulder_position: shoulder positions off the hub bar
- _chain_endpoint: serial chain endpoint computation
- _absolute_angles: relative-to-absolute angle conversion
- forward_kinematics: full FK for all named joints

Mathematical properties tested:
- Segment-length invariants (distance between consecutive joints)
- Symmetry under symmetric parameter sets
- Identity at zero configuration
- Finite positions for diverse configurations
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch

from double_pendulum_golf.golfer_kinematics import (
    _absolute_angles,
    _chain_endpoint,
    _hub_position,
    _shoulder_position,
    forward_kinematics,
)
from double_pendulum_golf.physics_golfer import GolferParams, N_DOF

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sym_params() -> GolferParams:
    """Symmetric golfer (equal left-right arm lengths) without scapula links."""
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
        L_rscap=0.0,
        L_lscap=0.0,
    )


@pytest.fixture
def scapula_params() -> GolferParams:
    """Params WITH non-zero scapula lengths to test scapula code paths."""
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
        L_rscap=0.05,
        L_lscap=0.05,
    )


@pytest.fixture
def zero_q() -> np.ndarray:
    """Zero generalised coordinates (all hanging down)."""
    return np.zeros(N_DOF)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dist(a: tuple, b: tuple) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


# ---------------------------------------------------------------------------
# Tests for _hub_position
# ---------------------------------------------------------------------------


class TestHubPosition:
    def test_zero_angle_hub_above_origin(self, sym_params: GolferParams) -> None:
        """theta_hub = 0 → hub is directly above origin (standoff upward)."""
        x, y = _hub_position(0.0, sym_params)
        assert abs(x) < 1e-10
        assert y == pytest.approx(sym_params.L_hub, abs=1e-10)

    def test_pi_half_hub_to_the_left(self, sym_params: GolferParams) -> None:
        """theta_hub = π/2 → hub moves to the left."""
        x, y = _hub_position(np.pi / 2, sym_params)
        assert x == pytest.approx(-sym_params.L_hub, abs=1e-10)
        assert abs(y) < 1e-10

    def test_hub_distance_from_origin_equals_L_hub(
        self, sym_params: GolferParams
    ) -> None:
        """Distance |hub| must equal L_hub for all angles."""
        for theta in np.linspace(-np.pi, np.pi, 20):
            x, y = _hub_position(theta, sym_params)
            dist = np.hypot(x, y)
            assert dist == pytest.approx(sym_params.L_hub, abs=1e-10)

    def test_returns_tuple_of_two_floats(self, sym_params: GolferParams) -> None:
        result = _hub_position(0.5, sym_params)
        assert isinstance(result, tuple)
        assert len(result) == 2
        x, y = result
        assert np.isfinite(x) and np.isfinite(y)


# ---------------------------------------------------------------------------
# Tests for _shoulder_position
# ---------------------------------------------------------------------------


class TestShoulderPosition:
    def test_distance_from_hub_equals_d_shoulder(
        self, sym_params: GolferParams
    ) -> None:
        hub = (0.0, sym_params.L_hub)
        for d in [0.15, 0.20, 0.25]:
            rs = _shoulder_position(hub, 0.0, d, +1.0)
            assert _dist(hub, rs) == pytest.approx(d, abs=1e-10)

    def test_right_and_left_symmetric_at_zero(self, sym_params: GolferParams) -> None:
        hub = _hub_position(0.0, sym_params)
        d = sym_params.d_rs
        rs = _shoulder_position(hub, 0.0, d, +1.0)
        ls = _shoulder_position(hub, 0.0, d, -1.0)
        # Should be mirror images in x, same y
        assert rs[0] == pytest.approx(-ls[0], abs=1e-10)
        assert rs[1] == pytest.approx(ls[1], abs=1e-10)

    def test_returns_tuple_of_two_floats(self, sym_params: GolferParams) -> None:
        hub = (0.0, 0.15)
        result = _shoulder_position(hub, 0.3, 0.2, 1.0)
        assert isinstance(result, tuple) and len(result) == 2


# ---------------------------------------------------------------------------
# Tests for _absolute_angles
# ---------------------------------------------------------------------------


class TestAbsoluteAngles:
    def test_zero_relative_angles_returns_hub_angle(self) -> None:
        result = _absolute_angles(np.pi / 4, [0.0, 0.0, 0.0])
        expected = [np.pi / 4, np.pi / 4, np.pi / 4]
        np.testing.assert_allclose(result, expected)

    def test_cumulative_sum(self) -> None:
        # hub=0.1, relatives=[0.2, 0.3, 0.4]
        # → [0.3, 0.6, 1.0]
        result = _absolute_angles(0.1, [0.2, 0.3, 0.4])
        assert result[0] == pytest.approx(0.1 + 0.2)
        assert result[1] == pytest.approx(0.1 + 0.2 + 0.3)
        assert result[2] == pytest.approx(0.1 + 0.2 + 0.3 + 0.4)

    def test_empty_relative_angles(self) -> None:
        result = _absolute_angles(1.0, [])
        assert result == []

    def test_negative_angles(self) -> None:
        result = _absolute_angles(0.5, [-0.3, 0.1])
        assert result[0] == pytest.approx(0.2)
        assert result[1] == pytest.approx(0.3)

    def test_returns_list(self) -> None:
        result = _absolute_angles(0.0, [0.1])
        assert isinstance(result, list)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Tests for _chain_endpoint
# ---------------------------------------------------------------------------


class TestChainEndpoint:
    def test_single_segment_straight_down(self) -> None:
        """angle=0, length=1 → endpoint at origin + (0, -1) in world frame."""
        origin = (0.0, 0.0)
        ep = _chain_endpoint(origin, [0.0], [1.0])
        # x += sin(0)*1 = 0; y -= cos(0)*1 = -1
        assert ep[0] == pytest.approx(0.0, abs=1e-10)
        assert ep[1] == pytest.approx(-1.0, abs=1e-10)

    def test_single_segment_horizontal(self) -> None:
        """angle=π/2 → endpoint entirely to the right."""
        ep = _chain_endpoint((0.0, 0.0), [np.pi / 2], [1.0])
        assert ep[0] == pytest.approx(1.0, abs=1e-10)
        assert ep[1] == pytest.approx(0.0, abs=1e-10)

    def test_two_segment_chain(self) -> None:
        """Two segments of length 1, both straight down → endpoint at (0, -2)."""
        ep = _chain_endpoint((0.0, 0.0), [0.0, 0.0], [1.0, 1.0])
        assert ep[0] == pytest.approx(0.0, abs=1e-10)
        assert ep[1] == pytest.approx(-2.0, abs=1e-10)

    def test_offset_origin(self) -> None:
        """Origin offset should propagate."""
        ep = _chain_endpoint((1.0, 2.0), [0.0], [1.0])
        assert ep[0] == pytest.approx(1.0)
        assert ep[1] == pytest.approx(1.0)

    def test_returns_tuple(self) -> None:
        ep = _chain_endpoint((0.0, 0.0), [0.0], [1.0])
        assert isinstance(ep, tuple)
        assert len(ep) == 2


# ---------------------------------------------------------------------------
# Tests for forward_kinematics (pure Python path)
# ---------------------------------------------------------------------------


class TestForwardKinematics:
    """Test forward_kinematics with native backend disabled (pure Python path)."""

    def _fk(self, q: np.ndarray, p: GolferParams) -> dict:
        """Call FK with native backend mocked to return None (force Python path)."""
        with patch(
            "double_pendulum_golf.golfer_kinematics._native_backend.golfer_forward_kinematics",
            return_value=None,
        ):
            return forward_kinematics(q, p)

    def test_zero_config_returns_required_keys(self, sym_params: GolferParams) -> None:
        pos = self._fk(np.zeros(N_DOF), sym_params)
        for key in (
            "origin",
            "hub",
            "rs",
            "re",
            "rh",
            "ls",
            "le",
            "lh",
            "club_base",
            "club_tip",
            "grip_right",
            "grip_left",
        ):
            assert key in pos, f"Missing key: {key}"

    def test_all_positions_finite(self, sym_params: GolferParams) -> None:
        rng = np.random.default_rng(7)
        for _ in range(10):
            q = rng.uniform(-np.pi / 2, np.pi / 2, N_DOF)
            pos = self._fk(q, sym_params)
            for key, (x, y) in pos.items():
                assert np.isfinite(x) and np.isfinite(
                    y
                ), f"Non-finite position for joint {key!r}: ({x}, {y})"

    def test_origin_always_zero(self, sym_params: GolferParams) -> None:
        for q in [np.zeros(N_DOF), np.ones(N_DOF) * 0.3]:
            pos = self._fk(q, sym_params)
            assert pos["origin"] == (0.0, 0.0)

    def test_hub_distance_equals_L_hub(self, sym_params: GolferParams) -> None:
        """Hub must always be L_hub away from origin."""
        for theta_hub in [0.0, 0.5, -0.5, np.pi / 3]:
            q = np.zeros(N_DOF)
            q[0] = theta_hub
            pos = self._fk(q, sym_params)
            d = _dist((0.0, 0.0), pos["hub"])
            assert d == pytest.approx(sym_params.L_hub, abs=1e-8)

    def test_upper_arm_length_right(self, sym_params: GolferParams) -> None:
        """Distance rs → re should equal L_r_upper."""
        q = np.zeros(N_DOF)
        pos = self._fk(q, sym_params)
        d = _dist(pos["rs"], pos["re"])
        assert d == pytest.approx(sym_params.L_r_upper, abs=1e-8)

    def test_forearm_length_right(self, sym_params: GolferParams) -> None:
        """Distance re → rh should equal L_r_fore."""
        q = np.zeros(N_DOF)
        pos = self._fk(q, sym_params)
        d = _dist(pos["re"], pos["rh"])
        assert d == pytest.approx(sym_params.L_r_fore, abs=1e-8)

    def test_upper_arm_length_left(self, sym_params: GolferParams) -> None:
        """Distance ls → le should equal L_l_upper."""
        q = np.zeros(N_DOF)
        pos = self._fk(q, sym_params)
        d = _dist(pos["ls"], pos["le"])
        assert d == pytest.approx(sym_params.L_l_upper, abs=1e-8)

    def test_forearm_length_left(self, sym_params: GolferParams) -> None:
        """Distance le → lh should equal L_l_fore."""
        q = np.zeros(N_DOF)
        pos = self._fk(q, sym_params)
        d = _dist(pos["le"], pos["lh"])
        assert d == pytest.approx(sym_params.L_l_fore, abs=1e-8)

    def test_club_length(self, sym_params: GolferParams) -> None:
        """Distance club_base → club_tip should equal L_club."""
        q = np.zeros(N_DOF)
        pos = self._fk(q, sym_params)
        d = _dist(pos["club_base"], pos["club_tip"])
        assert d == pytest.approx(sym_params.L_club, abs=1e-8)

    def test_symmetric_arms_at_zero(self, sym_params: GolferParams) -> None:
        """With symmetric parameters and zero angles, arms should be mirror images."""
        q = np.zeros(N_DOF)
        pos = self._fk(q, sym_params)
        # rh and lh should be mirror images in x
        assert pos["rh"][0] == pytest.approx(-pos["lh"][0], abs=1e-8)
        assert pos["rh"][1] == pytest.approx(pos["lh"][1], abs=1e-8)

    def test_rotation_changes_positions(self, sym_params: GolferParams) -> None:
        """Rotating hub must change the positions of all joints."""
        q0 = np.zeros(N_DOF)
        q1 = np.zeros(N_DOF)
        q1[0] = np.pi / 4
        pos0 = self._fk(q0, sym_params)
        pos1 = self._fk(q1, sym_params)
        assert pos0["hub"] != pos1["hub"]
        assert pos0["rs"] != pos1["rs"]

    def test_extended_state_is_truncated(self, sym_params: GolferParams) -> None:
        """FK should accept a 16-element state and truncate to first 8."""
        q_extended = np.zeros(2 * N_DOF)
        q_short = np.zeros(N_DOF)
        pos_ext = self._fk(q_extended, sym_params)
        pos_short = self._fk(q_short, sym_params)
        for key in ("hub", "rs", "rh"):
            assert pos_ext[key][0] == pytest.approx(pos_short[key][0], abs=1e-10)
            assert pos_ext[key][1] == pytest.approx(pos_short[key][1], abs=1e-10)

    def test_scapula_keys_present_when_nonzero(
        self, scapula_params: GolferParams
    ) -> None:
        """When L_rscap > 0, 'rscap' and 'lscap' should appear in the result."""
        q = np.zeros(N_DOF)
        pos = self._fk(q, scapula_params)
        assert "rscap" in pos
        assert "lscap" in pos

    def test_scapula_keys_absent_when_zero(self, sym_params: GolferParams) -> None:
        """When L_rscap = 0, 'rscap' and 'lscap' should NOT appear."""
        q = np.zeros(N_DOF)
        pos = self._fk(q, sym_params)
        assert "rscap" not in pos
        assert "lscap" not in pos

    def test_native_backend_bypass(self, sym_params: GolferParams) -> None:
        """If native backend returns a valid dict, FK should return it directly."""
        mock_positions = {"hub": (0.1, 0.2), "origin": (0.0, 0.0)}
        with patch(
            "double_pendulum_golf.golfer_kinematics._native_backend.golfer_forward_kinematics",
            return_value=mock_positions,
        ):
            result = forward_kinematics(np.zeros(N_DOF), sym_params)
        assert result is mock_positions

    def test_diverse_configs_finite(self, sym_params: GolferParams) -> None:
        """Large rotation angles should still produce finite positions."""
        rng = np.random.default_rng(99)
        for _ in range(20):
            q = rng.uniform(-np.pi, np.pi, N_DOF)
            pos = self._fk(q, sym_params)
            for key, (x, y) in pos.items():
                assert np.isfinite(x) and np.isfinite(y), f"Non-finite at key={key!r}"
