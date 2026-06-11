"""Tests for the corrected golfer model topology.

Validates that the model correctly represents:
- Massless standoff (COM offset adjustment only)
- Upper body (scapula) segments with significant mass (~2 x arm mass)
- Correct default parameter values per the physical description

Closes #1204.

Design by Contract:
- All tests validate physical invariants of the golfer model
- Tests are independent and can run in any order
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.physics_golfer import GolferParams

# ---------------------------------------------------------------------------
# Fixture: default "Address Position" parameters
# ---------------------------------------------------------------------------


@pytest.fixture
def address_params() -> GolferParams:
    """Create GolferParams with corrected default values."""
    return GolferParams(
        m_hub=0.001,  # Standoff: near-zero (massless)
        m_r_upper=3.5,
        m_r_fore=2.0,
        m_l_upper=3.5,
        m_l_fore=2.0,
        m_club=0.5,
        L_hub=0.15,
        L_r_upper=0.35,
        L_r_fore=0.30,
        L_l_upper=0.35,
        L_l_fore=0.30,
        L_club=1.1,
        d_rs=0.20,
        d_ls=0.20,
        grip_right=0.05,
        grip_left=0.25,
        m_clubhead=0.2,
        L_rscap=0.18,
        L_lscap=0.18,
        m_rscap=7.0,
        m_lscap=7.0,
    )


# ---------------------------------------------------------------------------
# Topology invariants
# ---------------------------------------------------------------------------


class TestStandoffMassless:
    """The standoff segment should be effectively massless."""

    def test_standoff_mass_near_zero(self, address_params: GolferParams) -> None:
        """Standoff mass must be near zero (< 0.01 kg)."""
        assert address_params.m_hub < 0.01, (
            f"Standoff mass should be near-zero, got {address_params.m_hub}"
        )

    def test_standoff_mass_positive(self, address_params: GolferParams) -> None:
        """Standoff mass must be positive (required by solver numerics)."""
        assert address_params.m_hub > 0, "Standoff mass must be strictly positive"

    def test_standoff_has_length(self, address_params: GolferParams) -> None:
        """Standoff must have a non-zero length for COM offset adjustment."""
        assert address_params.L_hub > 0, "Standoff length must be positive"


class TestUpperBodyMass:
    """Upper body (scapula) segments should have significant mass (~2x arms)."""

    def test_right_upper_body_heavier_than_arms(self, address_params: GolferParams) -> None:
        """Right upper body mass should be >= right arm total."""
        right_arm_total = address_params.m_r_upper + address_params.m_r_fore
        assert address_params.m_rscap >= right_arm_total, (
            f"Right upper body ({address_params.m_rscap} kg) should be >= "
            f"right arm total ({right_arm_total} kg)"
        )

    def test_left_upper_body_heavier_than_arms(self, address_params: GolferParams) -> None:
        """Left upper body mass should be >= left arm total."""
        left_arm_total = address_params.m_l_upper + address_params.m_l_fore
        assert address_params.m_lscap >= left_arm_total, (
            f"Left upper body ({address_params.m_lscap} kg) should be >= "
            f"left arm total ({left_arm_total} kg)"
        )

    def test_upper_body_symmetric(self, address_params: GolferParams) -> None:
        """Left and right upper body segments should have the same mass."""
        assert address_params.m_rscap == address_params.m_lscap, (
            f"Upper body should be symmetric: R={address_params.m_rscap} "
            f"vs L={address_params.m_lscap}"
        )

    def test_upper_body_length_positive(self, address_params: GolferParams) -> None:
        """Upper body segments must have positive length."""
        assert address_params.L_rscap > 0
        assert address_params.L_lscap > 0


class TestMassDistribution:
    """Overall mass distribution should be physically realistic."""

    def test_total_mass_reasonable(self, address_params: GolferParams) -> None:
        """Total system mass should be in a reasonable range for upper body."""
        total = (
            address_params.m_hub
            + address_params.m_rscap
            + address_params.m_lscap
            + address_params.m_r_upper
            + address_params.m_r_fore
            + address_params.m_l_upper
            + address_params.m_l_fore
            + address_params.m_club
            + address_params.m_clubhead
        )
        # Upper body + arms + club: roughly 10-40 kg is reasonable
        assert 10.0 < total < 40.0, f"Total mass {total:.1f} kg should be in 10-40 kg range"

    def test_standoff_negligible_fraction(self, address_params: GolferParams) -> None:
        """Standoff mass should be < 0.1% of total system mass."""
        total = (
            address_params.m_hub
            + address_params.m_rscap
            + address_params.m_lscap
            + address_params.m_r_upper
            + address_params.m_r_fore
            + address_params.m_l_upper
            + address_params.m_l_fore
            + address_params.m_club
            + address_params.m_clubhead
        )
        fraction = address_params.m_hub / total
        assert fraction < 0.001, f"Standoff mass fraction {fraction:.4f} should be < 0.001"

    def test_upper_body_dominates(self, address_params: GolferParams) -> None:
        """Upper body segments should be the heaviest components."""
        all_masses = [
            address_params.m_hub,
            address_params.m_r_upper,
            address_params.m_r_fore,
            address_params.m_l_upper,
            address_params.m_l_fore,
            address_params.m_club,
            address_params.m_clubhead,
        ]
        assert address_params.m_rscap >= max(all_masses), (
            "Right upper body should be the heaviest individual segment"
        )
        assert address_params.m_lscap >= max(all_masses), (
            "Left upper body should be the heaviest individual segment"
        )


class TestGolferParamsValidation:
    """Design by Contract: GolferParams should enforce its invariants."""

    def test_negative_mass_rejected(self) -> None:
        """Negative mass must raise AssertionError."""
        with pytest.raises((ValueError, TypeError), match="must be positive"):
            GolferParams(
                m_hub=-1.0,
                m_r_upper=3.5,
                m_r_fore=2.0,
                m_l_upper=3.5,
                m_l_fore=2.0,
                m_club=0.5,
                L_hub=0.15,
                L_r_upper=0.35,
                L_r_fore=0.30,
                L_l_upper=0.35,
                L_l_fore=0.30,
                L_club=1.1,
                d_rs=0.20,
                d_ls=0.20,
                grip_right=0.05,
                grip_left=0.25,
            )

    def test_zero_mass_rejected(self) -> None:
        """Zero mass must raise AssertionError (solver requires positive mass)."""
        with pytest.raises((ValueError, TypeError), match="must be positive"):
            GolferParams(
                m_hub=0.0,
                m_r_upper=3.5,
                m_r_fore=2.0,
                m_l_upper=3.5,
                m_l_fore=2.0,
                m_club=0.5,
                L_hub=0.15,
                L_r_upper=0.35,
                L_r_fore=0.30,
                L_l_upper=0.35,
                L_l_fore=0.30,
                L_club=1.1,
                d_rs=0.20,
                d_ls=0.20,
                grip_right=0.05,
                grip_left=0.25,
            )

    def test_grip_beyond_club_rejected(self) -> None:
        """Grip position beyond club length must raise AssertionError."""
        with pytest.raises((ValueError, TypeError), match="grip_right must be"):
            GolferParams(
                m_hub=0.001,
                m_r_upper=3.5,
                m_r_fore=2.0,
                m_l_upper=3.5,
                m_l_fore=2.0,
                m_club=0.5,
                L_hub=0.15,
                L_r_upper=0.35,
                L_r_fore=0.30,
                L_l_upper=0.35,
                L_l_fore=0.30,
                L_club=1.1,
                d_rs=0.20,
                d_ls=0.20,
                grip_right=2.0,  # > L_club
                grip_left=0.25,
            )

    def test_frozen_dataclass(self, address_params: GolferParams) -> None:
        """GolferParams must be immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            address_params.m_hub = 5.0  # type: ignore[misc]


class TestForwardKinematicsTopology:
    """Forward kinematics should produce correct positions for the topology."""

    def test_hub_below_origin(self, address_params: GolferParams) -> None:
        """With zero angles, hub should be directly below origin."""
        from double_pendulum_golf.golfer_kinematics import forward_kinematics

        q = np.zeros(8)
        pos = forward_kinematics(q, address_params)
        hub = pos["hub"]
        # Hub should be below origin (negative y with downward-positive convention)
        assert hub[0] == pytest.approx(0.0, abs=1e-10), "Hub should be on y-axis"

    def test_positions_finite(self, address_params: GolferParams) -> None:
        """All FK positions must be finite numbers."""
        from double_pendulum_golf.golfer_kinematics import forward_kinematics

        q = np.zeros(8)
        pos = forward_kinematics(q, address_params)
        for name, xy in pos.items():
            assert np.all(np.isfinite(xy)), f"Position {name} has non-finite values: {xy}"

    def test_scapula_positions_present(self, address_params: GolferParams) -> None:
        """When scapula lengths are nonzero, scapula positions must be in FK."""
        from double_pendulum_golf.golfer_kinematics import forward_kinematics

        q = np.zeros(8)
        pos = forward_kinematics(q, address_params)
        assert "rscap" in pos, "Right scapula position missing from FK"
        assert "lscap" in pos, "Left scapula position missing from FK"

    def test_no_scapula_when_length_zero(self) -> None:
        """When scapula lengths are zero, scapula positions should be absent."""
        from double_pendulum_golf.golfer_kinematics import forward_kinematics

        params = GolferParams(
            m_hub=0.001,
            m_r_upper=3.5,
            m_r_fore=2.0,
            m_l_upper=3.5,
            m_l_fore=2.0,
            m_club=0.5,
            L_hub=0.15,
            L_r_upper=0.35,
            L_r_fore=0.30,
            L_l_upper=0.35,
            L_l_fore=0.30,
            L_club=1.1,
            d_rs=0.20,
            d_ls=0.20,
            grip_right=0.05,
            grip_left=0.25,
            L_rscap=0.0,
            L_lscap=0.0,
        )
        q = np.zeros(8)
        pos = forward_kinematics(q, params)
        assert "rscap" not in pos, "Scapula should be absent when length is 0"
        assert "lscap" not in pos, "Scapula should be absent when length is 0"
