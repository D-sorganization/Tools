"""Tests for golfer force/mobility ellipsoid computation.

TDD: These tests verify that ellipsoids_golfer returns the expected
structure and that the ellipsoid data is physically meaningful.
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.jacobians_golfer import ellipsoids_golfer
from double_pendulum_golf.physics_golfer import GolferParams


@pytest.fixture
def default_golfer_params() -> GolferParams:
    """Standard golfer parameters for testing."""
    return GolferParams(
        m_hub=5.0,
        m_r_upper=3.5,
        m_r_fore=2.0,
        m_l_upper=3.5,
        m_l_fore=2.0,
        m_club=0.5,
        L_hub=0.5,
        L_r_upper=0.3,
        L_r_fore=0.25,
        L_l_upper=0.3,
        L_l_fore=0.25,
        L_club=1.0,
        d_rs=0.15,
        d_ls=0.15,
        grip_right=0.3,
        grip_left=0.3,
        m_clubhead=0.2,
    )


class TestGolferEllipsoids:
    """Tests for ellipsoids_golfer function."""

    def test_returns_dict(self, default_golfer_params):
        q = np.zeros(8)
        result = ellipsoids_golfer(q, default_golfer_params)
        assert isinstance(result, dict)

    def test_expected_endpoint_keys(self, default_golfer_params):
        """Should return ellipsoid data for key golfer endpoints."""
        q = np.zeros(8)
        result = ellipsoids_golfer(q, default_golfer_params)
        # At least these endpoints should be present
        for key in ["rh", "lh", "club_tip"]:
            assert key in result, f"Missing key: {key}"

    def test_ellipsoid_structure(self, default_golfer_params):
        """Each endpoint should have directions, mob_semi_axes, etc."""
        q = np.zeros(8)
        result = ellipsoids_golfer(q, default_golfer_params)
        for name, ell in result.items():
            assert "directions" in ell, f"{name} missing 'directions'"
            assert "mob_semi_axes" in ell, f"{name} missing 'mob_semi_axes'"
            assert "force_semi_axes" in ell, f"{name} missing 'force_semi_axes'"
            assert "singular_values" in ell, f"{name} missing 'singular_values'"

    def test_directions_shape(self, default_golfer_params):
        """Directions should be (2, 2) for 2D ellipsoids."""
        q = np.zeros(8)
        result = ellipsoids_golfer(q, default_golfer_params)
        for name, ell in result.items():
            dirs = ell["directions"]
            assert dirs.shape == (2, 2), f"{name}: directions shape {dirs.shape}"

    def test_mob_semi_axes_positive(self, default_golfer_params):
        """Mobility semi-axes should be non-negative."""
        q = np.zeros(8)
        result = ellipsoids_golfer(q, default_golfer_params)
        for name, ell in result.items():
            mob = ell["mob_semi_axes"]
            assert mob.shape == (2,), f"{name}: mob_semi_axes shape {mob.shape}"
            assert np.all(mob >= 0), f"{name}: negative mob_semi_axes: {mob}"

    def test_nonzero_configuration(self, default_golfer_params):
        """Ellipsoids should be computable at non-zero configuration."""
        q = np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, 0.0])
        result = ellipsoids_golfer(q, default_golfer_params)
        assert len(result) > 0
        for name, ell in result.items():
            assert np.all(np.isfinite(ell["mob_semi_axes"])), (
                f"{name}: non-finite mob_semi_axes"
            )

    def test_handles_full_state_vector(self, default_golfer_params):
        """Should accept q with shape (16,) and use only first 8."""
        q_full = np.zeros(16)
        q_full[0] = 0.1
        result = ellipsoids_golfer(q_full, default_golfer_params)
        assert len(result) > 0
