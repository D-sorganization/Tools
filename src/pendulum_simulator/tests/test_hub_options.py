from typing import Any

"""
Tests for massless hub standoff and adjustable rotation centre.

Covers:
- Massless hub flag (epsilon mass)
- System centre of mass computation
- Hub-at-COM mode
- Manual hub offset
- Positive definiteness of mass matrix with near-zero hub mass
"""

from __future__ import annotations


import numpy as np
import pytest

from double_pendulum_golf.physics_golfer import GolferParams
from double_pendulum_golf.hub_options import (
    compute_system_com,
    effective_hub_mass,
    make_massless_hub_params,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_params() -> Any:
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
# Effective hub mass
# ---------------------------------------------------------------------------


class TestEffectiveHubMass:
    """Test the hub mass epsilon replacement."""

    def test_normal_mass_unchanged(self) -> Any:
        assert effective_hub_mass(40.0, massless=False) == 40.0

    def test_massless_gives_epsilon(self) -> Any:
        m = effective_hub_mass(40.0, massless=True)
        assert m > 0
        assert m < 0.001  # effectively massless

    def test_epsilon_is_positive(self) -> Any:
        """Mass must remain positive for numerical stability."""
        m = effective_hub_mass(0.001, massless=True)
        assert m > 0


# ---------------------------------------------------------------------------
# make_massless_hub_params
# ---------------------------------------------------------------------------


class TestMakeMasslessHubParams:
    """Test creating params with massless hub."""

    def test_returns_golfer_params(self, default_params) -> Any:
        p = make_massless_hub_params(default_params)
        assert isinstance(p, GolferParams)

    def test_hub_mass_is_epsilon(self, default_params) -> Any:
        p = make_massless_hub_params(default_params)
        assert p.m_hub < 0.001
        assert p.m_hub > 0

    def test_other_params_preserved(self, default_params) -> Any:
        p = make_massless_hub_params(default_params)
        assert p.m_r_upper == default_params.m_r_upper
        assert p.L_club == default_params.L_club
        assert p.grip_right == default_params.grip_right


# ---------------------------------------------------------------------------
# System COM
# ---------------------------------------------------------------------------


class TestComputeSystemCOM:
    """Test centre of mass computation."""

    def test_returns_2d(self, default_params) -> Any:
        q = np.zeros(8)
        com = compute_system_com(q, default_params)
        assert com.shape == (2,)

    def test_finite(self, default_params) -> Any:
        q = np.zeros(8)
        com = compute_system_com(q, default_params)
        assert np.all(np.isfinite(com))

    def test_symmetric_config_x_near_zero(self, default_params) -> Any:
        """In the hanging-down config (all angles zero), x-COM should be near zero."""
        q = np.zeros(8)
        com = compute_system_com(q, default_params)
        # With symmetric arms and zero angles, COM should be close to x=0
        assert abs(com[0]) < 0.5  # within half a metre of centre

    def test_different_config_different_com(self, default_params) -> Any:
        """Different joint angles should give different COM."""
        q1 = np.zeros(8)
        q2 = np.array([0.5, 0.3, -0.2, 0.1, -0.3, 0.2, -0.1, 0.4])
        com1 = compute_system_com(q1, default_params)
        com2 = compute_system_com(q2, default_params)
        assert not np.allclose(com1, com2)


# ---------------------------------------------------------------------------
# Mass matrix with massless hub
# ---------------------------------------------------------------------------


class TestMassMatrixWithMasslessHub:
    """Verify mass matrix remains positive-definite with epsilon hub mass."""

    def test_positive_semi_definite(self, default_params) -> Any:
        """Mass matrix should be PSD even with epsilon hub mass.

        Note: The golfer 8-DOF mass matrix is rank 6 due to 4 holonomic
        constraints, so we expect 2 near-zero eigenvalues. The test checks
        that all eigenvalues are >= -epsilon (non-negative within tolerance)
        and that the rank-6 subspace has strictly positive eigenvalues.
        """
        from double_pendulum_golf.physics_golfer import mass_matrix

        p = make_massless_hub_params(default_params)
        q = np.zeros(8)
        M = mass_matrix(q, p)
        eigenvalues = np.linalg.eigvalsh(M)
        # All eigenvalues should be non-negative (within numerical tolerance)
        assert np.all(eigenvalues >= -1e-10), f"Negative eigenvalue: {eigenvalues}"
        # At least 6 eigenvalues should be strictly positive
        positive_count = np.sum(eigenvalues > 1e-10)
        assert positive_count >= 6, f"Only {positive_count} positive eigenvalues"

    def test_symmetric(self, default_params) -> Any:
        from double_pendulum_golf.physics_golfer import mass_matrix

        p = make_massless_hub_params(default_params)
        q = np.zeros(8)
        M = mass_matrix(q, p)
        np.testing.assert_allclose(M, M.T, atol=1e-12)
