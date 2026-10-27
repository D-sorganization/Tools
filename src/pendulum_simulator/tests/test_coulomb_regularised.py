"""Tests for regularised Coulomb friction in double pendulum dynamics (#5804, TOOLS-18).

Verifies that Coulomb friction is regularised with tanh(qdot / v_eps),
ensuring continuity near zero velocity and bounded RK45 integrator step counts
through velocity reversals.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from double_pendulum_golf.physics import (
    PendulumParams,
    equations_of_motion,
    friction_torque_vector,
)


@pytest.fixture
def coulomb_params() -> PendulumParams:
    """Fixture providing double pendulum parameters with pure Coulomb friction."""
    return PendulumParams(
        m1=5.0,
        m2=0.3,
        L1=0.65,
        L2=1.1,
        b1=0.0,
        b2=0.0,
        mu1=0.5,
        mu2=0.2,
    )


class TestCoulombRegularised:
    """TOOLS-18: Regularised Coulomb friction continuity and solver boundedness."""

    def test_coulomb_friction_continuous_at_zero(self, coulomb_params: PendulumParams) -> None:
        """TOOLS-18: Coulomb friction must be continuous at zero velocity.

        Under discontinuous sgn(qdot), tau jumps to -mu for any epsilon > 0,
        exhibiting a jump discontinuity of 2*mu across zero.
        Under tanh(qdot / v_eps), tau(epsilon) -> 0 continuously as epsilon -> 0.
        """
        eps = 1e-6  # Well below default v_eps = 1e-4
        tau_small = friction_torque_vector(eps, eps, coulomb_params)
        tau_zero = friction_torque_vector(0.0, 0.0, coulomb_params)

        # On main with sgn, tau_small[0] is -0.5 (|jump| = 0.5 = mu1)
        # With regularised tanh, |tau_small[0]| ~ mu1 * tanh(0.01) ~ 0.005
        assert abs(tau_small[0] - tau_zero[0]) < 0.05 * coulomb_params.mu1, (
            f"Friction discontinuous at zero: jump {abs(tau_small[0] - tau_zero[0])} "
            f"exceeds 5% of mu1={coulomb_params.mu1}"
        )
        assert abs(tau_small[1] - tau_zero[1]) < 0.05 * coulomb_params.mu2, (
            f"Friction discontinuous at zero: jump {abs(tau_small[1] - tau_zero[1])} "
            f"exceeds 5% of mu2={coulomb_params.mu2}"
        )

    def test_coulomb_regularised_rk45_step_count_bounded(
        self, coulomb_params: PendulumParams
    ) -> None:
        """TOOLS-18: RK45 integration step count must remain bounded through reversals.

        Under discontinuous sgn(qdot), the jump at zero velocity causes step-size
        collapse and chattering, requiring >3000 function evaluations over 1.0s.
        With regularised tanh(qdot / v_eps), the RHS is smooth and
        RK45 step count stays bounded with nfev < 1500.
        """
        # Initial displacement that swings through multiple velocity reversals
        y0 = np.array([0.5, 0.2, 0.0, 0.0])

        def zero_torque(_t: float) -> tuple[float, float]:
            return 0.0, 0.0

        sol = solve_ivp(
            lambda t, y: equations_of_motion(y, t, coulomb_params, zero_torque),
            (0.0, 1.0),
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )

        assert sol.success, f"Integration failed: {sol.message}"
        # Verify that velocity reversals occurred
        dtheta1 = sol.y[2]
        assert np.any(dtheta1 > 0.05) and np.any(dtheta1 < -0.05), (
            "Trajectory did not undergo velocity reversals"
        )
        # Step evaluations must remain bounded
        assert sol.nfev < 1500, (
            f"RK45 step evaluations ({sol.nfev}) not bounded (expected < 1500; "
            "discontinuous sgn causes collapse to >3000 evaluations)"
        )

    def test_coulomb_magnitude_bounded_by_mu_contract(
        self, coulomb_params: PendulumParams
    ) -> None:
        """DbC contract: |tau_coulomb| <= mu for all velocities."""
        velocities = np.linspace(-50.0, 50.0, 101)
        for v in velocities:
            tf = friction_torque_vector(float(v), float(v), coulomb_params)
            assert abs(tf[0]) <= coulomb_params.mu1 + 1e-12, (
                f"|tau_f1|={abs(tf[0])} exceeded mu1={coulomb_params.mu1} at v={v}"
            )
            assert abs(tf[1]) <= coulomb_params.mu2 + 1e-12, (
                f"|tau_f2|={abs(tf[1])} exceeded mu2={coulomb_params.mu2} at v={v}"
            )

    def test_coulomb_preserves_full_magnitude_away_from_zero(
        self, coulomb_params: PendulumParams
    ) -> None:
        """Coulomb friction preserves full mu magnitude for |v| >> v_eps."""
        for speed in [0.01, 0.1, 1.0, 10.0]:
            tf = friction_torque_vector(speed, -speed, coulomb_params)
            assert np.isclose(tf[0], -coulomb_params.mu1, rtol=1e-3)
            assert np.isclose(tf[1], coulomb_params.mu2, rtol=1e-3)
