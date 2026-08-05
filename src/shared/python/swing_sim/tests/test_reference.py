"""Unit tests for the pure-Python reference dynamics (parity oracle)."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.swing_sim import reference
from shared.python.swing_sim.types import (
    DEFAULT_GRAVITY_M_S2 as G,
)
from shared.python.swing_sim.types import (
    PendulumParameters,
    PendulumState,
)


def _undamped() -> PendulumParameters:
    p = PendulumParameters.golf_default()
    return PendulumParameters(
        m1=p.m1,
        l1=p.l1,
        lc1=p.lc1,
        i1=p.i1,
        m2=p.m2,
        l2=p.l2,
        lc2=p.lc2,
        i2=p.i2,
        d1=0.0,
        d2=0.0,
    )


@pytest.mark.unit
class TestPlaneRotation:
    def test_identity_pose(self) -> None:
        r = reference.plane_rotation(0.0, 0.0, 0.0)
        np.testing.assert_allclose(r, np.eye(3), atol=1e-15)

    def test_orthonormal_for_generic_tilts(self) -> None:
        r = reference.plane_rotation(1.2, -0.6, 0.35)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-12)

    def test_flat_plane_full_gravity_down(self) -> None:
        gx, gy = reference.in_plane_gravity_from_tilts(0.0, 0.0, 0.0, G)
        assert gx == pytest.approx(0.0, abs=1e-12)
        assert gy == pytest.approx(-G, abs=1e-12)

    def test_pure_yaw_is_invariant(self) -> None:
        for yaw in (-3.0, -1.2, 0.4, 1.7, 3.1):
            gx, gy = reference.in_plane_gravity_from_tilts(yaw, 0.0, 0.0, G)
            assert gx == pytest.approx(0.0, abs=1e-12)
            assert gy == pytest.approx(-G, abs=1e-12)

    def test_side_tilt_cosine_projection(self) -> None:
        # Matches UpstreamDrift's scalar projected_gravity = g·cos(inclination).
        for side in (0.2, 0.6, 1.0):
            _, gy = reference.in_plane_gravity_from_tilts(0.0, side, 0.0, G)
            assert gy == pytest.approx(-G * np.cos(side), abs=1e-12)

    def test_projection_never_amplifies(self) -> None:
        for yaw, side, fwd in ((0.5, 0.9, -0.3), (-2.0, 1.4, 1.1), (3.0, -0.7, 0.6)):
            gx, gy = reference.in_plane_gravity_from_tilts(yaw, side, fwd, G)
            assert gx * gx + gy * gy <= G * G + 1e-9


@pytest.mark.unit
class TestDynamics:
    def test_mass_matrix_symmetric_positive_definite(self) -> None:
        p = PendulumParameters.golf_default()
        for theta2 in (-3.0, -1.5, 0.0, 0.8, 1.6, 3.1):
            m = reference.mass_matrix(p, theta2)
            assert m[0, 1] == pytest.approx(m[1, 0], abs=1e-15)
            eigenvalues = np.linalg.eigvalsh(m)
            assert np.all(eigenvalues > 0.0)

    def test_gravity_vector_flat_plane_matches_scalar_reference(self) -> None:
        p = PendulumParameters.golf_default()
        theta1, theta2 = 0.7, -0.4
        g1, g2 = reference.gravity_vector(p, theta1, theta2, (0.0, -G))
        expected_g1 = (p.m1 * p.lc1 + p.m2 * p.l1) * G * np.sin(theta1) + (
            p.m2 * p.lc2 * G * np.sin(theta1 + theta2)
        )
        expected_g2 = p.m2 * p.lc2 * G * np.sin(theta1 + theta2)
        assert g1 == pytest.approx(expected_g1, abs=1e-12)
        assert g2 == pytest.approx(expected_g2, abs=1e-12)

    def test_undamped_energy_conserved_over_1000_steps(self) -> None:
        p = _undamped()
        g = reference.in_plane_gravity_from_tilts(0.4, 0.6, -0.2, G)
        initial = PendulumState(theta1=1.2, theta2=-0.5, omega1=0.0, omega2=0.0)
        states = reference.simulate(p, initial, g, 1e-4, 1000)
        e0 = reference.total_energy(p, initial, g)
        scale = max(abs(e0), 1.0)
        for row in states:
            s = PendulumState(
                theta1=row[0], theta2=row[1], omega1=row[2], omega2=row[3]
            )
            assert abs(reference.total_energy(p, s, g) - e0) / scale < 1e-6

    def test_damping_dissipates_energy(self) -> None:
        p = PendulumParameters.golf_default()
        g = (0.0, -G)
        initial = PendulumState(theta1=1.0, theta2=0.3, omega1=0.5, omega2=-0.2)
        states = reference.simulate(p, initial, g, 1e-3, 2000)
        final = PendulumState(
            theta1=states[-1, 0],
            theta2=states[-1, 1],
            omega1=states[-1, 2],
            omega2=states[-1, 3],
        )
        assert reference.total_energy(p, final, g) < reference.total_energy(
            p, initial, g
        )

    def test_simulate_shape_and_initial_row(self) -> None:
        p = PendulumParameters.golf_default()
        initial = PendulumState(theta1=0.1, theta2=0.0, omega1=0.0, omega2=0.0)
        states = reference.simulate(p, initial, (0.0, -G), 1e-3, 10)
        assert states.shape == (11, 4)
        np.testing.assert_allclose(states[0], (0.1, 0.0, 0.0, 0.0))

    def test_rk4_rejects_nonpositive_dt(self) -> None:
        p = PendulumParameters.golf_default()
        s = PendulumState(theta1=0.1, theta2=0.0, omega1=0.0, omega2=0.0)
        with pytest.raises(ValueError, match="dt"):
            reference.rk4_step(p, s, (0.0, -G), 0.0)

    def test_forced_derivatives_add_generalized_joint_torques(self) -> None:
        p = PendulumParameters.golf_default()
        state = PendulumState(theta1=0.2, theta2=-0.1, omega1=0.3, omega2=-0.2)
        g_inplane = (0.0, -G)
        passive = reference.derivatives(p, state, g_inplane)
        forced = reference.derivatives_forced(p, state, g_inplane, (4.0, -1.5))
        expected_delta = np.linalg.solve(
            reference.mass_matrix(p, state.theta2), [4.0, -1.5]
        )
        np.testing.assert_allclose(forced[:2], passive[:2], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            np.asarray(forced[2:]) - np.asarray(passive[2:]),
            expected_delta,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_zero_forced_simulation_matches_passive_reference(self) -> None:
        p = PendulumParameters.golf_default()
        state = PendulumState(theta1=0.5, theta2=-0.2, omega1=0.1, omega2=0.0)
        g_inplane = (0.0, -G)
        passive = reference.simulate(p, state, g_inplane, dt=0.001, n_steps=20)
        forced = reference.simulate_forced(
            p,
            state,
            g_inplane,
            dt=0.001,
            n_steps=20,
            torque_at=lambda _time_s: (0.0, 0.0),
        )
        np.testing.assert_allclose(forced, passive, rtol=0.0, atol=0.0)
