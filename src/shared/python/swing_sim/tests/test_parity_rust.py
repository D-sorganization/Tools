"""Rust <-> Python parity tests for the swing dynamics kernel.

The pure-Python :mod:`shared.python.swing_sim.reference` module is the
oracle; the ``swing_core`` wheel must reproduce it to floating-point
accumulation tolerance. Skipped cleanly when the wheel is absent (same
posture as ``signal_toolkit/tests/test_bilateral_rust_parity.py``).
"""

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

pytest.importorskip(
    "swing_core",
    reason="Rust swing_core wheel not installed on this interpreter",
)
# Imported after the skip so machines without the wheel can still collect.
from shared.python.swing_sim import _rust_facade  # noqa: E402

# Both sides evaluate the same formulas in the same order; differences come
# only from double-precision accumulation across RK4 stages.
_PARITY_TOL = 1e-9


@pytest.mark.parity
@pytest.mark.unit
def test_plane_rotation_parity() -> None:
    for yaw, side, fwd in ((0.0, 0.0, 0.0), (1.2, -0.6, 0.35), (-2.0, 1.1, 2.9)):
        py_r = reference.plane_rotation(yaw, side, fwd)
        rust_r = _rust_facade.plane_rotation_rust(yaw, side, fwd)
        np.testing.assert_allclose(rust_r, py_r, atol=1e-14)


@pytest.mark.parity
@pytest.mark.unit
def test_in_plane_gravity_parity() -> None:
    for yaw, side, fwd in ((0.0, 0.0, 0.0), (0.5, 0.9, -0.3), (-2.0, 1.4, 1.1)):
        py_g = reference.in_plane_gravity_from_tilts(yaw, side, fwd, G)
        rust_g = _rust_facade.in_plane_gravity_rust(yaw, side, fwd, G)
        assert rust_g[0] == pytest.approx(py_g[0], abs=1e-14)
        assert rust_g[1] == pytest.approx(py_g[1], abs=1e-14)


@pytest.mark.parity
@pytest.mark.unit
def test_single_step_parity() -> None:
    p = PendulumParameters.golf_default()
    state = PendulumState(theta1=1.2, theta2=-0.5, omega1=0.3, omega2=-0.1)
    g = reference.in_plane_gravity_from_tilts(0.4, 0.6, -0.2, G)
    py_next = reference.rk4_step(p, state, g, 1e-3)
    rust_next = _rust_facade.step_rust(p, state, g, 1e-3)
    assert rust_next.theta1 == pytest.approx(py_next.theta1, abs=_PARITY_TOL)
    assert rust_next.theta2 == pytest.approx(py_next.theta2, abs=_PARITY_TOL)
    assert rust_next.omega1 == pytest.approx(py_next.omega1, abs=_PARITY_TOL)
    assert rust_next.omega2 == pytest.approx(py_next.omega2, abs=_PARITY_TOL)


@pytest.mark.parity
def test_trajectory_parity_500_steps() -> None:
    p = PendulumParameters.golf_default()
    initial = PendulumState(theta1=1.5, theta2=-0.8, omega1=0.0, omega2=0.0)
    g = reference.in_plane_gravity_from_tilts(0.3, 0.7, 0.1, G)
    dt, n_steps = 1e-3, 500

    py_states = reference.simulate(p, initial, g, dt, n_steps)
    rust_states = _rust_facade.simulate_rust(p, initial, g, dt, n_steps)

    assert rust_states.shape == py_states.shape == (n_steps + 1, 4)
    max_diff = float(np.abs(py_states - rust_states).max())
    assert max_diff < _PARITY_TOL, f"trajectory diverged: max diff {max_diff}"


@pytest.mark.parity
@pytest.mark.unit
def test_total_energy_parity() -> None:
    p = PendulumParameters.golf_default()
    state = PendulumState(theta1=0.9, theta2=0.4, omega1=1.1, omega2=-0.6)
    g = (0.0, -G)
    py_e = reference.total_energy(p, state, g)
    rust_e = _rust_facade.total_energy_rust(p, state, g)
    assert rust_e == pytest.approx(py_e, abs=1e-10)
