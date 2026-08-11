# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Rust <-> NumPy parity for the batch COM-x hot path (issue #3518).

When the optional ``movement_optimizer_core`` maturin extension is installed,
``LagrangianKinematicsMixin.com_x_batch`` runs in Rust; otherwise a NumPy
fallback runs. COM-x sits on the optimizer cost hot path, so the two paths must
produce identical results regardless of whether the extension is present.

The Rust ``com_x_batch_rs`` previously placed the squat bar at the shoulder
(``bar_mass * shoulder_x``) while the NumPy path applies a bar offset when
``squat_bar_height``/``squat_bar_depth`` are nonzero. This module asserts parity
for BOTH a deadlift case AND a squat case with a nonzero bar offset, so that
divergence cannot reappear.

The whole module is skipped when the extension is unavailable (e.g. local dev
machines that never ran ``maturin develop``). CI builds the extension and runs
this as a required gate, so the parity contract is enforced where it matters.
"""

from __future__ import annotations

import numpy as np
import pytest

from movement_optimizer.models import BodyModel
from movement_optimizer.models.lagrangian_dynamics import LagrangianDynamics

# Skip the entire module unless the compiled Rust accelerator is importable.
rust = pytest.importorskip(
    "movement_optimizer_core",
    reason="Rust accelerator (movement_optimizer_core) not built; run `maturin develop --release`.",
)


def _make_squat_dynamics_with_offset() -> LagrangianDynamics:
    """Squat dynamics whose body has a nonzero bar offset (depth + height)."""
    body = BodyModel(75.0, 1.75, squat_bar_depth=0.05, squat_bar_height=0.03)
    return LagrangianDynamics(body, body.m_squat.copy(), body.I_squat.copy(), 60.0)


def _make_deadlift_dynamics() -> LagrangianDynamics:
    """Deadlift dynamics (arm mass folded into the load, no bar offset)."""
    body = BodyModel(75.0, 1.75)
    load = body.m_arms + 100.0
    return LagrangianDynamics(
        body, body.m_deadlift.copy(), body.I_deadlift.copy(), load
    )


def _random_q(rng: np.random.Generator, n: int) -> np.ndarray:
    """Finite, physically-plausible joint-angle batch of shape (n, 3)."""
    return rng.uniform(-1.5, 1.5, size=(n, 3))


@pytest.mark.parity
@pytest.mark.parametrize("n", [1, 8, 256])
def test_rust_matches_numpy_com_x_squat_with_offset(n: int) -> None:
    """Rust ``com_x_batch_rs`` matches the NumPy reference for a squat with a bar offset.

    Verifies the squat bar-offset formula
    ``shoulder_x - squat_bar_height*sin(q2) - squat_bar_depth*cos(q2)`` was
    threaded into the Rust kernel; otherwise the squat branch silently diverges.
    """
    dyn = _make_squat_dynamics_with_offset()
    b = dyn.body
    assert b.squat_bar_depth != 0.0 and b.squat_bar_height != 0.0
    rng = np.random.default_rng(35180 + n)
    q = _random_q(rng, n)
    bar_mass = 60.0

    numpy_com_x = dyn._numpy_com_x_batch(q, "squat", bar_mass)
    rust_com_x = np.asarray(
        rust.com_x_batch_rs(
            np.ascontiguousarray(q, dtype=np.float64),
            float(dyn.L_eff[0]),
            float(dyn.L_eff[1]),
            float(dyn.L_eff[2]),
            float(dyn.d_eff[0]),
            float(dyn.d_eff[1]),
            float(dyn.d_eff[2]),
            float(dyn.m[0]),
            float(dyn.m[1]),
            float(dyn.m[2]),
            float(b.m_feet),
            float(b.foot_com_x),
            float(bar_mass),
            float(b.body_mass),
            True,
            float(b.m_arms),
            float(b.squat_bar_height),
            float(b.squat_bar_depth),
        )
    )

    assert rust_com_x.shape == numpy_com_x.shape == (n,)
    assert np.all(np.isfinite(rust_com_x))
    np.testing.assert_allclose(rust_com_x, numpy_com_x, rtol=1e-9, atol=1e-9)


@pytest.mark.parity
@pytest.mark.parametrize("n", [1, 8, 256])
def test_rust_matches_numpy_com_x_deadlift(n: int) -> None:
    """Rust ``com_x_batch_rs`` matches the NumPy reference for a deadlift."""
    dyn = _make_deadlift_dynamics()
    b = dyn.body
    rng = np.random.default_rng(99100 + n)
    q = _random_q(rng, n)
    bar_mass = 100.0

    numpy_com_x = dyn._numpy_com_x_batch(q, "deadlift", bar_mass)
    rust_com_x = np.asarray(
        rust.com_x_batch_rs(
            np.ascontiguousarray(q, dtype=np.float64),
            float(dyn.L_eff[0]),
            float(dyn.L_eff[1]),
            float(dyn.L_eff[2]),
            float(dyn.d_eff[0]),
            float(dyn.d_eff[1]),
            float(dyn.d_eff[2]),
            float(dyn.m[0]),
            float(dyn.m[1]),
            float(dyn.m[2]),
            float(b.m_feet),
            float(b.foot_com_x),
            float(bar_mass),
            float(b.body_mass),
            False,
            float(b.m_arms),
            float(getattr(b, "squat_bar_height", 0.0)),
            float(getattr(b, "squat_bar_depth", 0.0)),
        )
    )

    assert rust_com_x.shape == numpy_com_x.shape == (n,)
    assert np.all(np.isfinite(rust_com_x))
    np.testing.assert_allclose(rust_com_x, numpy_com_x, rtol=1e-9, atol=1e-9)


@pytest.mark.parity
def test_public_method_uses_rust_for_squat_offset() -> None:
    """With the extension installed, ``com_x_batch`` equals the Rust kernel.

    Guards against a regression where ``com_x_batch`` stops dispatching to Rust
    and silently degrades to NumPy on a host that paid to build the accelerator.
    """
    dyn = _make_squat_dynamics_with_offset()
    b = dyn.body
    rng = np.random.default_rng(7)
    q = _random_q(rng, 16)
    bar_mass = 60.0

    public = np.asarray(dyn.com_x_batch(q, "squat", bar_mass))
    rust_direct = np.asarray(
        rust.com_x_batch_rs(
            np.ascontiguousarray(q, dtype=np.float64),
            float(dyn.L_eff[0]),
            float(dyn.L_eff[1]),
            float(dyn.L_eff[2]),
            float(dyn.d_eff[0]),
            float(dyn.d_eff[1]),
            float(dyn.d_eff[2]),
            float(dyn.m[0]),
            float(dyn.m[1]),
            float(dyn.m[2]),
            float(b.m_feet),
            float(b.foot_com_x),
            float(bar_mass),
            float(b.body_mass),
            True,
            float(b.m_arms),
            float(b.squat_bar_height),
            float(b.squat_bar_depth),
        )
    )
    np.testing.assert_allclose(public, rust_direct, rtol=1e-12, atol=0.0)
