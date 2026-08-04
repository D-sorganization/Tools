"""Rust-accelerated swing dynamics façade (STRICT posture).

Mirrors the posture of :mod:`shared.python.signal_toolkit.bilateral_rust`:
the ``swing_core`` wheel is imported at module level; hot-loop entry points
raise ``ImportError`` with an actionable message at call time when the wheel
is missing. We deliberately do NOT silently substitute the ~100x slower
pure-Python path for the integration loop — that would mask deployment
misconfiguration. One-shot analysis calls may fall back explicitly via
:func:`shared.python.swing_sim.reference.simulate`.

PyO3 submodule import gotcha: ``swing_core`` exposes ``swing`` as a runtime
PyO3 submodule (an attribute, not a filesystem module), so we must use
``from swing_core import swing`` — ``import swing_core.swing`` fails.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .types import PendulumParameters, PendulumState

logger = logging.getLogger(__name__)

_MISSING_WHEEL_MESSAGE = (
    "swing_core Rust extension is not installed; the swing integration hot "
    "loop requires it. Build/install it with "
    "`pip install rust_core/swing-core` or "
    "`maturin develop -m rust_core/swing-core/Cargo.toml`. For one-shot "
    "analysis you may explicitly fall back to "
    "shared.python.swing_sim.reference.simulate."
)

try:
    # Runtime PyO3 submodule: attribute access on the parent, never
    # `import swing_core.swing`.
    from swing_core import swing as _rust_swing

    _RUST_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on machines without wheel
    _rust_swing = None
    _RUST_AVAILABLE = False
    logger.warning(
        "swing_sim._rust_facade: swing_core wheel not available; hot-loop "
        "calls will raise ImportError. See docs/development/rust-setup.md"
    )


def rust_available() -> bool:
    """Return whether the ``swing_core`` Rust wheel is importable."""
    return _RUST_AVAILABLE


def _require_rust() -> Any:
    if not _RUST_AVAILABLE:
        raise ImportError(_MISSING_WHEEL_MESSAGE)
    return _rust_swing


def _to_rust_params(p: PendulumParameters) -> Any:
    rust = _require_rust()
    return rust.PendulumParameters(
        p.m1, p.l1, p.lc1, p.i1, p.m2, p.l2, p.lc2, p.i2, p.d1, p.d2
    )


def _to_rust_state(s: PendulumState) -> Any:
    rust = _require_rust()
    return rust.PendulumState(s.theta1, s.theta2, s.omega1, s.omega2)


def plane_rotation_rust(yaw: float, side_tilt: float, fwd_tilt: float) -> np.ndarray:
    """World-from-plane rotation matrix (3x3) — Rust path."""
    rust = _require_rust()
    return np.asarray(
        rust.plane_rotation(float(yaw), float(side_tilt), float(fwd_tilt)),
        dtype=np.float64,
    )


def in_plane_gravity_rust(
    yaw: float, side_tilt: float, fwd_tilt: float, g: float
) -> tuple[float, float]:
    """In-plane gravity ``(g_x, g_y)`` from tilt angles — Rust path."""
    rust = _require_rust()
    gx, gy = rust.in_plane_gravity(
        float(yaw), float(side_tilt), float(fwd_tilt), float(g)
    )
    return float(gx), float(gy)


def step_rust(
    params: PendulumParameters,
    state: PendulumState,
    g_inplane: tuple[float, float],
    dt: float,
) -> PendulumState:
    """One RK4 step — Rust path.

    Raises:
        ImportError: If the ``swing_core`` wheel is not installed.
        ValueError: If the mass matrix is numerically singular.
    """
    rust = _require_rust()
    out = rust.step(
        _to_rust_params(params),
        _to_rust_state(state),
        float(g_inplane[0]),
        float(g_inplane[1]),
        float(dt),
    )
    return PendulumState(
        theta1=out.theta1, theta2=out.theta2, omega1=out.omega1, omega2=out.omega2
    )


def simulate_rust(
    params: PendulumParameters,
    initial: PendulumState,
    g_inplane: tuple[float, float],
    dt: float,
    n_steps: int,
) -> np.ndarray:
    """Simulate ``n_steps`` RK4 steps — Rust path (hot loop).

    Returns an ``(n_steps + 1, 4)`` array of rows
    ``[theta1, theta2, omega1, omega2]`` including the initial state.

    Raises:
        ImportError: If the ``swing_core`` wheel is not installed.
        ValueError: If any step encounters a singular mass matrix.
    """
    rust = _require_rust()
    flat = rust.simulate(
        _to_rust_params(params),
        _to_rust_state(initial),
        float(g_inplane[0]),
        float(g_inplane[1]),
        float(dt),
        int(n_steps),
    )
    return np.asarray(flat, dtype=np.float64).reshape(int(n_steps) + 1, 4)


def total_energy_rust(
    params: PendulumParameters,
    state: PendulumState,
    g_inplane: tuple[float, float],
) -> float:
    """Total mechanical energy [J] — Rust path."""
    rust = _require_rust()
    return float(
        rust.total_energy(
            _to_rust_params(params),
            _to_rust_state(state),
            float(g_inplane[0]),
            float(g_inplane[1]),
        )
    )
