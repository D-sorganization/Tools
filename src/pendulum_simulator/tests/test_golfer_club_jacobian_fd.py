"""Central-difference consistency of the golfer club Jacobians (refs #5626).

Every Jacobian returned by ``analytical_fk_jacobians`` (NumPy) and
``analytical_fk_jacobians_jax`` must equal the derivative of the matching
``forward_kinematics`` point.  ``club_com`` is the midpoint of ``club_base`` and
``club_tip``.

Tolerance (error analysis): with step h = 1e-6, central-difference truncation is
(h^2/6)|f'''| <= ~2e-13 for link lengths <= 1.1 m, and round-off is
eps*|f|/h ~ 2.2e-16 * 2 / 1e-6 ~ 4e-10.  FD_TOL = 1e-8 leaves a >20x margin and
still catches O(0.1) analytic errors such as a flipped sign.
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf import native_backend as _native_backend
from double_pendulum_golf.physics_golfer import (
    N_DOF,
    GolferParams,
    analytical_fk_jacobians,
    forward_kinematics,
)

FD_STEP = 1e-6
FD_TOL = 1e-8

CONFIGS = [
    np.array([0.3, -0.5, 0.8, 0.2, 0.4, 0.7, -0.3, 1.1]),
    np.array([-0.7, 0.9, -0.4, -0.6, -0.2, 1.2, 0.5, -0.8]),
    np.array([1.2, 0.1, 0.3, 0.9, -1.0, -0.6, 0.7, 2.3]),
    np.array([-0.2, -1.1, 1.0, -0.4, 0.8, 0.3, 0.1, 0.4]),
]
# GolferParams rejects grip_right < 0, so span the sign of the COM coefficient
# (L_club/2 - grip_right) instead: 0.6 > L_club/2 = 0.55 flips it negative.
GRIP_RIGHTS = [0.3, 0.0, 0.6]
_JAX_FIELDS = (
    "m_hub",
    "m_r_upper",
    "m_r_fore",
    "m_l_upper",
    "m_l_fore",
    "m_club",
    "L_hub",
    "L_r_upper",
    "L_r_fore",
    "L_l_upper",
    "L_l_fore",
    "L_club",
    "d_rs",
    "d_ls",
    "grip_right",
    "grip_left",
    "m_clubhead",
)
POINTS = ["hub", "re", "rh", "le", "lh", "club_com", "club_tip"]


def _params(grip_right: float) -> GolferParams:
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
        L_l_upper=0.33,
        L_l_fore=0.27,
        L_club=1.1,
        d_rs=0.20,
        d_ls=0.18,
        grip_right=grip_right,
        grip_left=0.1,
        m_clubhead=0.2,
    )


def _point(name: str, q: np.ndarray, p: GolferParams) -> np.ndarray:
    fk = forward_kinematics(q, p)
    if name == "club_com":
        return 0.5 * (np.array(fk["club_base"]) + np.array(fk["club_tip"]))
    return np.array(fk[name], dtype=float)


def _fd_column(name: str, q: np.ndarray, p: GolferParams, col: int) -> np.ndarray:
    qp, qm = q.copy(), q.copy()
    qp[col] += FD_STEP
    qm[col] -= FD_STEP
    return (_point(name, qp, p) - _point(name, qm, p)) / (2.0 * FD_STEP)


@pytest.fixture(autouse=True)
def _pure_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the Python forward kinematics so FD uses the code under test."""
    monkeypatch.setenv("PENDULUM_GOLFER_BACKEND", "python")


@pytest.mark.parametrize("grip_right", GRIP_RIGHTS)
@pytest.mark.parametrize("q", CONFIGS)
@pytest.mark.parametrize("name", POINTS)
def test_numpy_jacobian_matches_central_difference(
    name: str, q: np.ndarray, grip_right: float
) -> None:
    p = _params(grip_right)
    jac = analytical_fk_jacobians(q, p)[name]
    for col in range(N_DOF):
        fd = _fd_column(name, q, p, col)
        np.testing.assert_allclose(
            jac[:, col], fd, atol=FD_TOL, rtol=0.0, err_msg=f"{name} col {col}"
        )


@pytest.mark.parametrize("grip_right", GRIP_RIGHTS)
@pytest.mark.parametrize("q", CONFIGS)
def test_jax_jacobian_matches_central_difference(q: np.ndarray, grip_right: float) -> None:
    jax = pytest.importorskip("jax")
    if not hasattr(jax, "config"):
        pytest.skip("jax is replaced by a stub in this session (test_simulation_golfer_gpu)")
    jax.config.update("jax_enable_x64", True)
    from double_pendulum_golf.physics_golfer_jax import (
        GolferParamsJAX,
        analytical_fk_jacobians_jax,
    )

    p = _params(grip_right)
    pj = GolferParamsJAX(**{f: getattr(p, f) for f in _JAX_FIELDS})
    jac = analytical_fk_jacobians_jax(jax.numpy.asarray(q), pj)
    for name in ["hub", "re", "rh", "le", "lh", "club_com", "club_tip"]:
        for col in range(N_DOF):
            np.testing.assert_allclose(
                np.asarray(jac[name])[:, col],
                _fd_column(name, q, p, col),
                atol=FD_TOL,
                rtol=0.0,
                err_msg=f"jax {name} col {col}",
            )


@pytest.mark.skipif(
    not _native_backend.golfer_native_available(),
    reason="pendulum_core native extension not built; Rust parity unavailable",
)
@pytest.mark.parametrize("grip_right", GRIP_RIGHTS)
@pytest.mark.parametrize("q", CONFIGS)
def test_python_club_points_match_rust_forward_kinematics(
    q: np.ndarray, grip_right: float
) -> None:
    p = _params(grip_right)
    native = _native_backend.golfer_forward_kinematics(q, p)
    assert native is not None
    for key in ("club_base", "club_tip"):
        py = np.array(forward_kinematics(q, p)[key])
        np.testing.assert_allclose(py, np.array(native[key]), atol=1e-9)


@pytest.mark.parametrize("grip_right", GRIP_RIGHTS)
@pytest.mark.parametrize("q", CONFIGS)
def test_club_points_follow_shaft_direction(q: np.ndarray, grip_right: float) -> None:
    """Club points are ``rh + s * (sin, -cos)(theta_club)``, s from the base.

    This is the direction used by ``constraint_vector`` rows 2-3 and by the Rust
    kernel; the Python FK used to flip the y sign (cos instead of -cos).
    """
    p = _params(grip_right)
    fk = forward_kinematics(q, p)
    rh = np.array(fk["rh"])
    u = np.array([np.sin(q[7]), -np.cos(q[7])])
    np.testing.assert_allclose(fk["club_base"], rh - grip_right * u, atol=1e-12)
    np.testing.assert_allclose(fk["club_tip"], rh + (p.L_club - grip_right) * u, atol=1e-12)
    np.testing.assert_allclose(
        fk["grip_left"], rh + (p.grip_left - grip_right) * u, atol=1e-12
    )
