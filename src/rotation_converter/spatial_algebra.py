"""Featherstone Spatial Algebra and Dynamics Algorithms.

Implements spatial vectors, cross-products, transforms, and algorithms
like Recursive Newton-Euler (ID) and Articulated-Body (FDab).
Ported strictly from Roy Featherstone's spatial_v1 MATLAB library.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from rotation_converter._contracts import require


@dataclass
class SpatialModel:
    """Robot model definition following Featherstone v1 format."""

    NB: int
    parent: list[int]  # 0 = world, 1 to NB = bodies
    pitch: list[float]  # 0 = revolute, np.inf = prismatic
    Xtree: list[np.ndarray]  # 6x6 transform from parent to this link
    I: list[np.ndarray]  # noqa: E741  # noqa: E741


def _validate_spatial_vector(v: np.ndarray, name: str) -> np.ndarray:
    assert v is not None, "v must be provided"
    v = np.asarray(v, dtype=float).flatten()
    require(v.shape == (6,), f"{name} must be a 6-vector [omega; v]")
    return v


def crm(v: Any) -> np.ndarray:
    """Spatial cross-product operator (motion)."""
    v = _validate_spatial_vector(v, "v")
    # v = [w1 w2 w3 v1 v2 v3]
    return np.array(
        [
            [0.0, -v[2], v[1], 0.0, 0.0, 0.0],
            [v[2], 0.0, -v[0], 0.0, 0.0, 0.0],
            [-v[1], v[0], 0.0, 0.0, 0.0, 0.0],
            [0.0, -v[5], v[4], 0.0, -v[2], v[1]],
            [v[5], 0.0, -v[3], v[2], 0.0, -v[0]],
            [-v[4], v[3], 0.0, -v[1], v[0], 0.0],
        ],
        dtype=float,
    )


def crf(v: Any) -> np.ndarray:
    """Spatial cross-product operator (force)."""
    # Simply -crm(v).T
    v = _validate_spatial_vector(v, "v")
    return -crm(v).T


def Xtrans(r: Any) -> np.ndarray:
    """Spatial coordinate transform (translation of origin by 3D vector)."""
    r = np.asarray(r, dtype=float).flatten()
    require(r.shape == (3,), "r must be a 3-vector")

    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, r[2], -r[1], 1.0, 0.0, 0.0],
            [-r[2], 0.0, r[0], 0.0, 1.0, 0.0],
            [r[1], -r[0], 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def Xrotx(theta: float) -> np.ndarray:
    """Spatial coordinate transform (X-axis rotation)."""
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, c, s, 0.0, 0.0, 0.0],
            [0.0, -s, c, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, c, s],
            [0.0, 0.0, 0.0, 0.0, -s, c],
        ],
        dtype=float,
    )


def Xroty(theta: float) -> np.ndarray:
    """Spatial coordinate transform (Y-axis rotation)."""
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [
            [c, 0.0, -s, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [s, 0.0, c, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, c, 0.0, -s],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, s, 0.0, c],
        ],
        dtype=float,
    )


def Xrotz(theta: float) -> np.ndarray:
    """Spatial coordinate transform (Z-axis rotation)."""
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [
            [c, s, 0.0, 0.0, 0.0, 0.0],
            [-s, c, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, c, s, 0.0],
            [0.0, 0.0, 0.0, -s, c, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def jcalc(pitch: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Calculate joint transform and motion subspace.

    Args:
        pitch: 0 for revolute, np.inf for prismatic, else helical
        q: Joint angle (or displacement for prismatic)

    Returns:
        Xj: 6x6 spatial transform
        S: 6-vector motion subspace
    """
    assert pitch is not None, "pitch must be provided"
    if pitch == 0.0:
        Xj = Xrotz(q)
        S = np.array([0, 0, 1, 0, 0, 0], dtype=float)
    elif math.isinf(pitch):
        Xj = Xtrans([0.0, 0.0, q])
        S = np.array([0, 0, 0, 0, 0, 1], dtype=float)
    else:
        Xj = Xrotz(q) @ Xtrans([0.0, 0.0, q * pitch])
        S = np.array([0, 0, 1, 0, 0, pitch], dtype=float)
    return Xj, S


def ID(
    model: SpatialModel,
    q: Any,
    qd: Any,
    qdd: Any,
    f_ext: Sequence[np.ndarray] | None = None,
    grav_accn: Any = None,
) -> np.ndarray:
    """Inverse Dynamics via Recursive Newton-Euler.

    Args:
        model: SpatialModel of the robot.
        q, qd, qdd: 1D arrays of joint positions, velocities, accelerations.
        f_ext: Optional list of external forces acting on each body.
        grav_accn: 3-vector of gravity offset (default = [0,0,-9.81]).

    Returns:
        tau: NB-vector of required joint forces/torques.
    """
    assert model is not None, "model must be provided"
    q = np.asarray(q, dtype=float).flatten()
    qd = np.asarray(qd, dtype=float).flatten()
    qdd = np.asarray(qdd, dtype=float).flatten()

    # 1-indexed internally (like MATLAB) to match featherstone's algorithm cleanly,
    # except we use 0-indexed arrays and adjust the parent lookup.
    NB = model.NB
    tau = np.zeros(NB, dtype=float)

    if grav_accn is None:
        a_grav = np.array([0, 0, 0, 0, 0, -9.81], dtype=float)
    else:
        g = np.asarray(grav_accn, dtype=float).flatten()
        a_grav = np.array([0, 0, 0, g[0], g[1], g[2]], dtype=float)

    v = [np.zeros(6)] * NB
    a = [np.zeros(6)] * NB
    f = [np.zeros(6)] * NB
    Xup = [np.eye(6)] * NB
    S_list = [np.zeros(6)] * NB

    # Forward Pass
    for i in range(NB):
        XJ, S = jcalc(model.pitch[i], q[i])
        S_list[i] = S
        vJ = S * qd[i]
        Xup[i] = XJ @ model.Xtree[i]

        parent = model.parent[i]  # 0 is world, 1..NB are bodies
        if parent == 0:
            v[i] = vJ
            a[i] = Xup[i] @ (-a_grav) + S * qdd[i]
        else:
            p_idx = parent - 1
            v[i] = Xup[i] @ v[p_idx] + vJ
            a[i] = Xup[i] @ a[p_idx] + S * qdd[i] + crm(v[i]) @ vJ

        f[i] = model.I[i] @ a[i] + crf(v[i]) @ (model.I[i] @ v[i])

        if f_ext is not None and len(f_ext) > i and f_ext[i] is not None:
            f[i] = f[i] - f_ext[i]

    # Backward Pass
    for i in range(NB - 1, -1, -1):
        tau[i] = S_list[i].T @ f[i]
        parent = model.parent[i]
        if parent != 0:
            p_idx = parent - 1
            f[p_idx] = f[p_idx] + Xup[i].T @ f[i]

    return tau


def FDab(
    model: SpatialModel,
    q: Any,
    qd: Any,
    tau: Any,
    f_ext: Sequence[np.ndarray] | None = None,
    grav_accn: Any = None,
) -> np.ndarray:
    """Forward Dynamics via Articulated-Body Algorithm.

    Args:
        model: SpatialModel of the robot.
        q, qd, tau: 1D arrays of joint position, velocity, force inputs.
        f_ext: Optional external forces logic.
        grav_accn: 3-vector gravity vector.

    Returns:
        qdd: NB-vector of resulting joint accelerations.
    """
    assert model is not None, "model must be provided"
    q = np.asarray(q, dtype=float).flatten()
    qd = np.asarray(qd, dtype=float).flatten()
    tau = np.asarray(tau, dtype=float).flatten()

    NB = model.NB
    qdd = np.zeros(NB, dtype=float)

    if grav_accn is None:
        a_grav = np.array([0, 0, 0, 0, 0, -9.81], dtype=float)
    else:
        g = np.asarray(grav_accn, dtype=float).flatten()
        a_grav = np.array([0, 0, 0, g[0], g[1], g[2]], dtype=float)

    v = [np.zeros(6)] * NB
    c = [np.zeros(6)] * NB
    pA = [np.zeros(6)] * NB
    IA = [np.zeros((6, 6))] * NB
    Xup = [np.eye(6)] * NB
    S_list = [np.zeros(6)] * NB

    # Pass 1: Kinematics and initial inertia
    for i in range(NB):
        XJ, S = jcalc(model.pitch[i], q[i])
        S_list[i] = S
        vJ = S * qd[i]
        Xup[i] = XJ @ model.Xtree[i]

        parent = model.parent[i]
        if parent == 0:
            v[i] = vJ
            c[i] = np.zeros(6)
        else:
            p_idx = parent - 1
            v[i] = Xup[i] @ v[p_idx] + vJ
            c[i] = crm(v[i]) @ vJ

        IA[i] = model.I[i].copy()
        pA[i] = crf(v[i]) @ (model.I[i] @ v[i])

        if f_ext is not None and len(f_ext) > i and f_ext[i] is not None:
            pA[i] = pA[i] - f_ext[i]

    U = [np.zeros(6)] * NB
    d = [0.0] * NB
    u = [0.0] * NB

    # Pass 2: Articulated body inertias
    for i in range(NB - 1, -1, -1):
        U[i] = IA[i] @ S_list[i]
        d[i] = float(S_list[i].T @ U[i])
        u[i] = float(tau[i] - S_list[i].T @ pA[i])

        parent = model.parent[i]
        if parent != 0:
            p_idx = parent - 1
            Ia = IA[i] - np.outer(U[i], U[i]) / d[i]
            pa = pA[i] + Ia @ c[i] + U[i] * (u[i] / d[i])
            IA[p_idx] = IA[p_idx] + Xup[i].T @ Ia @ Xup[i]
            pA[p_idx] = pA[p_idx] + Xup[i].T @ pa

    a = [np.zeros(6)] * NB

    # Pass 3: Accelerations
    for i in range(NB):
        parent = model.parent[i]
        if parent == 0:
            a[i] = Xup[i] @ (-a_grav) + c[i]
        else:
            p_idx = parent - 1
            a[i] = Xup[i] @ a[p_idx] + c[i]

        qdd[i] = (u[i] - U[i].T @ a[i]) / d[i]
        a[i] = a[i] + S_list[i] * qdd[i]

    return qdd
