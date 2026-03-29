"""Dynamics — Newton-Euler, mass matrix, forward/inverse dynamics, control.

InverseDynamics, MassMatrix, VelQuadraticForces, GravityForces,
EndEffectorForces, ForwardDynamics, EulerStep,
InverseDynamicsTrajectory, ForwardDynamicsTrajectory,
ComputedTorque, SimulateControl, ad.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .se3 import Adjoint, MatrixExp6, TransInv, VecTose3
from .so3 import VecToso3

logger = logging.getLogger(__name__)


def ad(V: Any) -> np.ndarray:
    """Calculate the 6x6 matrix [adV] of the given 6-vector."""
    omgmat = VecToso3([V[0], V[1], V[2]])
    return np.r_[
        np.c_[omgmat, np.zeros((3, 3))], np.c_[VecToso3([V[3], V[4], V[5]]), omgmat]
    ]


def InverseDynamics(
    thetalist: Any,
    dthetalist: Any,
    ddthetalist: Any,
    g: Any,
    Ftip: Any,
    Mlist: Any,
    Glist: Any,
    Slist: Any,
) -> np.ndarray:
    """Computes inverse dynamics in the space frame using forward-backward Newton-Euler."""
    if not (thetalist is not None):
        raise ValueError("thetalist must be provided")
    n = len(thetalist)
    Mi = np.eye(4)
    Ai = np.zeros((6, n))
    AdTi: list[Any] = [[None]] * (n + 1)
    Vi = np.zeros((6, n + 1))
    Vdi = np.zeros((6, n + 1))
    Vdi[:, 0] = np.r_[[0, 0, 0], -np.array(g)]
    AdTi[n] = Adjoint(TransInv(Mlist[n]))
    Fi = np.array(Ftip).copy()
    taulist = np.zeros(n)
    for i in range(n):
        Mi = np.dot(Mi, Mlist[i])
        Ai[:, i] = np.dot(Adjoint(TransInv(Mi)), np.array(Slist)[:, i])
        AdTi[i] = Adjoint(
            np.dot(MatrixExp6(VecTose3(Ai[:, i] * -thetalist[i])), TransInv(Mlist[i]))
        )
        Vi[:, i + 1] = np.dot(AdTi[i], Vi[:, i]) + Ai[:, i] * dthetalist[i]
        Vdi[:, i + 1] = (
            np.dot(AdTi[i], Vdi[:, i])
            + Ai[:, i] * ddthetalist[i]
            + np.dot(ad(Vi[:, i + 1]), Ai[:, i]) * dthetalist[i]
        )
    for i in range(n - 1, -1, -1):
        Fi = (
            np.dot(np.array(AdTi[i + 1]).T, Fi)
            + np.dot(np.array(Glist[i]), Vdi[:, i + 1])
            - np.dot(
                np.array(ad(Vi[:, i + 1])).T, np.dot(np.array(Glist[i]), Vi[:, i + 1])
            )
        )
        taulist[i] = np.dot(np.array(Fi).T, Ai[:, i])
    return taulist


def MassMatrix(thetalist: Any, Mlist: Any, Glist: Any, Slist: Any) -> np.ndarray:
    """Computes the mass matrix of an open chain robot."""
    if not (thetalist is not None):
        raise ValueError("thetalist must be provided")
    n = len(thetalist)
    M = np.zeros((n, n))
    for i in range(n):
        ddthetalist = [0] * n
        ddthetalist[i] = 1
        M[:, i] = InverseDynamics(
            thetalist,
            [0] * n,
            ddthetalist,
            [0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            Mlist,
            Glist,
            Slist,
        )
    return M


def VelQuadraticForces(
    thetalist: Any, dthetalist: Any, Mlist: Any, Glist: Any, Slist: Any
) -> np.ndarray:
    """Computes the Coriolis and centripetal terms."""
    return InverseDynamics(
        thetalist,
        dthetalist,
        [0] * len(thetalist),
        [0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        Mlist,
        Glist,
        Slist,
    )


def GravityForces(
    thetalist: Any, g: Any, Mlist: Any, Glist: Any, Slist: Any
) -> np.ndarray:
    """Computes the joint forces/torques required to overcome gravity."""
    if not (thetalist is not None):
        raise ValueError("thetalist must be provided")
    n = len(thetalist)
    return InverseDynamics(
        thetalist, [0] * n, [0] * n, g, [0, 0, 0, 0, 0, 0], Mlist, Glist, Slist
    )


def EndEffectorForces(
    thetalist: Any, Ftip: Any, Mlist: Any, Glist: Any, Slist: Any
) -> np.ndarray:
    """Computes the joint forces/torques required to create end-effector force Ftip."""
    if not (thetalist is not None):
        raise ValueError("thetalist must be provided")
    n = len(thetalist)
    return InverseDynamics(
        thetalist, [0] * n, [0] * n, [0, 0, 0], Ftip, Mlist, Glist, Slist
    )


def ForwardDynamics(
    thetalist: Any,
    dthetalist: Any,
    taulist: Any,
    g: Any,
    Ftip: Any,
    Mlist: Any,
    Glist: Any,
    Slist: Any,
) -> np.ndarray:
    """Computes forward dynamics in the space frame."""
    return np.dot(
        np.linalg.inv(MassMatrix(thetalist, Mlist, Glist, Slist)),
        np.array(taulist)
        - VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist)
        - GravityForces(thetalist, g, Mlist, Glist, Slist)
        - EndEffectorForces(thetalist, Ftip, Mlist, Glist, Slist),
    )


def EulerStep(
    thetalist: Any, dthetalist: Any, ddthetalist: Any, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the joint angles and velocities at the next timestep using Euler integration."""
    return thetalist + dt * np.array(dthetalist), dthetalist + dt * np.array(
        ddthetalist
    )


def InverseDynamicsTrajectory(
    thetamat: Any,
    dthetamat: Any,
    ddthetamat: Any,
    g: Any,
    Ftipmat: Any,
    Mlist: Any,
    Glist: Any,
    Slist: Any,
) -> np.ndarray:
    """Calculates joint forces/torques for a trajectory using inverse dynamics."""
    if not (thetamat is not None):
        raise ValueError("thetamat must be provided")
    thetamat = np.array(thetamat).T
    dthetamat = np.array(dthetamat).T
    ddthetamat = np.array(ddthetamat).T
    Ftipmat = np.array(Ftipmat).T
    taumat = np.array(thetamat).copy()
    for i in range(np.array(thetamat).shape[1]):
        taumat[:, i] = InverseDynamics(
            thetamat[:, i],
            dthetamat[:, i],
            ddthetamat[:, i],
            g,
            Ftipmat[:, i],
            Mlist,
            Glist,
            Slist,
        )
    taumat = np.array(taumat).T
    return taumat


def ForwardDynamicsTrajectory(
    thetalist: Any,
    dthetalist: Any,
    taumat: Any,
    g: Any,
    Ftipmat: Any,
    Mlist: Any,
    Glist: Any,
    Slist: Any,
    dt: float,
    intRes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulates the motion of a serial chain given an open-loop history of joint forces/torques."""
    if not (thetalist is not None):
        raise ValueError("thetalist must be provided")
    taumat = np.array(taumat).T
    Ftipmat = np.array(Ftipmat).T
    thetamat = taumat.copy().astype(float)
    thetamat[:, 0] = thetalist
    dthetamat = taumat.copy().astype(float)
    dthetamat[:, 0] = dthetalist
    for i in range(np.array(taumat).shape[1] - 1):
        for _j in range(intRes):
            ddthetalist = ForwardDynamics(
                thetalist,
                dthetalist,
                taumat[:, i],
                g,
                Ftipmat[:, i],
                Mlist,
                Glist,
                Slist,
            )
            thetalist, dthetalist = EulerStep(
                thetalist, dthetalist, ddthetalist, 1.0 * dt / intRes
            )
        thetamat[:, i + 1] = thetalist
        dthetamat[:, i + 1] = dthetalist
    thetamat = np.array(thetamat).T
    dthetamat = np.array(dthetamat).T
    return thetamat, dthetamat


def ComputedTorque(
    thetalist: Any,
    dthetalist: Any,
    eint: Any,
    g: Any,
    Mlist: Any,
    Glist: Any,
    Slist: Any,
    thetalistd: Any,
    dthetalistd: Any,
    ddthetalistd: Any,
    Kp: float,
    Ki: float,
    Kd: float,
) -> np.ndarray:
    """Computes the joint control torques at a particular time instant."""
    if not (thetalist is not None):
        raise ValueError("thetalist must be provided")
    e = np.subtract(thetalistd, thetalist)
    return np.dot(
        MassMatrix(thetalist, Mlist, Glist, Slist),
        Kp * e + Ki * (np.array(eint) + e) + Kd * np.subtract(dthetalistd, dthetalist),
    ) + InverseDynamics(
        thetalist, dthetalist, ddthetalistd, g, [0, 0, 0, 0, 0, 0], Mlist, Glist, Slist
    )


def SimulateControl(
    thetalist: Any,
    dthetalist: Any,
    g: Any,
    Ftipmat: Any,
    Mlist: Any,
    Glist: Any,
    Slist: Any,
    thetamatd: Any,
    dthetamatd: Any,
    ddthetamatd: Any,
    gtilde: Any,
    Mtildelist: Any,
    Gtildelist: Any,
    Kp: float,
    Ki: float,
    Kd: float,
    dt: float,
    intRes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulates the computed torque controller over a given desired trajectory."""
    if not (thetalist is not None):
        raise ValueError("thetalist must be provided")
    Ftipmat = np.array(Ftipmat).T
    thetamatd = np.array(thetamatd).T
    dthetamatd = np.array(dthetamatd).T
    ddthetamatd = np.array(ddthetamatd).T
    m, n = np.array(thetamatd).shape
    thetacurrent = np.array(thetalist).copy()
    dthetacurrent = np.array(dthetalist).copy()
    eint = np.zeros((m, 1)).reshape(
        m,
    )
    taumat = np.zeros(np.array(thetamatd).shape)
    thetamat = np.zeros(np.array(thetamatd).shape)
    for i in range(n):
        taulist = ComputedTorque(
            thetacurrent,
            dthetacurrent,
            eint,
            gtilde,
            Mtildelist,
            Gtildelist,
            Slist,
            thetamatd[:, i],
            dthetamatd[:, i],
            ddthetamatd[:, i],
            Kp,
            Ki,
            Kd,
        )
        for _j in range(intRes):
            ddthetalist_val = ForwardDynamics(
                thetacurrent,
                dthetacurrent,
                taulist,
                g,
                Ftipmat[:, i],
                Mlist,
                Glist,
                Slist,
            )
            thetacurrent, dthetacurrent = EulerStep(
                thetacurrent, dthetacurrent, ddthetalist_val, 1.0 * dt / intRes
            )
        taumat[:, i] = taulist
        thetamat[:, i] = thetacurrent
        eint = np.add(eint, dt * np.subtract(thetamatd[:, i], thetacurrent))
    # Plotting removed — callers can plot from returned data
    taumat = np.array(taumat).T
    thetamat = np.array(thetamat).T
    return (taumat, thetamat)
