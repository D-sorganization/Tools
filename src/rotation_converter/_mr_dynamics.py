# mypy: ignore-errors
"""Robot dynamics algorithms (Modern Robotics — Lynch & Park).

Inverse/forward dynamics, trajectory simulation, and computed-torque control.
Internal submodule extracted from modern_robotics.py to keep file size within
the 1200-line budget.  Import these symbols via ``rotation_converter.modern_robotics``
(the public shim) rather than directly from this private module.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rotation_converter._mr_rotation_matrices import (
    Adjoint,
    MatrixExp6,
    TransInv,
    VecTose3,
    VecToso3,
)

# ===========================================================================
# Lie bracket / adjoint action
# ===========================================================================


def ad(V: Any) -> np.ndarray:
    """Calculate the 6x6 matrix [adV] of the given 6-vector

    :param V: A 6-vector spatial velocity
    :return: The corresponding 6x6 matrix [adV]

    Used to calculate the Lie bracket [V1, V2] = [adV1]V2

    Example Input:
        V = np.array([1, 2, 3, 4, 5, 6])
    Output:
        np.array([[ 0, -3,  2,  0,  0,  0],
                  [ 3,  0, -1,  0,  0,  0],
                  [-2,  1,  0,  0,  0,  0],
                  [ 0, -6,  5,  0, -3,  2],
                  [ 6,  0, -4,  3,  0, -1],
                  [-5,  4,  0, -2,  1,  0]])
    """
    omgmat = VecToso3([V[0], V[1], V[2]])
    return np.r_[
        np.c_[omgmat, np.zeros((3, 3))], np.c_[VecToso3([V[3], V[4], V[5]]), omgmat]
    ]


# ===========================================================================
# Inverse Dynamics
# ===========================================================================


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
    """Computes inverse dynamics in the space frame for an open chain robot

    :param thetalist: n-vector of joint variables
    :param dthetalist: n-vector of joint rates
    :param ddthetalist: n-vector of joint accelerations
    :param g: Gravity vector g
    :param Ftip: Spatial force applied by the end-effector expressed in frame
                 {n+1}
    :param Mlist: List of link frames {i} relative to {i-1} at the home
                  position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :return: The n-vector of required joint forces/torques
    This function uses forward-backward Newton-Euler iterations to solve the
    equation:
    taulist = Mlist(thetalist)ddthetalist + c(thetalist,dthetalist)
              + g(thetalist) + Jtr(thetalist)Ftip

    Example Input (3 Link Robot):
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        ddthetalist = np.array([2, 1.5, 1])
        g = np.array([0, 0, -9.8])
        Ftip = np.array([1, 1, 1, 1, 1, 1])
        Slist = np.array([[1, 0, 1,      0, 1,     0],
                          [0, 1, 0, -0.089, 0,     0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
    Output:
        np.array([74.69616155, -33.06766016, -3.23057314])
    """
    assert thetalist is not None, "thetalist must be provided"
    n = len(thetalist)
    Mi = np.eye(4)
    Ai = np.zeros((6, n))
    AdTi = [[None]] * (n + 1)
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
    """Computes the mass matrix of an open chain robot based on the given
    configuration

    :param thetalist: A list of joint variables
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :return: The numerical inertia matrix M(thetalist) of an n-joint serial
             chain at the given configuration thetalist

    Example Input (3 Link Robot):
        thetalist = np.array([0.1, 0.1, 0.1])
    Output:
        np.array([[ 2.25433380e+01, -3.07146754e-01, -7.18426391e-03]
                  [-3.07146754e-01,  1.96850717e+00,  4.32157368e-01]
                  [-7.18426391e-03,  4.32157368e-01,  1.91630858e-01]])
    """
    assert thetalist is not None, "thetalist must be provided"
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
    """Computes the Coriolis and centripetal terms in the inverse dynamics of
    an open chain robot

    :param thetalist: A list of joint variables,
    :param dthetalist: A list of joint rates,
    :param Mlist: List of link frames i relative to i-1 at the home position,
    :param Glist: Spatial inertia matrices Gi of the links,
    :param Slist: Screw axes Si of the joints in a space frame.
    :return: The vector c(thetalist,dthetalist) of Coriolis and centripetal
             terms for a given thetalist and dthetalist.

    Example Input (3 Link Robot):
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
    Output:
        np.array([0.26453118, -0.05505157, -0.00689132])
    """
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
    """Computes the joint forces/torques an open chain robot requires to
    overcome gravity at its configuration

    :param thetalist: A list of joint variables
    :param g: 3-vector for gravitational acceleration
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame.
    :return grav: The joint forces/torques required to overcome gravity at
                  thetalist

    Example Inputs (3 Link Robot):
        thetalist = np.array([0.1, 0.1, 0.1])
        g = np.array([0, 0, -9.8])
    Output:
        np.array([28.40331262, -37.64094817, -5.4415892])
    """
    assert thetalist is not None, "thetalist must be provided"
    n = len(thetalist)
    return InverseDynamics(
        thetalist, [0] * n, [0] * n, g, [0, 0, 0, 0, 0, 0], Mlist, Glist, Slist
    )


def EndEffectorForces(
    thetalist: Any, Ftip: Any, Mlist: Any, Glist: Any, Slist: Any
) -> np.ndarray:
    """Computes the joint forces/torques an open chain robot requires only to
    create the end-effector force Ftip

    :param thetalist: A list of joint variables
    :param Ftip: Spatial force applied by the end-effector expressed in frame
                 {n+1}
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame.
    :return: The joint forces and torques required only to create the
             end-effector force Ftip

    Example Input (3 Link Robot):
        thetalist = np.array([0.1, 0.1, 0.1])
        Ftip = np.array([1, 1, 1, 1, 1, 1])
    Output:
        np.array([1.40954608, 1.85771497, 1.392409])
    """
    assert thetalist is not None, "thetalist must be provided"
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
    """Computes forward dynamics in the space frame for an open chain robot

    :param thetalist: A list of joint variables
    :param dthetalist: A list of joint rates
    :param taulist: An n-vector of joint forces/torques
    :param g: Gravity vector g
    :param Ftip: Spatial force applied by the end-effector expressed in frame
                 {n+1}
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame.
    :return: The resulting joint accelerations

    Example Input (3 Link Robot):
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        taulist = np.array([0.5, 0.6, 0.7])
        g = np.array([0, 0, -9.8])
        Ftip = np.array([1, 1, 1, 1, 1, 1])
    Output:
        np.array([-0.97392907, 25.58466784, -32.91499212])
    """
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
    """Compute the joint angles and velocities at the next timestep using
    first order Euler integration

    :param thetalist: n-vector of joint variables
    :param dthetalist: n-vector of joint rates
    :param ddthetalist: n-vector of joint accelerations
    :param dt: The timestep delta t
    :return thetalistNext: Vector of joint variables after dt from first
                           order Euler integration
    :return dthetalistNext: Vector of joint rates after dt from first order
                            Euler integration

    Example Inputs (3 Link Robot):
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        ddthetalist = np.array([2, 1.5, 1])
        dt = 0.1
    Output:
        thetalistNext:
        array([ 0.11,  0.12,  0.13])
        dthetalistNext:
        array([ 0.3 ,  0.35,  0.4 ])
    """
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
    """Calculates the joint forces/torques required to move the serial chain
    along the given trajectory using inverse dynamics

    :param thetamat: An N x n matrix of robot joint variables
    :param dthetamat: An N x n matrix of robot joint velocities
    :param ddthetamat: An N x n matrix of robot joint accelerations
    :param g: Gravity vector g
    :param Ftipmat: An N x 6 matrix of spatial forces applied by the end-
                    effector
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame.
    :return: The N x n matrix of joint forces/torques for the specified
             trajectory
    """
    assert thetamat is not None, "thetamat must be provided"
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
    """Simulates the motion of a serial chain given an open-loop history of
    joint forces/torques

    :param thetalist: n-vector of initial joint variables
    :param dthetalist: n-vector of initial joint rates
    :param taumat: An N x n matrix of joint forces/torques
    :param g: Gravity vector g
    :param Ftipmat: An N x 6 matrix of spatial forces applied by the end-
                    effector
    :param Mlist: List of link frames {i} relative to {i-1} at the home
                  position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame.
    :param dt: The timestep between consecutive joint forces/torques
    :param intRes: Integration resolution (Euler steps per timestep)
    :return thetamat: The N x n matrix of robot joint angles
    :return dthetamat: The N x n matrix of robot joint velocities
    """
    assert thetalist is not None, "thetalist must be provided"
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
    """Computes the joint control torques at a particular time instant

    :param thetalist: n-vector of joint variables
    :param dthetalist: n-vector of joint rates
    :param eint: n-vector of the time-integral of joint errors
    :param g: Gravity vector g
    :param Mlist: List of link frames {i} relative to {i-1} at the home
                  position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame.
    :param thetalistd: n-vector of reference joint variables
    :param dthetalistd: n-vector of reference joint velocities
    :param ddthetalistd: n-vector of reference joint accelerations
    :param Kp: The feedback proportional gain (identical for each joint)
    :param Ki: The feedback integral gain (identical for each joint)
    :param Kd: The feedback derivative gain (identical for each joint)
    :return: The vector of joint forces/torques computed by the feedback
             linearizing controller at the current instant

    Example Input:
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        eint = np.array([0.2, 0.2, 0.2])
        g = np.array([0, 0, -9.8])
    Output:
        np.array([133.00525246, -29.94223324, -3.03276856])
    """
    assert thetalist is not None, "thetalist must be provided"
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
    """Simulates the computed torque controller over a given desired
    trajectory

    :param thetalist: n-vector of initial joint variables
    :param dthetalist: n-vector of initial joint velocities
    :param g: Actual gravity vector g
    :param Ftipmat: An N x 6 matrix of spatial forces applied by the end-
                    effector
    :param Mlist: Actual list of link frames i relative to i-1
    :param Glist: Actual spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame.
    :param thetamatd: An Nxn matrix of desired joint variables
    :param dthetamatd: An Nxn matrix of desired joint velocities
    :param ddthetamatd: An Nxn matrix of desired joint accelerations
    :param gtilde: The gravity vector based on the model of the actual robot
    :param Mtildelist: The link frame locations based on the model
    :param Gtildelist: The link spatial inertias based on the model
    :param Kp: The feedback proportional gain (identical for each joint)
    :param Ki: The feedback integral gain (identical for each joint)
    :param Kd: The feedback derivative gain (identical for each joint)
    :param dt: The timestep between points on the reference trajectory
    :param intRes: Integration resolution
    :return taumat: An Nxn matrix of the controllers commanded joint
                    forces/torques
    :return thetamat: An Nxn matrix of actual joint angles
    """
    assert thetalist is not None, "thetalist must be provided"
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
            ddthetalist = ForwardDynamics(
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
                thetacurrent, dthetacurrent, ddthetalist, 1.0 * dt / intRes
            )
        taumat[:, i] = taulist
        thetamat[:, i] = thetacurrent
        eint = np.add(eint, dt * np.subtract(thetamatd[:, i], thetacurrent))
    # Output using matplotlib to plot
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pass
    else:
        links = np.array(thetamat).shape[0]
        N = np.array(thetamat).shape[1]
        Tf = N * dt
        timestamp = np.linspace(0, Tf, N)
        for i in range(links):
            col = [
                np.random.uniform(0, 1),
                np.random.uniform(0, 1),
                np.random.uniform(0, 1),
            ]
            plt.plot(
                timestamp,
                thetamat[i, :],
                "-",
                color=col,
                label=("ActualTheta" + str(i + 1)),
            )
            plt.plot(
                timestamp,
                thetamatd[i, :],
                ".",
                color=col,
                label=("DesiredTheta" + str(i + 1)),
            )
        plt.legend(loc="upper left")
        plt.xlabel("Time")
        plt.ylabel("Joint Angles")
        plt.title("Plot of Actual and Desired Joint Angles")
        plt.show()
    taumat = np.array(taumat).T
    thetamat = np.array(thetamat).T
    return (taumat, thetamat)
