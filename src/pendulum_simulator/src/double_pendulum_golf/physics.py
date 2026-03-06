"""
Double pendulum physics using Lagrangian formulation with relative coordinates.

Coordinate convention:
    q1 (theta1): Absolute angle of segment 1 from downward vertical,
                 positive counterclockwise.
    q2 (phi):    Angle of segment 2 relative to segment 1,
                 positive counterclockwise.

    When q1=0 and q2=0, both segments hang straight down (equilibrium).

All segments modeled as uniform rods with mass concentrated at the tip
(point-mass approximation) for clarity. Extend to distributed mass by
replacing L with l_c and adding rotational inertia terms.
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PendulumParams:
    """Immutable physical parameters for the double pendulum.

    Contract:
        - All lengths and masses must be strictly positive.
        - Gravity must be non-negative.
        - Damping coefficients b1, b2 must be non-negative  (N·m·s/rad).
        - Coulomb friction mu1, mu2 must be non-negative  (N·m peak magnitude).
    """

    m1: float  # mass of segment 1 (kg)
    m2: float  # mass of segment 2 (kg)
    L1: float  # length of segment 1 (m)
    L2: float  # length of segment 2 (m)
    g: float = 9.81  # gravitational acceleration (m/s^2)
    # --- Dissipative parameters (default 0 = no losses) ---
    b1: float = 0.0  # viscous damping at joint 1 (N·m·s/rad)
    b2: float = 0.0  # viscous damping at joint 2 (N·m·s/rad)
    mu1: float = 0.0  # Coulomb friction at joint 1 (N·m, constant magnitude)
    mu2: float = 0.0  # Coulomb friction at joint 2 (N·m, constant magnitude)

    def __post_init__(self) -> None:
        assert self.m1 > 0, f"m1 must be positive, got {self.m1}"
        assert self.m2 > 0, f"m2 must be positive, got {self.m2}"
        assert self.L1 > 0, f"L1 must be positive, got {self.L1}"
        assert self.L2 > 0, f"L2 must be positive, got {self.L2}"
        assert self.g >= 0, f"g must be non-negative, got {self.g}"
        assert self.b1 >= 0, f"b1 must be non-negative, got {self.b1}"
        assert self.b2 >= 0, f"b2 must be non-negative, got {self.b2}"
        assert self.mu1 >= 0, f"mu1 must be non-negative, got {self.mu1}"
        assert self.mu2 >= 0, f"mu2 must be non-negative, got {self.mu2}"


# Type alias: state vector [theta1, phi, dtheta1, dphi]
State = np.ndarray  # shape (4,)

# Torque function signature: (t) -> (tau1, tau2)
TorqueFunc = Callable[[float], Tuple[float, float]]


# ---------------------------------------------------------------------------
# Mass matrix and its components
# ---------------------------------------------------------------------------


def mass_matrix(phi: float, params: PendulumParams) -> np.ndarray:
    """Compute the 2x2 mass (inertia) matrix M(q).

    Preconditions:
        - phi is a finite float.
    Postconditions:
        - Returns a 2x2 symmetric positive-definite matrix.
        - M[0,1] == M[1,0]  (symmetry).
        - M[1,1] > 0  (positive diagonal).

    Parameters
    ----------
    phi : float
        Relative angle of segment 2 w.r.t. segment 1 (rad).
    params : PendulumParams

    Returns
    -------
    M : np.ndarray, shape (2, 2)
    """
    assert np.isfinite(phi), f"phi must be finite, got {phi}"
    m1, m2, L1, L2 = params.m1, params.m2, params.L1, params.L2
    cos_phi = np.cos(phi)

    M11 = (m1 + m2) * L1**2 + m2 * L2**2 + 2.0 * m2 * L1 * L2 * cos_phi
    M12 = m2 * L2**2 + m2 * L1 * L2 * cos_phi
    M22 = m2 * L2**2

    M = np.array([[M11, M12], [M12, M22]])

    # Postcondition: symmetry
    assert np.isclose(M[0, 1], M[1, 0]), "Mass matrix must be symmetric"
    return M


def mass_matrix_components(phi: float, params: PendulumParams) -> dict:
    """Return individual diagonal and off-diagonal terms with physical labels.

    This is the key decomposition: diagonal terms are 'self-coupling'
    (each joint's torque acting on its own acceleration) and off-diagonal
    terms are 'cross-coupling' (how one joint's torque accelerates the other).

    Parameters
    ----------
    phi : float
        Relative angle of segment 2 (rad).
    params : PendulumParams

    Returns
    -------
    dict with keys: 'M11', 'M12', 'M21', 'M22', 'M_full'
    """
    M = mass_matrix(phi, params)
    return {
        "M11": M[0, 0],
        "M12": M[0, 1],
        "M21": M[1, 0],
        "M22": M[1, 1],
        "M_full": M,
    }


# ---------------------------------------------------------------------------
# Coriolis and centrifugal terms
# ---------------------------------------------------------------------------


def coriolis_vector(
    phi: float, dtheta1: float, dphi: float, params: PendulumParams
) -> np.ndarray:
    """Compute the Coriolis/centrifugal force vector C(q, qdot) * qdot.

    This vector captures velocity-dependent forces: centrifugal effects
    from rotation and Coriolis coupling between the two angular velocities.

    Preconditions:
        - All inputs are finite floats.

    Parameters
    ----------
    phi : float
        Relative angle (rad).
    dtheta1 : float
        Angular velocity of segment 1 (rad/s).
    dphi : float
        Relative angular velocity of segment 2 (rad/s).
    params : PendulumParams

    Returns
    -------
    C_qdot : np.ndarray, shape (2,)
    """
    assert all(
        np.isfinite(v) for v in [phi, dtheta1, dphi]
    ), "All velocity inputs must be finite"
    m2, L1, L2 = params.m2, params.L1, params.L2
    h = -m2 * L1 * L2 * np.sin(phi)

    c1 = h * (2.0 * dtheta1 * dphi + dphi**2)
    c2 = -h * dtheta1**2

    return np.array([c1, c2])


# ---------------------------------------------------------------------------
# Gravity vector
# ---------------------------------------------------------------------------


def gravity_vector(theta1: float, phi: float, params: PendulumParams) -> np.ndarray:
    """Compute the gravitational torque vector G(q).

    Derived from potential energy V = -m1*g*L1*cos(theta1)
                                      - m2*g*(L1*cos(theta1) + L2*cos(theta1+phi))

    G_i = dV/dq_i

    Parameters
    ----------
    theta1 : float
        Absolute angle of segment 1 (rad).
    phi : float
        Relative angle of segment 2 (rad).
    params : PendulumParams

    Returns
    -------
    G : np.ndarray, shape (2,)
    """
    m1, m2, L1, L2, g = params.m1, params.m2, params.L1, params.L2, params.g
    abs_angle2 = theta1 + phi

    G1 = (m1 + m2) * g * L1 * np.sin(theta1) + m2 * g * L2 * np.sin(abs_angle2)
    G2 = m2 * g * L2 * np.sin(abs_angle2)

    return np.array([G1, G2])


# ---------------------------------------------------------------------------
# Friction and damping
# ---------------------------------------------------------------------------


def friction_torque_vector(
    dtheta1: float, dphi: float, params: PendulumParams
) -> np.ndarray:
    """Compute the total dissipative torque vector at the joints.

    Combines viscous (linear) damping and Coulomb (constant) friction.
    Both always oppose the direction of motion.

    Model:
        tau_friction_i = -b_i * qdot_i - mu_i * sign(qdot_i)

    The sign() function is zero when the joint is stationary, so Coulomb
    friction correctly does not apply a torque at rest.

    Preconditions:
        - dtheta1 and dphi are finite.
    Postconditions:
        - Returns shape (2,), both values finite.
        - Each component has opposite sign to the corresponding velocity
          (or is zero if both b_i and mu_i are zero).

    Parameters
    ----------
    dtheta1 : float
        Angular velocity of segment 1 (rad/s).
    dphi : float
        Relative angular velocity of segment 2 (rad/s).
    params : PendulumParams

    Returns
    -------
    tau_f : np.ndarray, shape (2,)  [N·m]
    """
    assert np.isfinite(dtheta1), f"dtheta1 must be finite, got {dtheta1}"
    assert np.isfinite(dphi), f"dphi must be finite, got {dphi}"

    tau_f1 = -params.b1 * dtheta1 - params.mu1 * np.sign(dtheta1)
    tau_f2 = -params.b2 * dphi - params.mu2 * np.sign(dphi)

    return np.array([tau_f1, tau_f2])


# ---------------------------------------------------------------------------
# Equations of motion
# ---------------------------------------------------------------------------


def equations_of_motion(
    state: State, t: float, params: PendulumParams, torque_func: TorqueFunc
) -> State:
    """Compute the state derivative: dx/dt = f(x, t).

    State vector x = [theta1, phi, dtheta1, dphi].

    Full equations of motion:

        M(q) * qddot = tau_drive(t) + tau_friction(qdot) - C(q,qdot)*qdot - G(q)

    where:
        tau_drive   = user-supplied driving torques (polynomial functions of t)
        tau_friction = dissipative joint torques (viscous damping + Coulomb friction)
                       These are NOT part of torque_func — they are computed from
                       current velocity so they can be tracked separately for the
                       total applied torque analysis.

    Preconditions:
        - state has shape (4,) with all finite values.
        - torque_func returns a 2-tuple of finite floats.
    Postconditions:
        - Returns shape (4,) with all finite values.

    Parameters
    ----------
    state : np.ndarray, shape (4,)
    t : float
    params : PendulumParams
    torque_func : callable (t) -> (tau1, tau2)  — driving torques only

    Returns
    -------
    state_dot : np.ndarray, shape (4,)
    """
    assert state.shape == (4,), f"State must have shape (4,), got {state.shape}"
    assert all(np.isfinite(state)), f"State values must be finite: {state}"

    theta1, phi, dtheta1, dphi = state

    M = mass_matrix(phi, params)
    C = coriolis_vector(phi, dtheta1, dphi, params)
    G = gravity_vector(theta1, phi, params)

    tau1, tau2 = torque_func(t)
    tau_drive = np.array([tau1, tau2])
    tau_friction = friction_torque_vector(dtheta1, dphi, params)

    # Solve: M * qddot = tau_drive + tau_friction - C - G
    rhs = tau_drive + tau_friction - C - G
    qddot = np.linalg.solve(M, rhs)

    state_dot = np.array([dtheta1, dphi, qddot[0], qddot[1]])

    assert all(
        np.isfinite(state_dot)
    ), f"State derivative has non-finite values: {state_dot}"
    return state_dot


# ---------------------------------------------------------------------------
# Forward kinematics (for visualization)
# ---------------------------------------------------------------------------


def forward_kinematics(theta1: float, phi: float, params: PendulumParams) -> dict:
    """Compute joint and tip positions in the world frame.

    Origin is at the shoulder (fixed pivot).
    x-axis points right, y-axis points up.

    Parameters
    ----------
    theta1 : float
        Absolute angle of segment 1 from downward vertical (rad).
    phi : float
        Relative angle of segment 2 (rad).
    params : PendulumParams

    Returns
    -------
    dict with 'shoulder', 'wrist', 'tip' as (x, y) tuples.
    """
    L1, L2 = params.L1, params.L2
    abs_angle2 = theta1 + phi

    # Segment 1 endpoint (wrist / elbow joint)
    wx = L1 * np.sin(theta1)
    wy = -L1 * np.cos(theta1)

    # Segment 2 endpoint (club tip)
    tx = wx + L2 * np.sin(abs_angle2)
    ty = wy - L2 * np.cos(abs_angle2)

    return {
        "shoulder": (0.0, 0.0),
        "wrist": (wx, wy),
        "tip": (tx, ty),
    }


def linear_accelerations(
    state: State, qddot: np.ndarray, params: PendulumParams
) -> dict:
    """Compute linear accelerations of joints in world coordinates.

    Returns
    -------
    dict with keys: 'wrist', 'tip' as (ax, ay) tuples.
    """
    assert state.shape == (4,), f"State must have shape (4,), got {state.shape}"
    assert qddot.shape == (2,), f"qddot must have shape (2,), got {qddot.shape}"
    theta1, phi, dtheta1, dphi = state
    ddtheta1, ddphi = qddot

    L1, L2 = params.L1, params.L2
    abs_angle2 = theta1 + phi
    dabs2 = dtheta1 + dphi
    ddabs2 = ddtheta1 + ddphi

    ax_w = L1 * (-np.sin(theta1) * dtheta1**2 + np.cos(theta1) * ddtheta1)
    ay_w = L1 * (np.cos(theta1) * dtheta1**2 + np.sin(theta1) * ddtheta1)

    ax_t = ax_w + L2 * (-np.sin(abs_angle2) * dabs2**2 + np.cos(abs_angle2) * ddabs2)
    ay_t = ay_w + L2 * (np.cos(abs_angle2) * dabs2**2 + np.sin(abs_angle2) * ddabs2)

    return {
        "wrist": (ax_w, ay_w),
        "tip": (ax_t, ay_t),
    }


def net_joint_forces(state: State, qddot: np.ndarray, params: PendulumParams) -> dict:
    """Compute net joint forces (proximal on distal) in world coordinates.

    Returns
    -------
    dict with keys: 'shoulder', 'wrist' as (fx, fy) tuples.
    """
    acc = linear_accelerations(state, qddot, params)
    g_vec = np.array([0.0, -params.g])

    m1, m2 = params.m1, params.m2
    a_w = np.array(acc["wrist"])
    a_t = np.array(acc["tip"])

    f_wrist = m2 * a_t - m2 * g_vec
    f_shoulder = (m1 * a_w + m2 * a_t) - (m1 + m2) * g_vec

    return {
        "shoulder": (float(f_shoulder[0]), float(f_shoulder[1])),
        "wrist": (float(f_wrist[0]), float(f_wrist[1])),
    }


# ---------------------------------------------------------------------------
# Energy calculations (for verification / display)
# ---------------------------------------------------------------------------


def kinetic_energy(state: State, params: PendulumParams) -> float:
    """Compute total kinetic energy T = 0.5 * qdot^T M qdot."""
    _, phi, dtheta1, dphi = state
    M = mass_matrix(phi, params)
    qdot = np.array([dtheta1, dphi])
    return float(0.5 * qdot @ M @ qdot)


def potential_energy(state: State, params: PendulumParams) -> float:
    """Compute gravitational potential energy (zero at shoulder height)."""
    theta1, phi = state[0], state[1]
    m1, m2, L1, L2, g = params.m1, params.m2, params.L1, params.L2, params.g
    abs_angle2 = theta1 + phi

    V = -(m1 + m2) * g * L1 * np.cos(theta1) - m2 * g * L2 * np.cos(abs_angle2)
    return float(V)


def total_energy(state: State, params: PendulumParams) -> float:
    """Total mechanical energy E = T + V."""
    return kinetic_energy(state, params) + potential_energy(state, params)
