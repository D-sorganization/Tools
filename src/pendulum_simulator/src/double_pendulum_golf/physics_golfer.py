"""
Golfer upper-body physics using Lagrangian formulation with a closed kinematic loop.

Topology (from the sketch)
--------------------------
Fixed hub connects via a standoff to two shoulder joints that branch
into independent arm chains.  Both wrist endpoints attach to different
points on a shared club segment, closing the kinematic loop.

    Hub ─── RS (right shoulder)
     │           │
     │          RE (right elbow)
     │           │
     │          RH (right hand / wrist)──┐
     │                                   Club ── Clubhead
     │          LH (left hand / wrist)───┘
     │           │
     │          LE (left elbow)
     │           │
     └── LS (left shoulder)

Segments (8 total):
    1. Hub standoff (fixed → hub point)
    2. Hub → Right Shoulder
    3. Right Shoulder → Right Elbow (right upper arm)
    4. Right Elbow → Right Hand (right forearm)
    5. Hub → Left Shoulder
    6. Left Shoulder → Left Elbow (left upper arm)
    7. Left Elbow → Left Hand (left forearm)
    8. Club (shaft + clubhead)

Generalized coordinates (open-chain, before constraint):
    q = [theta_hub,          # hub rotation (absolute, from downward vertical)
         alpha_rs,           # right shoulder relative angle
         alpha_re,           # right elbow relative angle
         alpha_rh,           # right wrist relative angle
         alpha_ls,           # left shoulder relative angle
         alpha_le,           # left elbow relative angle
         alpha_lh,           # left wrist relative angle
         theta_club]         # club absolute angle

The closed loop imposes a holonomic constraint:
    Right-hand grip position == Club grip point (right)
    Left-hand grip position  == Club grip point (left)
This gives 2 × 2 = 4 scalar constraints on 8 DOFs → 4 independent DOFs.

We use a minimal-coordinate formulation with constraint enforcement via
the augmented Lagrangian / Baumgarte stabilization approach.

Coordinate convention:
    Angles measured from downward vertical, positive counterclockwise.
    World frame: x→right, y→up, origin at hub.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GolferParams:
    """Immutable physical parameters for the golfer upper-body model.

    Contract:
        - All lengths and masses must be strictly positive.
        - Gravity must be non-negative.
        - Grip offsets must be non-negative and sum ≤ club length.
    """

    # Segment masses (kg)
    m_hub: float  # hub standoff mass
    m_r_upper: float  # right upper arm
    m_r_fore: float  # right forearm
    m_l_upper: float  # left upper arm
    m_l_fore: float  # left forearm
    m_club: float  # club shaft + head

    # Segment lengths (m)
    L_hub: float  # hub standoff length
    L_r_upper: float  # right upper arm length
    L_r_fore: float  # right forearm length
    L_l_upper: float  # left upper arm length
    L_l_fore: float  # left forearm length
    L_club: float  # club total length

    # Shoulder offsets from hub (m) — how far RS and LS are from hub center
    d_rs: float  # distance from hub to right shoulder along hub bar
    d_ls: float  # distance from hub to left shoulder along hub bar

    # Grip positions on club (distance from club base)
    grip_right: float  # right hand grip distance from club base
    grip_left: float  # left hand grip distance from club base

    # Clubhead mass (point mass at tip)
    m_clubhead: float = 0.2

    # Gravity
    g: float = 9.81

    # Dissipation (default: no losses)
    b_hub: float = 0.0
    b_rs: float = 0.0
    b_re: float = 0.0
    b_rh: float = 0.0
    b_ls: float = 0.0
    b_le: float = 0.0
    b_lh: float = 0.0

    def __post_init__(self) -> None:
        for name, val in [
            ("m_hub", self.m_hub),
            ("m_r_upper", self.m_r_upper),
            ("m_r_fore", self.m_r_fore),
            ("m_l_upper", self.m_l_upper),
            ("m_l_fore", self.m_l_fore),
            ("m_club", self.m_club),
        ]:
            assert val > 0, f"{name} must be positive, got {val}"

        for name, val in [
            ("L_hub", self.L_hub),
            ("L_r_upper", self.L_r_upper),
            ("L_r_fore", self.L_r_fore),
            ("L_l_upper", self.L_l_upper),
            ("L_l_fore", self.L_l_fore),
            ("L_club", self.L_club),
        ]:
            assert val > 0, f"{name} must be positive, got {val}"

        assert self.d_rs >= 0, f"d_rs must be non-negative, got {self.d_rs}"
        assert self.d_ls >= 0, f"d_ls must be non-negative, got {self.d_ls}"
        assert (
            self.grip_right >= 0
        ), f"grip_right must be non-negative, got {self.grip_right}"
        assert (
            self.grip_left >= 0
        ), f"grip_left must be non-negative, got {self.grip_left}"
        assert self.grip_right <= self.L_club, "grip_right must be ≤ L_club"
        assert self.grip_left <= self.L_club, "grip_left must be ≤ L_club"
        assert self.g >= 0, f"g must be non-negative, got {self.g}"
        assert (
            self.m_clubhead >= 0
        ), f"m_clubhead must be non-negative, got {self.m_clubhead}"

        for name in ["b_hub", "b_rs", "b_re", "b_rh", "b_ls", "b_le", "b_lh"]:
            val = getattr(self, name)
            assert val >= 0, f"{name} must be non-negative, got {val}"


# State: 8 angles + 8 angular velocities = 16 DOF
State = np.ndarray  # shape (16,)

# Torque function: (t) -> 7 torques (hub, rs, re, rh, ls, le, lh)
TorqueFunc = Callable[[float], tuple[float, float, float, float, float, float, float]]

# Number of generalized coordinates
N_DOF = 8

# Number of constraints (2 loop-closure constraints × 2D = 4)
N_CONSTRAINTS = 4


# ---------------------------------------------------------------------------
# Forward kinematics helpers
# ---------------------------------------------------------------------------


def _hub_position(theta_hub: float, p: GolferParams) -> tuple[float, float]:
    """Hub endpoint position (end of standoff from fixed origin)."""
    x = p.L_hub * np.sin(theta_hub)
    y = -p.L_hub * np.cos(theta_hub)
    return (x, y)


def _shoulder_position(
    hub_xy: tuple[float, float],
    theta_hub: float,
    d_shoulder: float,
    side: float,
) -> tuple[float, float]:
    """Shoulder joint position.

    The shoulder bar is perpendicular to the hub standoff.
    side: +1 for right, -1 for left (perpendicular offset direction).
    """
    # Perpendicular direction to hub standoff (rotated 90° from hub direction)
    perp_x = side * np.cos(theta_hub)
    perp_y = side * np.sin(theta_hub)
    x = hub_xy[0] + d_shoulder * perp_x
    y = hub_xy[1] + d_shoulder * perp_y
    return (x, y)


def _chain_endpoint(
    origin: tuple[float, float],
    angles_abs: list[float],
    lengths: list[float],
) -> tuple[float, float]:
    """Compute endpoint of a serial chain from origin.

    Parameters
    ----------
    origin : (x, y) start position
    angles_abs : list of absolute angles for each segment
    lengths : list of segment lengths

    Returns
    -------
    (x, y) endpoint position
    """
    x, y = origin
    for angle, length in zip(angles_abs, lengths):
        x += length * np.sin(angle)
        y -= length * np.cos(angle)
    return (x, y)


def _absolute_angles(theta_hub: float, relative_angles: list[float]) -> list[float]:
    """Convert relative joint angles to absolute angles for a chain.

    Each relative angle is added cumulatively to the hub angle.
    """
    result = []
    cumulative = theta_hub
    for rel in relative_angles:
        cumulative += rel
        result.append(cumulative)
    return result


def forward_kinematics(
    q: np.ndarray, p: GolferParams
) -> dict[str, tuple[float, float]]:
    """Compute all joint positions in world frame.

    Parameters
    ----------
    q : np.ndarray, shape (8,) or (16,) — generalized coordinates
        [theta_hub, alpha_rs, alpha_re, alpha_rh,
         alpha_ls, alpha_le, alpha_lh, theta_club]

    Returns
    -------
    dict with keys: 'hub', 'rs', 're', 'rh', 'ls', 'le', 'lh',
                    'club_base', 'club_tip', 'grip_right', 'grip_left'
    """
    if q.shape[0] > N_DOF:
        q = q[:N_DOF]
    assert q.shape == (N_DOF,), f"q must have shape ({N_DOF},), got {q.shape}"

    th_hub = q[0]
    alpha_rs, alpha_re, alpha_rh = q[1], q[2], q[3]
    alpha_ls, alpha_le, alpha_lh = q[4], q[5], q[6]
    th_club = q[7]

    hub = _hub_position(th_hub, p)

    # Shoulder positions (perpendicular to hub standoff)
    rs = _shoulder_position(hub, th_hub, p.d_rs, +1.0)
    ls = _shoulder_position(hub, th_hub, p.d_ls, -1.0)

    # Right arm chain: RS → RE → RH
    r_abs = _absolute_angles(th_hub, [alpha_rs, alpha_re, alpha_rh])
    re = _chain_endpoint(rs, [r_abs[0]], [p.L_r_upper])
    rh = _chain_endpoint(rs, r_abs[:2], [p.L_r_upper, p.L_r_fore])

    # Left arm chain: LS → LE → LH
    l_abs = _absolute_angles(th_hub, [alpha_ls, alpha_le, alpha_lh])
    le = _chain_endpoint(ls, [l_abs[0]], [p.L_l_upper])
    lh = _chain_endpoint(ls, l_abs[:2], [p.L_l_upper, p.L_l_fore])

    # Club direction unit vector
    club_dx = np.sin(th_club)
    club_dy = -np.cos(th_club)

    # Club base defined from right-hand grip position along club direction
    club_base = (
        rh[0] - p.grip_right * club_dx,
        rh[1] + p.grip_right * club_dy,
    )
    grip_l_on_club = (
        club_base[0] + p.grip_left * club_dx,
        club_base[1] - p.grip_left * club_dy,
    )
    club_tip = (
        club_base[0] + p.L_club * club_dx,
        club_base[1] - p.L_club * club_dy,
    )

    return {
        "origin": (0.0, 0.0),
        "hub": hub,
        "rs": rs,
        "re": re,
        "rh": rh,
        "ls": ls,
        "le": le,
        "lh": lh,
        "club_base": club_base,
        "club_tip": club_tip,
        "grip_right": rh,
        "grip_left": grip_l_on_club,
    }


# ---------------------------------------------------------------------------
# Constraint equations (closed kinematic loop)
# ---------------------------------------------------------------------------


def constraint_vector(q: np.ndarray, p: GolferParams) -> np.ndarray:
    """Evaluate the 4 loop-closure constraint equations.

    Phi(q) = 0 when the loop is closed:
        Phi[0:2] = RH_position - club_grip_right_position = 0
        Phi[2:4] = LH_position - club_grip_left_position = 0

    Returns
    -------
    Phi : np.ndarray, shape (4,)
    """
    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    fk = forward_kinematics(q, p)

    # Right hand must coincide with right grip on club
    # (already coincides by construction — the club is placed from RH)
    # Left hand must coincide with left grip on club
    phi = np.zeros(N_CONSTRAINTS)

    # The real constraint: LH (from left arm chain) must match
    # grip_left (from club geometry anchored at RH)
    phi[0] = fk["lh"][0] - fk["grip_left"][0]
    phi[1] = fk["lh"][1] - fk["grip_left"][1]

    # Additional constraint: right hand position determines club base
    # Club angle must be consistent with the grip positions
    # This is implicit when theta_club is correct, but we enforce it:
    # The vector from grip_right to grip_left on the club must match
    # the vector from RH to LH projected onto the club direction
    rh = np.array(fk["rh"])
    lh_arm = np.array(fk["lh"])
    club_dir = np.array([np.sin(q[7]), -np.cos(q[7])])
    grip_sep = p.grip_left - p.grip_right  # signed distance along club

    # Club direction constraint: RH→LH vector dot club_perp = 0
    club_perp = np.array([-club_dir[1], club_dir[0]])
    rh_to_lh = lh_arm - rh

    phi[2] = np.dot(rh_to_lh, club_perp)  # perpendicular distance = 0
    phi[3] = np.dot(rh_to_lh, club_dir) - grip_sep  # along-club distance

    return phi


def numerical_constraint_jacobian(q: np.ndarray, p: GolferParams) -> np.ndarray:
    """Compute the constraint Jacobian Phi_q = dPhi/dq numerically.

    Parameters
    ----------
    q : np.ndarray, shape (8,) — generalized coordinates

    Returns
    -------
    Phi_q : np.ndarray, shape (4, 8)
    """
    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    eps = 1e-7
    Phi_0 = constraint_vector(q, p)
    n_c = len(Phi_0)
    Phi_q = np.zeros((n_c, N_DOF))

    for j in range(N_DOF):
        q_plus = q.copy()
        q_plus[j] += eps
        Phi_q[:, j] = (constraint_vector(q_plus, p) - Phi_0) / eps

    return Phi_q


# ---------------------------------------------------------------------------
# Mass matrix (point-mass approximation, open chain)
# ---------------------------------------------------------------------------


def numerical_mass_matrix(q: np.ndarray, p: GolferParams) -> np.ndarray:
    """Compute the 8×8 mass matrix M(q) via numerical Jacobian method.

    Uses the kinetic energy definition:
        T = 0.5 * sum_i(m_i * v_i^T * v_i)
        M(q) = sum_i(m_i * J_i^T * J_i)

    where J_i = d(pos_i)/dq is the position Jacobian of mass i.

    Returns
    -------
    M : np.ndarray, shape (8, 8) — symmetric positive semi-definite
    """
    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    eps = 1e-7
    M = np.zeros((N_DOF, N_DOF))

    # Mass points: each segment's tip carries its mass
    mass_points = _mass_point_positions(q, p)

    for mass_val, pos_func in mass_points:
        J = np.zeros((2, N_DOF))
        pos_0 = pos_func(q)
        for j in range(N_DOF):
            q_plus = q.copy()
            q_plus[j] += eps
            pos_j = pos_func(q_plus)
            J[0, j] = (pos_j[0] - pos_0[0]) / eps
            J[1, j] = (pos_j[1] - pos_0[1]) / eps
        M += mass_val * J.T @ J

    return M


def _mass_point_positions(
    q: np.ndarray, p: GolferParams
) -> list[tuple[float, Callable]]:
    """Return list of (mass, position_function) for all point masses."""

    def hub_pos(qq: np.ndarray) -> tuple[float, float]:
        return _hub_position(qq[0], p)

    def re_pos(qq: np.ndarray) -> tuple[float, float]:
        fk = forward_kinematics(qq, p)
        return fk["re"]

    def rh_pos(qq: np.ndarray) -> tuple[float, float]:
        fk = forward_kinematics(qq, p)
        return fk["rh"]

    def le_pos(qq: np.ndarray) -> tuple[float, float]:
        fk = forward_kinematics(qq, p)
        return fk["le"]

    def lh_pos(qq: np.ndarray) -> tuple[float, float]:
        fk = forward_kinematics(qq, p)
        return fk["lh"]

    def club_tip_pos(qq: np.ndarray) -> tuple[float, float]:
        fk = forward_kinematics(qq, p)
        return fk["club_tip"]

    def club_com_pos(qq: np.ndarray) -> tuple[float, float]:
        fk = forward_kinematics(qq, p)
        base = fk["club_base"]
        tip = fk["club_tip"]
        return (0.5 * (base[0] + tip[0]), 0.5 * (base[1] + tip[1]))

    return [
        (p.m_hub, hub_pos),
        (p.m_r_upper, re_pos),
        (p.m_r_fore, rh_pos),
        (p.m_l_upper, le_pos),
        (p.m_l_fore, lh_pos),
        (p.m_club, club_com_pos),
        (p.m_clubhead, club_tip_pos),
    ]


# ---------------------------------------------------------------------------
# Coriolis and gravity via numerical differentiation
# ---------------------------------------------------------------------------


def numerical_coriolis_matrix(
    q: np.ndarray, qdot: np.ndarray, p: GolferParams
) -> np.ndarray:
    """Compute C(q, qdot) * qdot using finite differences of M(q).

    Uses Christoffel symbols: C_ij = sum_k (c_ijk * qdot_k)
    where c_ijk = 0.5 * (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i)

    Returns
    -------
    C_qdot : np.ndarray, shape (8,)
    """
    if q.shape[0] > N_DOF:
        q = q[:N_DOF]
    if qdot.shape[0] > N_DOF:
        qdot = qdot[:N_DOF]

    eps = 1e-7
    M0 = numerical_mass_matrix(q, p)
    dM = np.zeros((N_DOF, N_DOF, N_DOF))  # dM[i,j,k] = dM_ij/dq_k

    for k in range(N_DOF):
        q_plus = q.copy()
        q_plus[k] += eps
        dM[:, :, k] = (numerical_mass_matrix(q_plus, p) - M0) / eps

    C_qdot = np.zeros(N_DOF)
    for i in range(N_DOF):
        for j in range(N_DOF):
            christoffel = 0.0
            for k in range(N_DOF):
                christoffel += 0.5 * (dM[i, j, k] + dM[i, k, j] - dM[j, k, i]) * qdot[k]
            C_qdot[i] += christoffel * qdot[j]

    return C_qdot


def numerical_gravity_vector(q: np.ndarray, p: GolferParams) -> np.ndarray:
    """Compute gravitational torque vector G(q) via potential energy gradient.

    G_i = dV/dq_i where V = sum(m_k * g * y_k)

    Returns
    -------
    G : np.ndarray, shape (8,)
    """
    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    eps = 1e-7
    V0 = potential_energy_from_q(q, p)
    G = np.zeros(N_DOF)

    for i in range(N_DOF):
        q_plus = q.copy()
        q_plus[i] += eps
        G[i] = (potential_energy_from_q(q_plus, p) - V0) / eps

    return G


def potential_energy_from_q(q: np.ndarray, p: GolferParams) -> float:
    """Compute total gravitational potential energy from coordinates."""
    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    mass_points = _mass_point_positions(q, p)
    V = 0.0
    for mass_val, pos_func in mass_points:
        _, y = pos_func(q)
        V += mass_val * p.g * y

    return V


# ---------------------------------------------------------------------------
# Friction
# ---------------------------------------------------------------------------


def friction_torque_vector(qdot: np.ndarray, p: GolferParams) -> np.ndarray:
    """Compute viscous damping torque at all joints.

    Returns
    -------
    tau_f : np.ndarray, shape (8,)
        Note: club DOF (index 7) has no independent damping.
    """
    if qdot.shape[0] > N_DOF:
        qdot = qdot[:N_DOF]

    b = np.array([p.b_hub, p.b_rs, p.b_re, p.b_rh, p.b_ls, p.b_le, p.b_lh, 0.0])
    result: np.ndarray = -b * qdot
    return result


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------


def kinetic_energy(q: np.ndarray, qdot: np.ndarray, p: GolferParams) -> float:
    """Compute T = 0.5 * qdot^T M qdot."""
    if q.shape[0] > N_DOF:
        q = q[:N_DOF]
    if qdot.shape[0] > N_DOF:
        qdot = qdot[:N_DOF]

    M = mass_matrix(q, p)
    return float(0.5 * qdot @ M @ qdot)


def potential_energy(state: State, p: GolferParams) -> float:
    """Compute gravitational PE from full state vector."""
    return potential_energy_from_q(state[:N_DOF], p)


def total_energy(state: State, p: GolferParams) -> float:
    """Compute E = T + V from full state."""
    q = state[:N_DOF]
    qdot = state[N_DOF:]
    return kinetic_energy(q, qdot, p) + potential_energy(state, p)


# ---------------------------------------------------------------------------
# Joint force / acceleration computations
# ---------------------------------------------------------------------------


def linear_accelerations(
    q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray, p: GolferParams
) -> dict:
    """Compute linear accelerations at all joints via numerical Jacobian.

    Returns dict with keys matching forward_kinematics joint names.
    """
    if q.shape[0] > N_DOF:
        q = q[:N_DOF]
    if qdot.shape[0] > N_DOF:
        qdot = qdot[:N_DOF]

    eps = 1e-7
    fk0 = forward_kinematics(q, p)
    joint_names = ["hub", "rs", "re", "rh", "ls", "le", "lh", "club_tip"]

    result = {}
    for name in joint_names:
        J = np.zeros((2, N_DOF))
        pos_0 = np.array(fk0[name])
        for j in range(N_DOF):
            q_plus = q.copy()
            q_plus[j] += eps
            fk_j = forward_kinematics(q_plus, p)
            pos_j = np.array(fk_j[name])
            J[:, j] = (pos_j - pos_0) / eps

        # Jdot*qdot via finite difference
        fk_dt = forward_kinematics(q + eps * qdot, p)
        J_dt = np.zeros((2, N_DOF))
        pos_dt = np.array(fk_dt[name])
        for j in range(N_DOF):
            q_plus_dt = (q + eps * qdot).copy()
            q_plus_dt[j] += eps
            fk_jdt = forward_kinematics(q_plus_dt, p)
            pos_jdt = np.array(fk_jdt[name])
            J_dt[:, j] = (pos_jdt - pos_dt) / eps
        Jdot = (J_dt - J) / eps

        acc = J @ qddot + Jdot @ qdot
        result[name] = (float(acc[0]), float(acc[1]))

    return result


def net_joint_forces(
    q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray, p: GolferParams
) -> dict:
    """Compute net force at each joint using Newton's second law.

    F = m * a - m * g_vec for each point mass.

    Returns dict with joint name → (fx, fy) tuples.
    """
    acc = linear_accelerations(q, qdot, qddot, p)
    g_vec = np.array([0.0, -p.g])

    forces = {}
    joint_mass_map = {
        "hub": p.m_hub,
        "re": p.m_r_upper,
        "rh": p.m_r_fore,
        "le": p.m_l_upper,
        "lh": p.m_l_fore,
        "club_tip": p.m_club + p.m_clubhead,
    }

    for name, mass in joint_mass_map.items():
        a = np.array(acc[name])
        f = mass * a - mass * g_vec
        forces[name] = (float(f[0]), float(f[1]))

    return forces


# ---------------------------------------------------------------------------
# Analytical Jacobian computation (Phase 1 optimization)
# ---------------------------------------------------------------------------


def analytical_fk_jacobians(q: np.ndarray, p: GolferParams) -> dict[str, np.ndarray]:
    """Compute position Jacobians analytically for all mass points.

    Parameters
    ----------
    q : np.ndarray, shape (8,)
        Generalized coordinates
    p : GolferParams
        Physical parameters

    Returns
    -------
    dict with keys: 'hub', 're', 'rh', 'le', 'lh', 'club_com', 'club_tip'
    Each value is a 2×8 matrix: J[point][row, col] = d(pos[row])/dq[col]
    """
    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    # Extract coordinates for clarity
    th_hub = q[0]
    alpha_rs, alpha_re, alpha_rh = q[1], q[2], q[3]
    alpha_ls, alpha_le, alpha_lh = q[4], q[5], q[6]
    th_club = q[7]

    # Precompute sine/cosine values
    sin_hub = np.sin(th_hub)
    cos_hub = np.cos(th_hub)

    th_rs_abs = th_hub + alpha_rs
    sin_rs = np.sin(th_rs_abs)
    cos_rs = np.cos(th_rs_abs)

    th_re_abs = th_hub + alpha_rs + alpha_re
    sin_re = np.sin(th_re_abs)
    cos_re = np.cos(th_re_abs)

    th_rh_abs = th_hub + alpha_rs + alpha_re + alpha_rh  # noqa: F841

    th_ls_abs = th_hub + alpha_ls
    sin_ls = np.sin(th_ls_abs)
    cos_ls = np.cos(th_ls_abs)

    th_le_abs = th_hub + alpha_ls + alpha_le
    sin_le = np.sin(th_le_abs)
    cos_le = np.cos(th_le_abs)

    th_lh_abs = th_hub + alpha_ls + alpha_le + alpha_lh  # noqa: F841

    sin_club = np.sin(th_club)
    cos_club = np.cos(th_club)

    jacobians = {}

    # -----------------------------------------------------------------------
    # 1. HUB: position = (L_hub * sin(th_hub), -L_hub * cos(th_hub))
    # Depends only on q[0]
    # -----------------------------------------------------------------------
    J_hub = np.zeros((2, N_DOF))
    J_hub[0, 0] = p.L_hub * cos_hub
    J_hub[1, 0] = p.L_hub * sin_hub
    jacobians["hub"] = J_hub

    # -----------------------------------------------------------------------
    # 2. RS (Right Shoulder): position from hub + perpendicular offset
    # rs_x = hub_x + d_rs * cos(th_hub)
    # rs_y = hub_y + d_rs * sin(th_hub)
    # -----------------------------------------------------------------------
    J_rs = np.zeros((2, N_DOF))
    # d/dq[0]: d(hub_x)/dq[0] + d(d_rs*cos)/dq[0]
    J_rs[0, 0] = p.L_hub * cos_hub - p.d_rs * sin_hub
    J_rs[1, 0] = p.L_hub * sin_hub + p.d_rs * cos_hub
    jacobians["rs"] = J_rs

    # -----------------------------------------------------------------------
    # 3. RE (Right Elbow): from RS along right upper arm
    # re_x = rs_x + L_r_upper * sin(th_rs_abs)
    # re_y = rs_y - L_r_upper * cos(th_rs_abs)
    # Depends on q[0], q[1]
    # -----------------------------------------------------------------------
    J_re = np.zeros((2, N_DOF))
    # d/dq[0]: d(rs)/dq[0] + d(L_r_upper*sin(th_rs_abs))/dq[0]
    J_re[0, 0] = p.L_hub * cos_hub - p.d_rs * sin_hub + p.L_r_upper * cos_rs
    J_re[1, 0] = p.L_hub * sin_hub + p.d_rs * cos_hub + p.L_r_upper * sin_rs
    # d/dq[1]: d(L_r_upper*sin(th_rs_abs))/dq[1]
    J_re[0, 1] = p.L_r_upper * cos_rs
    J_re[1, 1] = p.L_r_upper * sin_rs
    jacobians["re"] = J_re

    # -----------------------------------------------------------------------
    # 4. RH (Right Hand): from RS along right upper + forearm
    # rh_x = rs_x + L_r_upper*sin(th_rs_abs) + L_r_fore*sin(th_re_abs)
    # rh_y = rs_y - L_r_upper*cos(th_rs_abs) - L_r_fore*cos(th_re_abs)
    # Depends on q[0], q[1], q[2]
    # -----------------------------------------------------------------------
    J_rh = np.zeros((2, N_DOF))
    # d/dq[0]
    J_rh[0, 0] = (
        p.L_hub * cos_hub
        - p.d_rs * sin_hub
        + p.L_r_upper * cos_rs
        + p.L_r_fore * cos_re
    )
    J_rh[1, 0] = (
        p.L_hub * sin_hub
        + p.d_rs * cos_hub
        + p.L_r_upper * sin_rs
        + p.L_r_fore * sin_re
    )
    # d/dq[1]
    J_rh[0, 1] = p.L_r_upper * cos_rs + p.L_r_fore * cos_re
    J_rh[1, 1] = p.L_r_upper * sin_rs + p.L_r_fore * sin_re
    # d/dq[2]
    J_rh[0, 2] = p.L_r_fore * cos_re
    J_rh[1, 2] = p.L_r_fore * sin_re
    jacobians["rh"] = J_rh

    # -----------------------------------------------------------------------
    # 5. LS (Left Shoulder): similar to RS but on left side
    # ls_x = hub_x - d_ls * cos(th_hub)
    # ls_y = hub_y - d_ls * sin(th_hub)
    # -----------------------------------------------------------------------
    J_ls = np.zeros((2, N_DOF))
    J_ls[0, 0] = p.L_hub * cos_hub + p.d_ls * sin_hub
    J_ls[1, 0] = p.L_hub * sin_hub - p.d_ls * cos_hub
    jacobians["ls"] = J_ls

    # -----------------------------------------------------------------------
    # 6. LE (Left Elbow): from LS along left upper arm
    # le_x = ls_x + L_l_upper * sin(th_ls_abs)
    # le_y = ls_y - L_l_upper * cos(th_ls_abs)
    # Depends on q[0], q[4]
    # -----------------------------------------------------------------------
    J_le = np.zeros((2, N_DOF))
    # d/dq[0]
    J_le[0, 0] = p.L_hub * cos_hub + p.d_ls * sin_hub + p.L_l_upper * cos_ls
    J_le[1, 0] = p.L_hub * sin_hub - p.d_ls * cos_hub + p.L_l_upper * sin_ls
    # d/dq[4]
    J_le[0, 4] = p.L_l_upper * cos_ls
    J_le[1, 4] = p.L_l_upper * sin_ls
    jacobians["le"] = J_le

    # -----------------------------------------------------------------------
    # 7. LH (Left Hand): from LS along left upper + forearm
    # lh_x = ls_x + L_l_upper*sin(th_ls_abs) + L_l_fore*sin(th_le_abs)
    # lh_y = ls_y - L_l_upper*cos(th_ls_abs) - L_l_fore*cos(th_le_abs)
    # Depends on q[0], q[4], q[5]
    # -----------------------------------------------------------------------
    J_lh = np.zeros((2, N_DOF))
    # d/dq[0]
    J_lh[0, 0] = (
        p.L_hub * cos_hub
        + p.d_ls * sin_hub
        + p.L_l_upper * cos_ls
        + p.L_l_fore * cos_le
    )
    J_lh[1, 0] = (
        p.L_hub * sin_hub
        - p.d_ls * cos_hub
        + p.L_l_upper * sin_ls
        + p.L_l_fore * sin_le
    )
    # d/dq[4]
    J_lh[0, 4] = p.L_l_upper * cos_ls + p.L_l_fore * cos_le
    J_lh[1, 4] = p.L_l_upper * sin_ls + p.L_l_fore * sin_le
    # d/dq[5]
    J_lh[0, 5] = p.L_l_fore * cos_le
    J_lh[1, 5] = p.L_l_fore * sin_le
    jacobians["lh"] = J_lh

    # -----------------------------------------------------------------------
    # 8. Club COM: midpoint between club base and club tip
    # Using club_dx = sin(th_club), club_dy = -cos(th_club):
    # club_base_x = rh_x - grip_right * sin(th_club)
    # club_base_y = rh_y + grip_right * (-cos(th_club))
    # club_tip_x = club_base_x + L_club * sin(th_club)
    # club_tip_y = club_base_y - L_club * (-cos(th_club))
    # club_com = 0.5 * (club_base + club_tip)
    # Depends on q[0], q[1], q[2], q[3], q[7]
    # -----------------------------------------------------------------------
    # Expanded form:
    # club_com_x = rh_x + (0.5*L_club - grip_right)*sin(th_club)
    # club_com_y = rh_y + 0.5*(L_club - 2*grip_right)*cos(th_club)
    # But note: club_dy = -cos(th_club), so club_base_y = rh_y - grip_right*cos
    # Derivatives w.r.t. th_club:
    # d(club_com_x)/dth_club = (0.5*L_club - grip_right)*cos(th_club)
    # d(club_com_y)/dth_club = 0.5*(L_club - 2*grip_right)*(-sin(th_club))
    #                        = -0.5*(L_club - 2*grip_right)*sin(th_club)
    coeff_club_x = 0.5 * p.L_club - p.grip_right
    coeff_club_y = -0.5 * (p.L_club - 2 * p.grip_right)  # = -0.5*L_club + grip_right

    J_club_com = np.zeros((2, N_DOF))
    # d/dq[0], d/dq[1], d/dq[2], d/dq[3] from rh
    J_club_com[0, 0] = (
        p.L_hub * cos_hub
        - p.d_rs * sin_hub
        + p.L_r_upper * cos_rs
        + p.L_r_fore * cos_re
    )
    J_club_com[1, 0] = (
        p.L_hub * sin_hub
        + p.d_rs * cos_hub
        + p.L_r_upper * sin_rs
        + p.L_r_fore * sin_re
    )
    J_club_com[0, 1] = p.L_r_upper * cos_rs + p.L_r_fore * cos_re
    J_club_com[1, 1] = p.L_r_upper * sin_rs + p.L_r_fore * sin_re
    J_club_com[0, 2] = p.L_r_fore * cos_re
    J_club_com[1, 2] = p.L_r_fore * sin_re
    # d/dq[3]: none (right wrist angle doesn't affect hand position)
    # d/dq[7]: from club angle
    J_club_com[0, 7] = coeff_club_x * cos_club
    J_club_com[1, 7] = coeff_club_y * sin_club
    jacobians["club_com"] = J_club_com

    # -----------------------------------------------------------------------
    # 9. Club TIP
    # Using club_dx = sin(th_club), club_dy = -cos(th_club):
    # club_tip_x = club_base_x + L_club * sin(th_club)
    #            = rh_x - grip_right*sin + L_club*sin
    #            = rh_x + (L_club - grip_right)*sin
    # club_tip_y = club_base_y - L_club*(-cos(th_club))
    #            = rh_y - grip_right*cos(th_club) + L_club*cos(th_club)
    #            = rh_y + (L_club - grip_right)*cos(th_club)
    # Derivatives w.r.t. th_club:
    # d(club_tip_x)/dth_club = (L_club - grip_right)*cos(th_club)
    # d(club_tip_y)/dth_club = (L_club - grip_right)*(-sin(th_club))
    #                        = -(L_club - grip_right)*sin(th_club)
    # Depends on q[0], q[1], q[2], q[3], q[7]
    # -----------------------------------------------------------------------
    coeff_tip_x = p.L_club - p.grip_right
    coeff_tip_y = -(p.L_club - p.grip_right)

    J_club_tip = np.zeros((2, N_DOF))
    # d/dq[0], d/dq[1], d/dq[2], d/dq[3] from rh (via club_base)
    J_club_tip[0, 0] = (
        p.L_hub * cos_hub
        - p.d_rs * sin_hub
        + p.L_r_upper * cos_rs
        + p.L_r_fore * cos_re
    )
    J_club_tip[1, 0] = (
        p.L_hub * sin_hub
        + p.d_rs * cos_hub
        + p.L_r_upper * sin_rs
        + p.L_r_fore * sin_re
    )
    J_club_tip[0, 1] = p.L_r_upper * cos_rs + p.L_r_fore * cos_re
    J_club_tip[1, 1] = p.L_r_upper * sin_rs + p.L_r_fore * sin_re
    J_club_tip[0, 2] = p.L_r_fore * cos_re
    J_club_tip[1, 2] = p.L_r_fore * sin_re
    # d/dq[7]: from club direction
    J_club_tip[0, 7] = coeff_tip_x * cos_club
    J_club_tip[1, 7] = coeff_tip_y * sin_club
    jacobians["club_tip"] = J_club_tip

    return jacobians


def analytical_mass_matrix(q: np.ndarray, p: GolferParams) -> np.ndarray:
    """Compute mass matrix M(q) analytically from Jacobians.

    Uses M = sum_i(m_i * J_i^T @ J_i) where J_i is the 2×8 Jacobian
    of mass point i.

    Parameters
    ----------
    q : np.ndarray, shape (8,)
    p : GolferParams

    Returns
    -------
    M : np.ndarray, shape (8, 8) — symmetric positive semi-definite
    """
    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    jacobians = analytical_fk_jacobians(q, p)

    M = np.zeros((N_DOF, N_DOF))

    # Mass point contributions: (mass, jacobian_key)
    mass_contributions = [
        (p.m_hub, "hub"),
        (p.m_r_upper, "re"),
        (p.m_r_fore, "rh"),
        (p.m_l_upper, "le"),
        (p.m_l_fore, "lh"),
        (p.m_club, "club_com"),
        (p.m_clubhead, "club_tip"),
    ]

    for mass_val, key in mass_contributions:
        J = jacobians[key]
        M += mass_val * J.T @ J

    return M


def analytical_coriolis(q: np.ndarray, qdot: np.ndarray, p: GolferParams) -> np.ndarray:
    """Compute C(q, qdot) = (dM/dt) * qdot analytically.

    Uses Christoffel symbols: C_i = sum_jk c_ijk * qdot_j * qdot_k
    where c_ijk = 0.5 * (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i)

    dM/dq_k is computed analytically from the chain rule applied to
    each Jacobian term.

    Parameters
    ----------
    q : np.ndarray, shape (8,)
    qdot : np.ndarray, shape (8,)
    p : GolferParams

    Returns
    -------
    C_qdot : np.ndarray, shape (8,)
    """
    if q.shape[0] > N_DOF:
        q = q[:N_DOF]
    if qdot.shape[0] > N_DOF:
        qdot = qdot[:N_DOF]

    # Compute dM/dq_k analytically using finite differences of Jacobians
    # (This is faster than recomputing M multiple times)
    eps = 1e-7
    M0 = analytical_mass_matrix(q, p)
    dM = np.zeros((N_DOF, N_DOF, N_DOF))  # dM[i,j,k] = dM_ij/dq_k

    for k in range(N_DOF):
        q_plus = q.copy()
        q_plus[k] += eps
        dM[:, :, k] = (analytical_mass_matrix(q_plus, p) - M0) / eps

    # Compute Christoffel symbols and contract with qdot twice
    C_qdot = np.zeros(N_DOF)
    for i in range(N_DOF):
        for j in range(N_DOF):
            christoffel = 0.0
            for k in range(N_DOF):
                christoffel += 0.5 * (dM[i, j, k] + dM[i, k, j] - dM[j, k, i]) * qdot[k]
            C_qdot[i] += christoffel * qdot[j]

    return C_qdot


def analytical_gravity_vector(q: np.ndarray, p: GolferParams) -> np.ndarray:
    """Compute gravity torque vector G(q) analytically.

    G_i = dV/dq_i where V = sum_k(m_k * g * y_k)

    For each mass point, dV/dq_i = m * g * dy/dq_i.

    Parameters
    ----------
    q : np.ndarray, shape (8,)
    p : GolferParams

    Returns
    -------
    G : np.ndarray, shape (8,)
    """
    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    jacobians = analytical_fk_jacobians(q, p)

    G = np.zeros(N_DOF)

    mass_contributions = [
        (p.m_hub, "hub"),
        (p.m_r_upper, "re"),
        (p.m_r_fore, "rh"),
        (p.m_l_upper, "le"),
        (p.m_l_fore, "lh"),
        (p.m_club, "club_com"),
        (p.m_clubhead, "club_tip"),
    ]

    for mass_val, key in mass_contributions:
        J = jacobians[key]
        # G_i += m * g * dy/dq_i = m * g * J[1, i]
        G += mass_val * p.g * J[1, :]

    return G


def analytical_constraint_jacobian(q: np.ndarray, p: GolferParams) -> np.ndarray:
    """Compute constraint Jacobian Phi_q analytically.

    The constraints enforce that the left hand matches the left grip position
    on the club, and that the relative positions satisfy the club geometry.

    Parameters
    ----------
    q : np.ndarray, shape (8,)
    p : GolferParams

    Returns
    -------
    Phi_q : np.ndarray, shape (4, 8)
    """
    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    jacobians = analytical_fk_jacobians(q, p)
    J_lh = jacobians["lh"]
    J_rh = jacobians["rh"]

    th_club = q[7]
    sin_club = np.sin(th_club)
    cos_club = np.cos(th_club)

    # Club direction and perpendicular vectors
    club_dir = np.array([sin_club, -cos_club])
    club_perp = np.array([-(-cos_club), sin_club])  # rotated 90° ccw

    # Constraint 1-2: LH position matches grip_left on club
    Phi_q = np.zeros((4, N_DOF))

    # dPhi[0]/dq = dLH_x/dq - d(club_base_x + grip_left*sin(th_club))/dq
    # dPhi[1]/dq = dLH_y/dq - d(club_base_y - grip_left*cos(th_club))/dq
    Phi_q[0, :] = J_lh[0, :]
    Phi_q[1, :] = J_lh[1, :]

    # Subtract RH-dependent part
    Phi_q[0, :] -= J_rh[0, :]
    Phi_q[1, :] -= J_rh[1, :]

    # d(grip_left * sin(th_club))/dq_7 = grip_left * cos(th_club)
    # d(-grip_left * cos(th_club))/dq_7 = grip_left * sin(th_club)
    Phi_q[0, 7] -= p.grip_left * cos_club
    Phi_q[1, 7] += p.grip_left * sin_club

    # Constraint 3: perpendicular distance = 0
    # Phi[2] = (lh - rh) · club_perp = lh_x * club_perp_x + lh_y * club_perp_y
    #                                 - rh_x * club_perp_x - rh_y * club_perp_y
    Phi_q[2, :] = club_perp[0] * (J_lh[0, :] - J_rh[0, :]) + club_perp[1] * (
        J_lh[1, :] - J_rh[1, :]
    )

    # d(club_perp)/dq_7: club_perp = (-(-cos(th_club)), sin(th_club))
    #                                 = (cos(th_club), sin(th_club))
    # d/dq_7: (-sin(th_club), cos(th_club))
    d_club_perp_dth = np.array([-sin_club, cos_club])
    rh_array = np.zeros(2)
    # Evaluate RH position from jacobian: rh = rh_0 + J_rh @ q (but we need pos)
    fk = forward_kinematics(q, p)
    rh_array[0] = fk["rh"][0]
    rh_array[1] = fk["rh"][1]
    lh_array = np.zeros(2)
    lh_array[0] = fk["lh"][0]
    lh_array[1] = fk["lh"][1]

    rh_to_lh = lh_array - rh_array
    Phi_q[2, 7] += np.dot(rh_to_lh, d_club_perp_dth)

    # Constraint 4: along-club distance = grip_sep
    # Phi[3] = (lh - rh) · club_dir - (grip_left - grip_right)
    Phi_q[3, :] = club_dir[0] * (J_lh[0, :] - J_rh[0, :]) + club_dir[1] * (
        J_lh[1, :] - J_rh[1, :]
    )

    # d(club_dir)/dq_7: club_dir = (sin(th_club), -cos(th_club))
    # d/dq_7: (cos(th_club), sin(th_club))
    d_club_dir_dth = np.array([cos_club, sin_club])
    Phi_q[3, 7] += np.dot(rh_to_lh, d_club_dir_dth)

    return Phi_q


# ---------------------------------------------------------------------------
# Switch to analytical versions as the default
# ---------------------------------------------------------------------------
# These aliases replace the old numerical functions with analytical ones.
# All analytical functions are now defined above, so we can safely create
# these aliases.
constraint_jacobian = analytical_constraint_jacobian
mass_matrix = analytical_mass_matrix
coriolis_matrix = analytical_coriolis
gravity_vector = analytical_gravity_vector
