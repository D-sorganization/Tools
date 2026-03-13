"""Dynamics (mass matrix, Coriolis, gravity, energy) for golfer model.

Uses Lagrangian formulation with analytical Jacobian-based computation
and optional native backend acceleration.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import native_backend as _native_backend
from .golfer_kinematics import forward_kinematics
from .physics_golfer import GolferParams, N_DOF, State


def _mass_point_positions(
    q: np.ndarray, p: GolferParams
) -> list[tuple[float, Callable]]:
    """Return list of (mass, position_function) for all point masses."""
    if not isinstance(q, np.ndarray):
        raise TypeError("q must be a numpy ndarray")
    if q.ndim != 1 or q.shape[0] < N_DOF:
        raise ValueError(f"q must be a 1D array of at least {N_DOF} elements")
    if not isinstance(p, GolferParams):
        raise TypeError("p must be a GolferParams instance")

    def hub_pos(qq: np.ndarray) -> tuple[float, float]:
        from .golfer_kinematics import _hub_position

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


def potential_energy_from_q(q: np.ndarray, p: GolferParams) -> float:
    """Compute total gravitational potential energy from coordinates."""
    if not isinstance(q, np.ndarray):
        raise TypeError("q must be a numpy ndarray")
    if q.ndim != 1 or q.shape[0] < N_DOF:
        raise ValueError(f"q must be a 1D array of at least {N_DOF} elements")
    if not isinstance(p, GolferParams):
        raise TypeError("p must be a GolferParams instance")

    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    mass_points = _mass_point_positions(q, p)
    V = 0.0
    for mass_val, pos_func in mass_points:
        _, y = pos_func(q)
        V += mass_val * p.g * y

    return V


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
    if not isinstance(q, np.ndarray):
        raise TypeError("q must be a numpy ndarray")
    if q.ndim != 1 or q.shape[0] < N_DOF:
        raise ValueError(f"q must be a 1D array of at least {N_DOF} elements")
    if not isinstance(p, GolferParams):
        raise TypeError("p must be a GolferParams instance")

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
    # 1. HUB: position = (-L_hub * sin(th_hub), L_hub * cos(th_hub))  (#1103)
    # Depends only on q[0]
    # -----------------------------------------------------------------------
    J_hub = np.zeros((2, N_DOF))
    J_hub[0, 0] = -p.L_hub * cos_hub
    J_hub[1, 0] = -p.L_hub * sin_hub
    jacobians["hub"] = J_hub

    # -----------------------------------------------------------------------
    # 2. RS (Right Shoulder): position from hub + perpendicular offset
    # rs_x = hub_x + d_rs * cos(th_hub)
    # rs_y = hub_y + d_rs * sin(th_hub)
    # -----------------------------------------------------------------------
    J_rs = np.zeros((2, N_DOF))
    # d/dq[0]: d(hub_x)/dq[0] + d(d_rs*cos)/dq[0)  (#1103 reversed hub)
    J_rs[0, 0] = -p.L_hub * cos_hub - p.d_rs * sin_hub
    J_rs[1, 0] = -p.L_hub * sin_hub + p.d_rs * cos_hub
    jacobians["rs"] = J_rs

    # -----------------------------------------------------------------------
    # 3. RE (Right Elbow): from RS along right upper arm
    # re_x = rs_x + L_r_upper * sin(th_rs_abs)
    # re_y = rs_y - L_r_upper * cos(th_rs_abs)
    # Depends on q[0], q[1]
    # -----------------------------------------------------------------------
    J_re = np.zeros((2, N_DOF))
    # d/dq[0]: d(rs)/dq[0] + d(L_r_upper*sin(th_rs_abs))/dq[0]  (#1103)
    J_re[0, 0] = -p.L_hub * cos_hub - p.d_rs * sin_hub + p.L_r_upper * cos_rs
    J_re[1, 0] = -p.L_hub * sin_hub + p.d_rs * cos_hub + p.L_r_upper * sin_rs
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
    # d/dq[0]  (#1103 reversed hub)
    J_rh[0, 0] = (
        -p.L_hub * cos_hub
        - p.d_rs * sin_hub
        + p.L_r_upper * cos_rs
        + p.L_r_fore * cos_re
    )
    J_rh[1, 0] = (
        -p.L_hub * sin_hub
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
    J_ls[0, 0] = -p.L_hub * cos_hub + p.d_ls * sin_hub  # (#1103)
    J_ls[1, 0] = -p.L_hub * sin_hub - p.d_ls * cos_hub
    jacobians["ls"] = J_ls

    # -----------------------------------------------------------------------
    # 6. LE (Left Elbow): from LS along left upper arm
    # le_x = ls_x + L_l_upper * sin(th_ls_abs)
    # le_y = ls_y - L_l_upper * cos(th_ls_abs)
    # Depends on q[0], q[4]
    # -----------------------------------------------------------------------
    J_le = np.zeros((2, N_DOF))
    # d/dq[0]  (#1103)
    J_le[0, 0] = -p.L_hub * cos_hub + p.d_ls * sin_hub + p.L_l_upper * cos_ls
    J_le[1, 0] = -p.L_hub * sin_hub - p.d_ls * cos_hub + p.L_l_upper * sin_ls
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
    # d/dq[0]  (#1103)
    J_lh[0, 0] = (
        -p.L_hub * cos_hub
        + p.d_ls * sin_hub
        + p.L_l_upper * cos_ls
        + p.L_l_fore * cos_le
    )
    J_lh[1, 0] = (
        -p.L_hub * sin_hub
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
    # d/dq[0], d/dq[1], d/dq[2], d/dq[3] from rh  (#1103 reversed hub)
    J_club_com[0, 0] = (
        -p.L_hub * cos_hub
        - p.d_rs * sin_hub
        + p.L_r_upper * cos_rs
        + p.L_r_fore * cos_re
    )
    J_club_com[1, 0] = (
        -p.L_hub * sin_hub
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
    # d/dq[0], d/dq[1], d/dq[2], d/dq[3] from rh (via club_base)  (#1103)
    J_club_tip[0, 0] = (
        -p.L_hub * cos_hub
        - p.d_rs * sin_hub
        + p.L_r_upper * cos_rs
        + p.L_r_fore * cos_re
    )
    J_club_tip[1, 0] = (
        -p.L_hub * sin_hub
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
    native_mass_matrix = _native_backend.golfer_mass_matrix(q, p)
    if native_mass_matrix is not None:
        return native_mass_matrix

    if not isinstance(q, np.ndarray):
        raise TypeError("q must be a numpy ndarray")
    if q.ndim != 1 or q.shape[0] < N_DOF:
        raise ValueError(f"q must be a 1D array of at least {N_DOF} elements")
    if not isinstance(p, GolferParams):
        raise TypeError("p must be a GolferParams instance")

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
    if not isinstance(q, np.ndarray) or not isinstance(qdot, np.ndarray):
        raise TypeError("q and qdot must be numpy ndarrays")
    if q.ndim != 1 or q.shape[0] < N_DOF:
        raise ValueError(f"q must be a 1D array of at least {N_DOF} elements")
    if qdot.ndim != 1 or qdot.shape[0] < N_DOF:
        raise ValueError(f"qdot must be a 1D array of at least {N_DOF} elements")
    if not isinstance(p, GolferParams):
        raise TypeError("p must be a GolferParams instance")

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

    christoffel = 0.5 * (dM + dM.transpose(0, 2, 1) - dM.transpose(1, 2, 0))
    result: np.ndarray = np.einsum("ijk,j,k->i", christoffel, qdot, qdot)
    return result


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
    native_gravity = _native_backend.golfer_gravity_vector(q, p)
    if native_gravity is not None:
        return native_gravity

    if not isinstance(q, np.ndarray):
        raise TypeError("q must be a numpy ndarray")
    if q.ndim != 1 or q.shape[0] < N_DOF:
        raise ValueError(f"q must be a 1D array of at least {N_DOF} elements")
    if not isinstance(p, GolferParams):
        raise TypeError("p must be a GolferParams instance")

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


def kinetic_energy(q: np.ndarray, qdot: np.ndarray, p: GolferParams) -> float:
    """Compute T = 0.5 * qdot^T M qdot."""
    from .physics_base import kinetic_energy_from_M

    if not isinstance(q, np.ndarray) or not isinstance(qdot, np.ndarray):
        raise TypeError("q and qdot must be numpy ndarrays")
    if q.ndim != 1 or q.shape[0] < N_DOF:
        raise ValueError(f"q must be a 1D array of at least {N_DOF} elements")
    if qdot.ndim != 1 or qdot.shape[0] < N_DOF:
        raise ValueError(f"qdot must be a 1D array of at least {N_DOF} elements")
    if not isinstance(p, GolferParams):
        raise TypeError("p must be a GolferParams instance")

    if q.shape[0] > N_DOF:
        q = q[:N_DOF]
    if qdot.shape[0] > N_DOF:
        qdot = qdot[:N_DOF]

    M = analytical_mass_matrix(q, p)
    return kinetic_energy_from_M(M, qdot)


def potential_energy(state: State, p: GolferParams) -> float:
    """Compute gravitational PE from full state vector."""
    return potential_energy_from_q(state[:N_DOF], p)


def total_energy(state: State, p: GolferParams) -> float:
    """Compute E = T + V from full state."""
    from .physics_base import total_energy_from_parts

    q = state[:N_DOF]
    qdot = state[N_DOF:]
    return total_energy_from_parts(
        kinetic_energy(q, qdot, p), potential_energy(state, p)
    )
