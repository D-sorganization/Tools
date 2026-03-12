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

from . import native_backend as _native_backend

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

    # Scapula joint parameters (#1104)
    # Optional scapula links between hub bar and shoulder joints.
    # When L_rscap/L_lscap = 0, the scapula is absent (backwards compatible).
    L_rscap: float = 0.0  # right scapula link length
    L_lscap: float = 0.0  # left scapula link length
    m_rscap: float = 0.0  # right scapula mass
    m_lscap: float = 0.0  # left scapula mass
    b_rscap: float = 0.0  # right scapula damping
    b_lscap: float = 0.0  # left scapula damping

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


# Re-export all public APIs for backward compatibility
from .golfer_kinematics import forward_kinematics  # noqa: F401, E402
from .golfer_dynamics import (  # noqa: F401, E402
    analytical_fk_jacobians,
    analytical_mass_matrix,
    analytical_coriolis,
    analytical_gravity_vector,
    kinetic_energy,
    potential_energy,
    total_energy,
    potential_energy_from_q,
)
from .golfer_constraints import (  # noqa: F401, E402
    constraint_vector,
    numerical_constraint_jacobian,
    analytical_constraint_jacobian,
    linear_accelerations,
    net_joint_forces,
    friction_torque_vector,
)

# ---------------------------------------------------------------------------
# Default function aliases for backward compatibility
# ---------------------------------------------------------------------------
# These use the analytical versions defined in the sub-modules
constraint_jacobian = analytical_constraint_jacobian  # noqa: F405
mass_matrix = analytical_mass_matrix  # noqa: F405
coriolis_matrix = analytical_coriolis  # noqa: F405
gravity_vector = analytical_gravity_vector  # noqa: F405


def get_native_backend_info() -> dict[str, object]:
    """Expose the golfer native-backend configuration and availability."""
    return _native_backend.get_native_backend_info()
